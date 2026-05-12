use std::future::Future;
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{check_status, sys, Env, JsError, NapiValue, Result};

pub struct FuturePromise<T, V: NapiValue> {
  deferred: sys::napi_deferred,
  env: sys::napi_env,
  tsfn: sys::napi_threadsafe_function,
  async_resource_name: sys::napi_value,
  resolver: Box<dyn FnOnce(&mut Env, T) -> Result<V>>,
  /// Set to `true` when the N-API environment begins teardown. When this is
  /// true, calling `napi_resolve_deferred` / `napi_reject_deferred` can crash
  /// in Node.js >= 22 because the V8 `Persistent` backing the `Deferred` may
  /// already be invalidated (SIGSEGV in `GlobalHandles::Destroy`).
  env_tearing_down: Arc<AtomicBool>,
  /// Raw pointer to the cleanup hook data, needed to unregister the hook on
  /// normal completion. Null if the hook has already been unregistered.
  cleanup_hook_data: *mut c_void,
  /// DEVX-877: a strong (`refcount=1`) `napi_ref` to the JSPromise returned
  /// to JS. Pins the promise so JS GC can't collect it before `call_js_cb`
  /// resolves the deferred. Without this, an internal SDK call whose
  /// returned promise is unreferenced from JS lets the JSPromise (and the
  /// deferred's backing v8 Persistent) be GC'd; when `napi_resolve_deferred`
  /// later runs, it crashes inside V8's `GlobalHandles::Destroy` walking a
  /// freed slot. Released in `call_js_cb` after resolve/reject.
  promise_ref: sys::napi_ref,
}

unsafe impl<T, V: NapiValue> Send for FuturePromise<T, V> {}

struct CleanupHookData {
  flag: Arc<AtomicBool>,
}

unsafe extern "C" fn env_teardown_cleanup(data: *mut c_void) {
  let hook_data = Box::from_raw(data as *mut CleanupHookData);
  hook_data.flag.store(true, Ordering::Release);
}

impl<T, V: NapiValue> FuturePromise<T, V> {
  #[inline]
  pub fn create(
    env: sys::napi_env,
    raw_deferred: sys::napi_deferred,
    raw_promise: sys::napi_value,
    resolver: Box<dyn FnOnce(&mut Env, T) -> Result<V>>,
  ) -> Result<Self> {
    let mut async_resource_name = ptr::null_mut();
    let s = "napi_resolve_promise_from_future";
    check_status!(unsafe {
      sys::napi_create_string_utf8(
        env,
        s.as_ptr() as *const c_char,
        s.len(),
        &mut async_resource_name,
      )
    })?;

    // DEVX-877: pin the JSPromise with an initial-refcount-1 napi_ref so JS GC
    // can't collect it (and with it, the deferred's backing v8 Persistent)
    // before we get a chance to resolve. Released in `call_js_cb`.
    let mut promise_ref = ptr::null_mut();
    check_status!(unsafe { sys::napi_create_reference(env, raw_promise, 1, &mut promise_ref) })?;

    let env_tearing_down = Arc::new(AtomicBool::new(false));
    let hook_data = Box::into_raw(Box::new(CleanupHookData {
      flag: Arc::clone(&env_tearing_down),
    }));
    unsafe {
      sys::napi_add_env_cleanup_hook(env, Some(env_teardown_cleanup), hook_data as *mut c_void);
    }

    Ok(FuturePromise {
      deferred: raw_deferred,
      resolver,
      env,
      tsfn: ptr::null_mut(),
      async_resource_name,
      env_tearing_down,
      cleanup_hook_data: hook_data as *mut c_void,
      promise_ref,
    })
  }

  #[inline]
  pub(crate) fn start(self) -> Result<TSFNValue> {
    let mut tsfn_value = ptr::null_mut();
    let async_resource_name = self.async_resource_name;
    let env = self.env;
    let self_ref = Box::leak(Box::from(self));
    check_status!(unsafe {
      sys::napi_create_threadsafe_function(
        env,
        ptr::null_mut(),
        ptr::null_mut(),
        async_resource_name,
        0,
        1,
        ptr::null_mut(),
        None,
        self_ref as *mut _ as *mut c_void,
        Some(call_js_cb::<T, V>),
        &mut tsfn_value,
      )
    })?;
    self_ref.tsfn = tsfn_value;
    Ok(TSFNValue(tsfn_value))
  }
}

pub(crate) struct TSFNValue(sys::napi_threadsafe_function);

unsafe impl Send for TSFNValue {}

#[inline(always)]
pub(crate) async fn resolve_from_future<T: Send, F: Future<Output = Result<T>>>(
  tsfn_value: TSFNValue,
  fut: F,
) {
  let val = fut.await;
  check_status!(unsafe {
    sys::napi_call_threadsafe_function(
      tsfn_value.0,
      Box::into_raw(Box::from(val)) as *mut _ as *mut c_void,
      sys::napi_threadsafe_function_call_mode::napi_tsfn_nonblocking,
    )
  })
  .expect("Failed to call thread safe function");
  check_status!(unsafe {
    sys::napi_release_threadsafe_function(
      tsfn_value.0,
      sys::napi_threadsafe_function_release_mode::napi_tsfn_release,
    )
  })
  .expect("Failed to release thread safe function");
}

unsafe extern "C" fn call_js_cb<T, V: NapiValue>(
  raw_env: sys::napi_env,
  _js_callback: sys::napi_value,
  context: *mut c_void,
  data: *mut c_void,
) {
  let future_promise = Box::from_raw(context as *mut FuturePromise<T, V>);
  // DEVX-877: also consult the process-wide shutdown signal — fires earlier than
  // env-cleanup-hooks, which run after v8 has begun finalizing GlobalHandles.
  let env_tearing_down = future_promise.env_tearing_down.load(Ordering::Acquire)
    || crate::lifecycle::shutdown_requested();
  let cleanup_hook_data = future_promise.cleanup_hook_data;
  let promise_ref = future_promise.promise_ref;

  if env_tearing_down {
    // The environment is tearing down. Calling `napi_resolve_deferred` /
    // `napi_reject_deferred` would crash inside V8's `GlobalHandles::Destroy`
    // because the `Persistent` backing the `Deferred` may already be invalidated.
    //
    // Drop the future_promise and value to free Rust-side state, but skip every
    // N-API cleanup call that would touch the partially-torn-down environment.
    // The cleanup hook data is freed by the hook itself during teardown.
    // The promise_ref is intentionally leaked: V8's GlobalHandle table is being
    // torn down anyway, and `napi_delete_reference` would have the same crash
    // window as `napi_resolve_deferred`. One bounded leak per teardown is OK.
    let value: Result<T> = ptr::read(data as *const _);
    drop(value);
    drop(future_promise);
    return;
  }

  // Normal completion: unregister the cleanup hook since we no longer need it.
  if !cleanup_hook_data.is_null() {
    sys::napi_remove_env_cleanup_hook(raw_env, Some(env_teardown_cleanup), cleanup_hook_data);
    drop(Box::from_raw(cleanup_hook_data as *mut CleanupHookData));
  }

  let mut env = Env::from_raw(raw_env);
  let value: Result<T> = ptr::read(data as *const _);
  let resolver = future_promise.resolver;
  let deferred = future_promise.deferred;
  let js_value_to_resolve = value.and_then(move |v| (resolver)(&mut env, v));
  match js_value_to_resolve {
    Ok(v) => {
      let status = sys::napi_resolve_deferred(raw_env, deferred, v.raw());
      debug_assert!(status == sys::Status::napi_ok, "Resolve promise failed");
    }
    Err(e) => {
      let status =
        sys::napi_reject_deferred(raw_env, deferred, JsError::from(e).into_value(raw_env));
      debug_assert!(status == sys::Status::napi_ok, "Reject promise failed");
    }
  };

  // DEVX-877: release the strong ref we took on the JSPromise in
  // `FuturePromise::create`. The promise has now been resolved or rejected, so
  // JS-land already saw a settled state and ordinary GC can collect the
  // promise object whenever it stops being referenced from JS.
  //
  // Use `napi_reference_unref` (drops refcount → 0) rather than
  // `napi_delete_reference` (destroys the Reference object and its v8
  // Persistent slot). The latter has triggered V8_Fatal in
  // `GlobalHandles::Destroy` from the parallel async_work pin under
  // teardown-window conditions; the same pattern could surface here for
  // long-lived futures. With unref, V8's weak-handle path clears the slot
  // during normal GC instead, and node's env teardown destroys the (now
  // empty) Reference object safely. Small bounded leak (one Reference per
  // FuturePromise call) until env teardown.
  if !promise_ref.is_null() {
    let mut new_refcount = 0u32;
    let status = sys::napi_reference_unref(raw_env, promise_ref, &mut new_refcount);
    debug_assert!(
      status == sys::Status::napi_ok,
      "Unref promise reference failed"
    );
  }
}
