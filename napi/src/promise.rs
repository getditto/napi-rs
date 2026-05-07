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
  let env_tearing_down = future_promise.env_tearing_down.load(Ordering::Acquire);
  let cleanup_hook_data = future_promise.cleanup_hook_data;

  if env_tearing_down {
    // The environment is tearing down. Calling `napi_resolve_deferred` /
    // `napi_reject_deferred` would crash inside V8's `GlobalHandles::Destroy`
    // because the `Persistent` backing the `Deferred` may already be invalidated.
    //
    // Drop the future_promise and value to free Rust-side state, but skip every
    // N-API cleanup call that would touch the partially-torn-down environment.
    // The cleanup hook data is freed by the hook itself during teardown.
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
}
