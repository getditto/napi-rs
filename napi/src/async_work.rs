use std::mem;
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{check_status, js_values::NapiValue, sys, Env, JsError, JsObject, Result, Task};

struct AsyncWork<T: Task> {
  inner_task: T,
  deferred: sys::napi_deferred,
  value: Result<mem::MaybeUninit<T::Output>>,
  napi_async_work: sys::napi_async_work,
  /// Set to `true` when the N-API environment begins teardown. When this is
  /// true, calling `napi_delete_async_work` can crash in Node.js >= 22 because
  /// the internal `AsyncResource` may already be partially torn down.
  env_tearing_down: Arc<AtomicBool>,
  /// Raw pointer to the cleanup hook data, needed to unregister the hook on
  /// normal completion. Null if the hook has already been unregistered.
  cleanup_hook_data: *mut c_void,
  /// DEVX-877: a strong (`refcount=1`) `napi_ref` to the JSPromise returned to
  /// JS. Mirrors the pin in `FuturePromise::create`. Without this, an internal
  /// SDK call whose returned promise is unreferenced from JS lets the
  /// JSPromise (and the deferred's backing v8 Persistent) be GC'd; when
  /// `napi_resolve_deferred` later runs in `complete`, it crashes inside V8's
  /// `GlobalHandles::Destroy` walking a freed slot. Released in `complete`
  /// after resolve/reject.
  promise_ref: sys::napi_ref,
}

pub struct AsyncWorkPromise<'env> {
  napi_async_work: sys::napi_async_work,
  raw_promise: sys::napi_value,
  env: &'env Env,
}

impl<'env> AsyncWorkPromise<'env> {
  #[inline]
  pub fn promise_object(&self) -> JsObject {
    unsafe { JsObject::from_raw_unchecked(self.env.0, self.raw_promise) }
  }

  #[inline]
  pub fn cancel(self) -> Result<()> {
    check_status!(unsafe { sys::napi_cancel_async_work(self.env.0, self.napi_async_work) })
  }
}

struct CleanupHookData {
  flag: Arc<AtomicBool>,
}

unsafe extern "C" fn env_teardown_cleanup(data: *mut c_void) {
  let hook_data = Box::from_raw(data as *mut CleanupHookData);
  hook_data.flag.store(true, Ordering::Release);
}

#[inline]
pub fn run<T: Task>(env: &Env, task: T) -> Result<AsyncWorkPromise<'_>> {
  let mut raw_resource = ptr::null_mut();
  check_status!(unsafe { sys::napi_create_object(env.0, &mut raw_resource) })?;
  let mut raw_promise = ptr::null_mut();
  let mut deferred = ptr::null_mut();
  check_status!(unsafe { sys::napi_create_promise(env.0, &mut deferred, &mut raw_promise) })?;
  let mut raw_name = ptr::null_mut();
  let s = "napi_rs_async_work";
  check_status!(unsafe {
    sys::napi_create_string_utf8(env.0, s.as_ptr() as *const c_char, s.len(), &mut raw_name)
  })?;

  // DEVX-877: pin the JSPromise with an initial-refcount-1 napi_ref so JS GC
  // can't collect it (and with it, the deferred's backing v8 Persistent)
  // before `complete` resolves the deferred. Mirrors the pin in
  // `FuturePromise::create`. Released in `complete` after resolve/reject.
  let mut promise_ref = ptr::null_mut();
  check_status!(unsafe { sys::napi_create_reference(env.0, raw_promise, 1, &mut promise_ref) })?;
  // DEVX-877 instrumentation: log every Reference allocation so we can
  // cross-reference with the V8 NodeSpace::Free double-free log to identify
  // which Reference is being destroyed twice.
  eprintln!(
    "[DEVX-877] napi_create_reference async_work env={:p} promise_ref={:p} raw_promise={:p}",
    env.0, promise_ref, raw_promise
  );

  // Create a shared flag that the env cleanup hook will set when teardown begins.
  let env_tearing_down = Arc::new(AtomicBool::new(false));
  let hook_data = Box::into_raw(Box::new(CleanupHookData {
    flag: Arc::clone(&env_tearing_down),
  }));
  unsafe {
    sys::napi_add_env_cleanup_hook(env.0, Some(env_teardown_cleanup), hook_data as *mut c_void);
  }

  let result = Box::leak(Box::new(AsyncWork {
    inner_task: task,
    deferred,
    value: Ok(mem::MaybeUninit::zeroed()),
    napi_async_work: ptr::null_mut(),
    env_tearing_down,
    cleanup_hook_data: hook_data as *mut c_void,
    promise_ref,
  }));
  check_status!(unsafe {
    sys::napi_create_async_work(
      env.0,
      raw_resource,
      raw_name,
      Some(execute::<T> as unsafe extern "C" fn(env: sys::napi_env, data: *mut c_void)),
      Some(
        complete::<T>
          as unsafe extern "C" fn(env: sys::napi_env, status: sys::napi_status, data: *mut c_void),
      ),
      result as *mut _ as *mut c_void,
      &mut result.napi_async_work,
    )
  })?;
  check_status!(unsafe { sys::napi_queue_async_work(env.0, result.napi_async_work) })?;
  Ok(AsyncWorkPromise {
    napi_async_work: result.napi_async_work,
    raw_promise,
    env,
  })
}

unsafe impl<T: Task> Send for AsyncWork<T> {}

unsafe impl<T: Task> Sync for AsyncWork<T> {}

/// env here is the same with the one in `CallContext`.
/// So it actually could do nothing here, because `execute` function is called in the other thread mostly.
unsafe extern "C" fn execute<T: Task>(_env: sys::napi_env, data: *mut c_void) {
  let mut work = Box::from_raw(data as *mut AsyncWork<T>);
  let _ = mem::replace(
    &mut work.value,
    work.inner_task.compute().map(mem::MaybeUninit::new),
  );
  Box::leak(work);
}

unsafe extern "C" fn complete<T: Task>(
  env: sys::napi_env,
  status: sys::napi_status,
  data: *mut c_void,
) {
  let mut work = Box::from_raw(data as *mut AsyncWork<T>);
  let napi_async_work = mem::replace(&mut work.napi_async_work, ptr::null_mut());
  // DEVX-877: in addition to the per-resource env-cleanup-hook flag, also consult the
  // process-wide shutdown signal. The latter is set from JS (`process.on('beforeExit')`)
  // and fires earlier than env-cleanup-hooks, which on Node.js >= 22 / arm64 run after
  // v8 has already started finalizing GlobalHandles — too late to safely resolve a
  // deferred or delete the async work.
  let env_tearing_down =
    work.env_tearing_down.load(Ordering::Acquire) || crate::lifecycle::shutdown_requested();
  let cleanup_hook_data = mem::replace(&mut work.cleanup_hook_data, ptr::null_mut());
  let promise_ref = mem::replace(&mut work.promise_ref, ptr::null_mut());

  if status == sys::Status::napi_cancelled || env_tearing_down {
    // The work was cancelled (e.g., during Node.js shutdown) or the environment
    // is tearing down. Attempting to resolve/reject the deferred or delete the
    // async work can crash in Node.js >= 22 due to stricter V8 checks on
    // AsyncResource lifecycle (Check failed: node->IsInUse()).
    //
    // Drop `work` to free the Rust-side resources, but skip N-API cleanup calls
    // that would touch the partially-torn-down environment.
    //
    // The cleanup hook data is freed by the hook itself during teardown, or leaked
    // if we got napi_cancelled without teardown (acceptable — it's a small allocation
    // and this only happens during process exit).
    //
    // The promise_ref is intentionally leaked: V8's GlobalHandle table is being
    // torn down anyway, and `napi_delete_reference` would have the same crash
    // window as `napi_resolve_deferred`. One bounded leak per teardown is OK.
    eprintln!(
      "[DEVX-877] async_work teardown leak: env={:p} promise_ref={:p} env_tearing_down={}",
      env, promise_ref, env_tearing_down
    );
    let _ = promise_ref;
    drop(work);
    return;
  }

  // Normal completion: unregister the cleanup hook since we no longer need it.
  if !cleanup_hook_data.is_null() {
    sys::napi_remove_env_cleanup_hook(env, Some(env_teardown_cleanup), cleanup_hook_data);
    // Free the hook data since the hook won't fire now.
    drop(Box::from_raw(cleanup_hook_data as *mut CleanupHookData));
  }

  let value_ptr = mem::replace(&mut work.value, Ok(mem::MaybeUninit::zeroed()));
  let deferred = mem::replace(&mut work.deferred, ptr::null_mut());
  let value = match value_ptr {
    Ok(v) => {
      let output = v.assume_init();
      work.inner_task.resolve(Env::from_raw(env), output)
    }
    Err(e) => work.inner_task.reject(Env::from_raw(env), e),
  };
  match check_status!(status).and_then(move |_| value) {
    Ok(v) => {
      let status = sys::napi_resolve_deferred(env, deferred, v.raw());
      debug_assert!(status == sys::Status::napi_ok, "Resolve promise failed");
    }
    Err(e) => {
      let status = sys::napi_reject_deferred(env, deferred, JsError::from(e).into_value(env));
      debug_assert!(status == sys::Status::napi_ok, "Reject promise failed");
    }
  };
  let delete_status = sys::napi_delete_async_work(env, napi_async_work);
  debug_assert!(
    delete_status == sys::Status::napi_ok,
    "Delete async work failed"
  );

  // DEVX-877: release the strong ref we took on the JSPromise in `run`. The
  // promise has now been resolved or rejected, so JS-land already saw a settled
  // state and ordinary GC can collect the promise object whenever it stops
  // being referenced from JS.
  //
  // We use `napi_reference_unref` (drops refcount → 0) rather than
  // `napi_delete_reference` (destroys the Reference object and its v8
  // Persistent slot). The latter triggered V8_Fatal in
  // `GlobalHandles::Destroy` on iter 38 of the first validation run for
  // `ditto_logger_set_custom_log_cb`, because by the time `complete` fires
  // for a long-lived async_work, the env may already be transiently
  // tearing down state that makes explicit slot destruction unsafe — the
  // same teardown window the env-tearing-down check above guards the
  // resolve against, but our cleanup hook is unregistered before we get
  // here so we can't re-check it. With unref, V8's weak-handle path
  // clears the slot during normal GC instead, and node's env teardown
  // destroys the (now empty) Reference object safely. Small bounded leak
  // (one Reference per async_work call) until env teardown.
  if !promise_ref.is_null() {
    let mut new_refcount = 0u32;
    let ref_status = sys::napi_reference_unref(env, promise_ref, &mut new_refcount);
    // DEVX-877 instrumentation
    eprintln!(
      "[DEVX-877] napi_reference_unref async_work env={:p} promise_ref={:p} new_refcount={} status={}",
      env, promise_ref, new_refcount, ref_status as i32
    );
    debug_assert!(
      ref_status == sys::Status::napi_ok,
      "Unref async_work promise reference failed"
    );
  }
}
