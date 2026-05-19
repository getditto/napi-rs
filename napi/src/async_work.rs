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
    {
      use std::io::Write as _;
      let _ = writeln!(
        std::io::stderr(),
        "[DEVX-877] async_work teardown leak: env={:p} promise_ref={:p} env_tearing_down={}",
        env, promise_ref, env_tearing_down
      );
    }
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

  // DEVX-877: leak the strong ref we took on the JSPromise in `run`.
  //
  // We tried two things and both crash. `napi_delete_reference` triggers
  // V8_Fatal in `GlobalHandles::Destroy` during teardown windows.
  // `napi_reference_unref` (the "safer" alternative that drops refcount
  // to 0 and lets V8's weak-callback path clean up) triggers V8_Fatal in
  // `GlobalHandles::MakeWeak`, hitting V8's
  // `CHECK_NE(object_, kGlobalHandleZapValue)` because by the time we
  // reach this point the slot has already been zapped — still unclear
  // by what. The instrumented patched-node run (20060e/3b35d87)
  // captured the MakeWeak stack: `napi::async_work::complete` →
  // `napi_reference_unref` → `GlobalHandles::MakeWeak` → `V8_Fatal`,
  // mid-flight in `uv__work_done` (not during teardown).
  //
  // So: do nothing. The Reference is bounded (one per async_work call)
  // and is destroyed at napi env teardown via the env's `reflist` walk
  // calling `~Reference` → `persistent_.Reset()` →
  // `GlobalHandles::Destroy` → `NodeSpace::Free`. If the slot has
  // already been freed by V8 internals, the patched-node experimental
  // V8 patch absorbs the double-free silently; on stock node it would
  // CHECK-abort (Class A), which is the original DEVX-877 crash.
  let _ = promise_ref;
}
