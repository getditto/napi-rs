use std::convert::Into;
use std::ffi::CString;
use std::marker::PhantomData;
use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::Arc;

use crate::{check_status, sys, Env, Error, JsError, JsFunction, NapiValue, Result, Status};

use sys::napi_threadsafe_function_call_mode;

/// ThreadSafeFunction Context object
/// the `value` is the value passed to `call` method
pub struct ThreadSafeCallContext<T: 'static> {
  pub env: Env,
  pub value: T,
}

#[repr(u8)]
pub enum ThreadsafeFunctionCallMode {
  NonBlocking,
  Blocking,
}

impl Into<napi_threadsafe_function_call_mode> for ThreadsafeFunctionCallMode {
  fn into(self) -> napi_threadsafe_function_call_mode {
    match self {
      ThreadsafeFunctionCallMode::Blocking => {
        napi_threadsafe_function_call_mode::napi_tsfn_blocking
      }
      ThreadsafeFunctionCallMode::NonBlocking => {
        napi_threadsafe_function_call_mode::napi_tsfn_nonblocking
      }
    }
  }
}

type_level_enum! {
  /// Type-level `enum` to express how to feed [`ThreadsafeFunction`] errors to
  /// the inner [`JsFunction`].
  ///
  /// ### Context
  ///
  /// For callbacks that expect a `Result`-like kind of input, the convention is
  /// to have the callback take an `error` parameter as its first parameter.
  ///
  /// This way receiving a `Result<Args…>` can be modelled as follows:
  ///
  ///   - In case of `Err(error)`, feed that `error` entity as the first parameter
  ///     of the callback;
  ///
  ///   - Otherwise (in case of `Ok(_)`), feed `null` instead.
  ///
  /// In pseudo-code:
  ///
  /// ```rust,ignore
  /// match result_args {
  ///     Ok(args) => {
  ///         let js_null = /* … */;
  ///         callback.call(
  ///             // this
  ///             None,
  ///             // args…
  ///             &iter::once(js_null).chain(args).collect::<Vec<_>>(),
  ///         )
  ///     },
  ///     Err(err) => callback.call(None, &[JsError::from(err)]),
  /// }
  /// ```
  ///
  /// **Note that the `Err` case can stem from a failed conversion from native
  /// values to js values when calling the callback!**
  ///
  /// That's why:
  ///
  /// > **[This][`ErrorStrategy::CalleeHandled`] is the default error strategy**.
  ///
  /// In order to opt-out of it, [`ThreadsafeFunction`] has an optional second
  /// generic parameter (of "kind" [`ErrorStrategy::T`]) that defines whether
  /// this behavior ([`ErrorStrategy::CalleeHandled`]) or a non-`Result` one
  /// ([`ErrorStrategy::Fatal`]) is desired.
  pub enum ErrorStrategy {
    /// Input errors (including conversion errors) are left for the callee to
    /// handle:
    ///
    /// The callee receives an extra `error` parameter (the first one), which is
    /// `null` if no error occurred, and the error payload otherwise.
    CalleeHandled,

    /// Input errors (including conversion errors) are deemed fatal:
    ///
    /// they can thus cause a `panic!` or abort the process.
    ///
    /// The callee thus is not expected to have to deal with [that extra `error`
    /// parameter][CalleeHandled], which is thus not added.
    Fatal,
  }
}

/// Communicate with the addon's main thread by invoking a JavaScript function from other threads.
///
/// ## Example
/// An example of using `ThreadsafeFunction`:
///
/// ```rust
/// #[macro_use]
/// extern crate napi_derive;
///
/// use std::thread;
///
/// use napi::{
///     threadsafe_function::{
///         ThreadSafeCallContext, ThreadsafeFunctionCallMode, ThreadsafeFunctionReleaseMode,
///     },
///     CallContext, Error, JsFunction, JsNumber, JsUndefined, Result, Status,
/// };
///
/// #[js_function(1)]
/// pub fn test_threadsafe_function(ctx: CallContext) -> Result<JsUndefined> {
///   let func = ctx.get::<JsFunction>(0)?;
///
///   let tsfn =
///       ctx
///           .env
///           .create_threadsafe_function(&func, 0, |ctx: ThreadSafeCallContext<Vec<u32>>| {
///             ctx.value
///                 .iter()
///                 .map(|v| ctx.env.create_uint32(*v))
///                 .collect::<Result<Vec<JsNumber>>>()
///           })?;
///
///   let tsfn_cloned = tsfn.clone();
///
///   thread::spawn(move || {
///       let output: Vec<u32> = vec![0, 1, 2, 3];
///       // It's okay to call a threadsafe function multiple times.
///       tsfn.call(Ok(output.clone()), ThreadsafeFunctionCallMode::Blocking);
///   });
///
///   thread::spawn(move || {
///       let output: Vec<u32> = vec![3, 2, 1, 0];
///       // It's okay to call a threadsafe function multiple times.
///       tsfn_cloned.call(Ok(output.clone()), ThreadsafeFunctionCallMode::NonBlocking);
///   });
///
///   ctx.env.get_undefined()
/// }
/// ```
pub struct ThreadsafeFunction<T: 'static, ES: ErrorStrategy::T = ErrorStrategy::CalleeHandled> {
  raw_tsfn: sys::napi_threadsafe_function,
  aborted: Arc<AtomicBool>,
  /// Set to `true` when the N-API environment begins teardown. When this is
  /// true, calling napi functions from `call_js_cb` can crash in V8's
  /// `node->IsInUse()` check. Shared with the env cleanup hook registered in
  /// `create`.
  env_tearing_down: Arc<AtomicBool>,
  /// DEVX-877: shared lifecycle handle, registered with the shutdown registry
  /// on creation. The registry's walk during `signal_shutdown_requested`
  /// drives this TSFN's napi-side refcount to zero via repeated
  /// `napi_release_threadsafe_function(napi_tsfn_release)` calls. Once that
  /// completes, the per-handle `shutdown_released` flag is set and every
  /// subsequent `Drop` on this `ThreadsafeFunction` (or any clone) skips its
  /// own `napi_release`, avoiding a double-release on a TSFN that has already
  /// been closed.
  lifecycle: Arc<TsfnLifecycleHandle>,
  _phantom: PhantomData<(T, ES)>,
}

/// DEVX-877: shared lifecycle handle for a `ThreadsafeFunction`.
///
/// One handle per TSFN, shared across every `ThreadsafeFunction` clone via
/// `Arc`. The shutdown registry holds a `Weak<TsfnLifecycleHandle>`; on
/// shutdown it calls [`crate::lifecycle::Releasable::release_for_shutdown`]
/// which loops `napi_release_threadsafe_function(napi_tsfn_release)` until
/// the call returns a non-`napi_ok` status, indicating the refcount has hit
/// zero and the TSFN is closing. Subsequent `Drop` calls then no-op.
struct TsfnLifecycleHandle {
  raw_tsfn: AtomicPtr<sys::napi_threadsafe_function__>,
  /// Set to `true` once the registry has driven this TSFN's refcount to zero
  /// via the `release_for_shutdown` walk. Subsequent `Drop` calls check this
  /// flag and skip their own `napi_release`, since Node has already begun
  /// closing the TSFN.
  shutdown_released: AtomicBool,
}

impl crate::lifecycle::Releasable for TsfnLifecycleHandle {
  fn release_for_shutdown(&self) {
    // CAS so concurrent registry walks (idempotency) and parallel Drops
    // don't both attempt to drain the refcount.
    if self
      .shutdown_released
      .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
      .is_err()
    {
      return;
    }
    let raw = self.raw_tsfn.load(Ordering::Acquire);
    if raw.is_null() {
      return;
    }
    // Loop releasing in `napi_tsfn_release` mode (NOT `napi_tsfn_abort`)
    // until the refcount hits zero. `release` lets pending AsyncCbs drain
    // via Node's normal close path; `abort` would forcibly invalidate the
    // TSFN while AsyncCbs are still in flight, opening the UAF window that
    // caused Class D in a prior attempt (napi-rs commit 25420a05, reverted
    // in 1378924b). Bounded iteration is a paranoia guard against an API
    // contract change that could otherwise spin.
    for _ in 0..1024 {
      let status = unsafe {
        sys::napi_release_threadsafe_function(
          raw,
          sys::napi_threadsafe_function_release_mode::napi_tsfn_release,
        )
      };
      if status != sys::Status::napi_ok {
        break;
      }
    }
  }
}

impl<T: 'static, ES: ErrorStrategy::T> Clone for ThreadsafeFunction<T, ES> {
  fn clone(&self) -> Self {
    if !self.aborted.load(Ordering::Acquire) {
      let acquire_status = unsafe { sys::napi_acquire_threadsafe_function(self.raw_tsfn) };
      debug_assert!(
        acquire_status == sys::Status::napi_ok,
        "Acquire threadsafe function failed in clone"
      );
    }

    Self {
      raw_tsfn: self.raw_tsfn,
      aborted: Arc::clone(&self.aborted),
      env_tearing_down: Arc::clone(&self.env_tearing_down),
      lifecycle: Arc::clone(&self.lifecycle),
      _phantom: PhantomData,
    }
  }
}

struct ThreadsafeFunctionContext<R> {
  callback: R,
  env_tearing_down: Arc<AtomicBool>,
}

struct ThreadsafeFunctionCleanupHookData {
  flag: Arc<AtomicBool>,
}

unsafe extern "C" fn threadsafe_function_env_teardown_cleanup(data: *mut c_void) {
  let hook_data = Box::from_raw(data as *mut ThreadsafeFunctionCleanupHookData);
  hook_data.flag.store(true, Ordering::Release);
}

unsafe impl<T, ES: ErrorStrategy::T> Send for ThreadsafeFunction<T, ES> {}
unsafe impl<T, ES: ErrorStrategy::T> Sync for ThreadsafeFunction<T, ES> {}

impl<T: 'static, ES: ErrorStrategy::T> ThreadsafeFunction<T, ES> {
  /// See [napi_create_threadsafe_function](https://nodejs.org/api/n-api.html#n_api_napi_create_threadsafe_function)
  /// for more information.
  #[inline]
  pub fn create<
    V: NapiValue,
    R: 'static + Send + FnMut(ThreadSafeCallContext<T>) -> Result<Vec<V>>,
  >(
    env: sys::napi_env,
    func: &JsFunction,
    max_queue_size: usize,
    callback: R,
  ) -> Result<Self> {
    let mut async_resource_name = ptr::null_mut();
    let s = "napi_rs_threadsafe_function";
    let len = s.len();
    let s = CString::new(s)?;
    check_status!(unsafe {
      sys::napi_create_string_utf8(env, s.as_ptr(), len, &mut async_resource_name)
    })?;

    // Register an env cleanup hook so we can detect when the V8 environment
    // begins tearing down. `call_js_cb` checks the flag on every invocation
    // and bails out without making any napi calls if it is set, avoiding
    // `Check failed: node->IsInUse()` aborts during teardown.
    //
    // The hook data is freed by the hook itself when it fires. If the TSFN is
    // dropped before env teardown, the hook data is leaked (a small, bounded
    // allocation per TSFN). The atomic flag itself is reference-counted via
    // `Arc` and stays alive as long as any TSFN clone or pending `call_js_cb`
    // invocation holds a reference.
    let env_tearing_down = Arc::new(AtomicBool::new(false));
    let hook_data = Box::into_raw(Box::new(ThreadsafeFunctionCleanupHookData {
      flag: Arc::clone(&env_tearing_down),
    }));
    unsafe {
      sys::napi_add_env_cleanup_hook(
        env,
        Some(threadsafe_function_env_teardown_cleanup),
        hook_data as *mut c_void,
      );
    }

    let initial_thread_count = 1usize;
    let mut raw_tsfn = ptr::null_mut();
    let ctx = Box::into_raw(Box::new(ThreadsafeFunctionContext {
      callback,
      env_tearing_down: Arc::clone(&env_tearing_down),
    })) as *mut _;
    check_status!(unsafe {
      sys::napi_create_threadsafe_function(
        env,
        func.0.value,
        ptr::null_mut(),
        async_resource_name,
        max_queue_size,
        initial_thread_count,
        ctx,
        Some(thread_finalize_cb::<T, V, R>),
        ctx,
        Some(call_js_cb::<T, V, R, ES>),
        &mut raw_tsfn,
      )
    })?;

    // DEVX-877: register this TSFN's lifecycle handle with the shutdown
    // registry. On `process.on('beforeExit')` the registry walks every live
    // entry and drives the napi refcount to zero, so Node never reaches
    // `Environment::RunCleanup` with an outstanding TSFN whose internal
    // napi_ref's GlobalHandle would crash in `GlobalHandles::Destroy`
    // (Class A).
    let lifecycle = Arc::new(TsfnLifecycleHandle {
      raw_tsfn: AtomicPtr::new(raw_tsfn as *mut sys::napi_threadsafe_function__),
      shutdown_released: AtomicBool::new(false),
    });
    crate::lifecycle::register_releasable(
      Arc::downgrade(&lifecycle) as std::sync::Weak<dyn crate::lifecycle::Releasable>
    );

    Ok(ThreadsafeFunction {
      raw_tsfn,
      aborted: Arc::new(AtomicBool::new(false)),
      env_tearing_down,
      lifecycle,
      _phantom: PhantomData,
    })
  }

  /// See [napi_ref_threadsafe_function](https://nodejs.org/api/n-api.html#n_api_napi_ref_threadsafe_function)
  /// for more information.
  ///
  /// "ref" is a keyword so that we use "refer" here.
  pub fn refer(&mut self, env: &Env) -> Result<()> {
    if self.aborted.load(Ordering::Acquire) {
      return Err(Error::new(
        Status::Closing,
        format!("Can not ref, Thread safe function already aborted"),
      ));
    }
    check_status!(unsafe { sys::napi_ref_threadsafe_function(env.0, self.raw_tsfn) })
  }

  /// See [napi_unref_threadsafe_function](https://nodejs.org/api/n-api.html#n_api_napi_unref_threadsafe_function)
  /// for more information.
  pub fn unref(&mut self, env: &Env) -> Result<()> {
    if self.aborted.load(Ordering::Acquire) {
      return Err(Error::new(
        Status::Closing,
        format!("Can not unref, Thread safe function already aborted"),
      ));
    }
    check_status!(unsafe { sys::napi_unref_threadsafe_function(env.0, self.raw_tsfn) })
  }

  pub fn aborted(&self) -> bool {
    self.aborted.load(Ordering::Acquire)
  }

  pub fn abort(self) -> Result<()> {
    check_status!(unsafe {
      sys::napi_release_threadsafe_function(
        self.raw_tsfn,
        sys::napi_threadsafe_function_release_mode::napi_tsfn_abort,
      )
    })?;
    self.aborted.store(true, Ordering::Release);
    Ok(())
  }

  /// Get the raw `ThreadSafeFunction` pointer
  pub fn raw(&self) -> sys::napi_threadsafe_function {
    self.raw_tsfn
  }
}

impl<T: 'static> ThreadsafeFunction<T, ErrorStrategy::CalleeHandled> {
  /// See [napi_call_threadsafe_function](https://nodejs.org/api/n-api.html#n_api_napi_call_threadsafe_function)
  /// for more information.
  pub fn call(&self, value: Result<T>, mode: ThreadsafeFunctionCallMode) -> Status {
    if self.aborted.load(Ordering::Acquire) {
      return Status::Closing;
    }
    unsafe {
      sys::napi_call_threadsafe_function(
        self.raw_tsfn,
        Box::into_raw(Box::new(value)) as *mut _,
        mode.into(),
      )
    }
    .into()
  }
}

impl<T: 'static> ThreadsafeFunction<T, ErrorStrategy::Fatal> {
  /// See [napi_call_threadsafe_function](https://nodejs.org/api/n-api.html#n_api_napi_call_threadsafe_function)
  /// for more information.
  pub fn call(&self, value: T, mode: ThreadsafeFunctionCallMode) -> Status {
    if self.aborted.load(Ordering::Acquire) {
      return Status::Closing;
    }
    unsafe {
      sys::napi_call_threadsafe_function(
        self.raw_tsfn,
        Box::into_raw(Box::new(value)) as *mut _,
        mode.into(),
      )
    }
    .into()
  }
}

impl<T: 'static, ES: ErrorStrategy::T> Drop for ThreadsafeFunction<T, ES> {
  fn drop(&mut self) {
    // DEVX-877: if the shutdown registry has already driven this TSFN's
    // refcount to zero (during `process.on('beforeExit')`), there's nothing
    // left to release — Node has begun closing the underlying napi handle.
    // Calling `napi_release` again would either no-op (returning
    // `napi_closing`) or trip the assert below, so we skip.
    if self.lifecycle.shutdown_released.load(Ordering::Acquire) {
      return;
    }
    if !self.aborted.load(Ordering::Acquire) && !self.env_tearing_down.load(Ordering::Acquire) {
      let release_status = unsafe {
        sys::napi_release_threadsafe_function(
          self.raw_tsfn,
          sys::napi_threadsafe_function_release_mode::napi_tsfn_release,
        )
      };
      assert!(
        release_status == sys::Status::napi_ok,
        "Threadsafe Function release failed"
      );
    }
  }
}

unsafe extern "C" fn thread_finalize_cb<T: 'static, V: NapiValue, R>(
  _raw_env: sys::napi_env,
  finalize_data: *mut c_void,
  _finalize_hint: *mut c_void,
) where
  R: 'static + Send + FnMut(ThreadSafeCallContext<T>) -> Result<Vec<V>>,
{
  // cleanup
  drop(Box::<ThreadsafeFunctionContext<R>>::from_raw(
    finalize_data.cast(),
  ));
}

unsafe extern "C" fn call_js_cb<T: 'static, V: NapiValue, R, ES>(
  raw_env: sys::napi_env,
  js_callback: sys::napi_value,
  context: *mut c_void,
  data: *mut c_void,
) where
  R: 'static + Send + FnMut(ThreadSafeCallContext<T>) -> Result<Vec<V>>,
  ES: ErrorStrategy::T,
{
  let tsfn_ctx: &mut ThreadsafeFunctionContext<R> = &mut *context.cast();

  // If the env is tearing down, drop the data without making any napi calls.
  // Calling napi_get_undefined / napi_call_function / napi_get_null on a
  // partially-torn-down env trips V8's `Check failed: node->IsInUse()` and
  // aborts the process.
  // DEVX-877: also consult the process-wide shutdown signal — fires earlier than
  // env-cleanup-hooks.
  if tsfn_ctx.env_tearing_down.load(Ordering::Acquire) || crate::lifecycle::shutdown_requested() {
    match ES::VALUE {
      ErrorStrategy::CalleeHandled::VALUE => {
        drop(Box::<Result<T>>::from_raw(data.cast()));
      }
      ErrorStrategy::Fatal::VALUE => {
        drop(Box::<T>::from_raw(data.cast()));
      }
    }
    return;
  }

  let ctx: &mut R = &mut tsfn_ctx.callback;
  let val: Result<T> = match ES::VALUE {
    ErrorStrategy::CalleeHandled::VALUE => *Box::<Result<T>>::from_raw(data.cast()),
    ErrorStrategy::Fatal::VALUE => Ok(*Box::<T>::from_raw(data.cast())),
  };

  let mut recv = ptr::null_mut();
  sys::napi_get_undefined(raw_env, &mut recv);

  let ret = val.and_then(|v| {
    (ctx)(ThreadSafeCallContext {
      env: Env::from_raw(raw_env),
      value: v,
    })
  });

  let status;

  // Follow async callback conventions: https://nodejs.org/en/knowledge/errors/what-are-the-error-conventions/
  // Check if the Result is okay, if so, pass a null as the first (error) argument automatically.
  // If the Result is an error, pass that as the first argument.
  match ret {
    Ok(values) => {
      let values = values.iter().map(|v| v.raw());
      let args: Vec<sys::napi_value> = if ES::VALUE == ErrorStrategy::CalleeHandled::VALUE {
        let mut js_null = ptr::null_mut();
        sys::napi_get_null(raw_env, &mut js_null);
        ::core::iter::once(js_null).chain(values).collect()
      } else {
        values.collect()
      };
      status = sys::napi_call_function(
        raw_env,
        recv,
        js_callback,
        args.len(),
        args.as_ptr(),
        ptr::null_mut(),
      );
    }
    Err(e) if ES::VALUE == ErrorStrategy::Fatal::VALUE => {
      panic!("{:?}", e);
    }
    Err(e) => {
      status = sys::napi_call_function(
        raw_env,
        recv,
        js_callback,
        1,
        [JsError::from(e).into_value(raw_env)].as_mut_ptr(),
        ptr::null_mut(),
      );
    }
  }
  if status == sys::Status::napi_pending_exception {
    let mut error_result = ptr::null_mut();
    assert_eq!(
      sys::napi_get_and_clear_last_exception(raw_env, &mut error_result),
      sys::Status::napi_ok
    );
    assert_eq!(
      sys::napi_fatal_exception(raw_env, error_result),
      sys::Status::napi_ok
    );
  } else if status != sys::Status::napi_ok {
    let error_code: Status = status.into();
    let error_code_string = format!("{:?}", error_code);
    let mut error_code_value = ptr::null_mut();
    assert_eq!(
      sys::napi_create_string_utf8(
        raw_env,
        error_code_string.as_ptr() as *const _,
        error_code_string.len(),
        &mut error_code_value
      ),
      sys::Status::napi_ok,
    );
    let error_msg = "Call JavaScript callback failed in thread safe function";
    let mut error_msg_value = ptr::null_mut();
    assert_eq!(
      sys::napi_create_string_utf8(
        raw_env,
        error_msg.as_ptr() as *const _,
        error_msg.len(),
        &mut error_msg_value,
      ),
      sys::Status::napi_ok,
    );
    let mut error_value = ptr::null_mut();
    assert_eq!(
      sys::napi_create_error(raw_env, error_code_value, error_msg_value, &mut error_value),
      sys::Status::napi_ok,
    );
    assert_eq!(
      sys::napi_fatal_exception(raw_env, error_value),
      sys::Status::napi_ok
    );
  }
}

/// Helper
macro_rules! type_level_enum {(
  $( #[doc = $doc:tt] )*
  $pub:vis
  enum $EnumName:ident {
    $(
      $( #[doc = $doc_variant:tt] )*
      $Variant:ident
    ),* $(,)?
  }
) => (type_level_enum! { // This requires the macro to be in scope when called.
  with_docs! {
    $( #[doc = $doc] )*
    ///
    /// ### Type-level `enum`
    ///
    /// Until `const_generics` can handle custom `enum`s, this pattern must be
    /// implemented at the type level.
    ///
    /// We thus end up with:
    ///
    /// ```rust,ignore
    /// #[type_level_enum]
    #[doc = ::core::concat!(
      " enum ", ::core::stringify!($EnumName), " {",
    )]
    $(
      #[doc = ::core::concat!(
        "     ", ::core::stringify!($Variant), ",",
      )]
    )*
    #[doc = " }"]
    /// ```
    ///
    #[doc = ::core::concat!(
      "With [`", ::core::stringify!($EnumName), "::T`](#reexports) \
      being the type-level \"enum type\":",
    )]
    ///
    /// ```rust,ignore
    #[doc = ::core::concat!(
      "<Param: ", ::core::stringify!($EnumName), "::T>"
    )]
    /// ```
  }
  #[allow(warnings)]
  $pub mod $EnumName {
    #[doc(no_inline)]
    pub use $EnumName as T;

    super::type_level_enum! {
      with_docs! {
        #[doc = ::core::concat!(
          "See [`", ::core::stringify!($EnumName), "`]\
          [super::", ::core::stringify!($EnumName), "]"
        )]
      }
      pub trait $EnumName : __sealed::$EnumName + ::core::marker::Sized + 'static {
        const VALUE: __value::$EnumName;
      }
    }

    mod __sealed { pub trait $EnumName {} }

    mod __value {
      #[derive(Debug, PartialEq, Eq)]
      pub enum $EnumName { $( $Variant ),* }
    }

    $(
      $( #[doc = $doc_variant] )*
      pub enum $Variant {}
      impl __sealed::$EnumName for $Variant {}
      impl $EnumName for $Variant {
        const VALUE: __value::$EnumName = __value::$EnumName::$Variant;
      }
      impl $Variant {
        pub const VALUE: __value::$EnumName = __value::$EnumName::$Variant;
      }
    )*
  }
});(
  with_docs! {
    $( #[doc = $doc:expr] )*
  }
  $item:item
) => (
  $( #[doc = $doc] )*
  $item
)}
use type_level_enum;
