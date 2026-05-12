//! DEVX-877: process-lifecycle shutdown signal that fires earlier than `napi_add_env_cleanup_hook`.
//!
//! The env-cleanup-hooks used by `async_work`, `promise`, and `threadsafe_function` to detect
//! teardown fire too late on Node.js >= 22 / arm64: by the time the hook runs, the v8 isolate
//! has already begun finalizing GlobalHandles, and any napi call into a deferred or threadsafe
//! function can crash inside V8.
//!
//! The Node embedder (the SDK consuming this crate) is expected to call
//! [`signal_shutdown_requested`] from a JS-side `process.on('beforeExit')` (and `'exit'`)
//! hook. That callback fires while v8 is still healthy.
//!
//! Two layers run when the embedder signals:
//!
//! 1. A process-wide `SHUTDOWN_REQUESTED` flag is set, consulted by the existing
//!    callback-path checks in `async_work::complete`, `promise::call_js_cb`,
//!    and `threadsafe_function::call_js_cb` to short-circuit napi calls that
//!    would reach into a partially-torn-down v8.
//!
//! 2. A registry of [`Releasable`] resources (`ThreadsafeFunction` etc.) is
//!    walked, calling [`Releasable::release_for_shutdown`] on each. This
//!    proactively drives every TSFN's napi-side refcount to zero while v8 is
//!    still healthy, so Node's `Environment::RunCleanup` never has to
//!    force-close an outstanding TSFN whose underlying napi_ref would crash
//!    in `GlobalHandles::Destroy` (Class A).
//!
//! All checks and walks are advisory: if the embedder doesn't wire up the
//! signal, behavior falls back to the per-resource env-cleanup-hooks.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Weak};

static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

/// A resource that can be released proactively during shutdown.
///
/// Implementors register a `Weak<Self>` with the global registry via
/// [`register_releasable`]; on shutdown the registry walks every live entry
/// and calls [`Releasable::release_for_shutdown`].
///
/// Implementations must be:
/// - **Idempotent**: the registry walk and the regular `Drop` path may both
///   call this; second and subsequent calls must be no-ops.
/// - **Thread-safe**: the registry walk runs on the JS main thread, but
///   regular `Drop` may run on any thread.
/// - **Safe at any time v8 is healthy**: this is called during
///   `process.on('beforeExit')`, so v8 is still alive but draining.
pub trait Releasable: Send + Sync {
  fn release_for_shutdown(&self);
}

static REGISTRY: Mutex<Vec<Weak<dyn Releasable>>> = Mutex::new(Vec::new());

/// Register a resource for proactive release during shutdown.
///
/// The registry holds a `Weak`, so resources that are dropped before
/// shutdown fall out of the registry naturally.
pub fn register_releasable(resource: Weak<dyn Releasable>) {
  let mut registry = REGISTRY.lock().expect("napi lifecycle registry poisoned");
  registry.push(resource);
}

/// Signal that the embedder is about to enter teardown.
///
/// Sets the global shutdown flag and walks the resource registry, calling
/// [`Releasable::release_for_shutdown`] on every live entry.
///
/// Safe to call from any thread, but expected to be called from the JS main
/// thread (during `process.on('beforeExit')`). Idempotent — subsequent calls
/// just re-set the flag and walk the (now-empty) registry.
pub fn signal_shutdown_requested() {
  SHUTDOWN_REQUESTED.store(true, Ordering::Release);

  // Take the registry contents under the lock so individual release calls
  // can re-enter the registry (e.g. if a release path frees something that
  // tries to deregister itself) without deadlocking. The Vec we drain is
  // already final — no future registrations need to fire here.
  let entries: Vec<Weak<dyn Releasable>> = {
    let mut registry = REGISTRY.lock().expect("napi lifecycle registry poisoned");
    std::mem::take(&mut *registry)
  };
  for entry in entries {
    if let Some(strong) = entry.upgrade() {
      strong.release_for_shutdown();
    }
  }
}

/// Returns whether [`signal_shutdown_requested`] has been called.
///
/// Used by napi-rs internals; embedders typically don't need this directly.
#[inline]
pub fn shutdown_requested() -> bool {
  SHUTDOWN_REQUESTED.load(Ordering::Acquire)
}
