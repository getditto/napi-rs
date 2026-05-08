//! DEVX-877: process-lifecycle signal that fires earlier than `napi_add_env_cleanup_hook`.
//!
//! The env-cleanup-hooks used by `async_work`, `promise`, and `threadsafe_function` to detect
//! teardown fire too late on Node.js >= 22 / arm64: by the time the hook runs, the v8 isolate
//! has already begun finalizing GlobalHandles, and any napi call into a deferred or threadsafe
//! function can crash inside V8.
//!
//! The Node embedder (the SDK consuming this crate) is expected to call
//! [`signal_shutdown_requested`] from a JS-side `process.on('beforeExit')` (and `'exit'`)
//! hook. That callback fires while v8 is still healthy, giving napi-rs a chance to short-circuit
//! its callback paths before the unsafe window opens.
//!
//! All checks are advisory: if the embedder doesn't wire up the signal, napi-rs falls back to
//! the per-resource env-cleanup-hooks, preserving the existing behavior.

use std::sync::atomic::{AtomicBool, Ordering};

static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Signal that the embedder is about to enter teardown. Once set, napi-rs's
/// `async_work::complete`, `FuturePromise`, and `ThreadsafeFunction` callback
/// paths will treat the env as unsafe-to-touch and skip napi calls that would
/// reach into v8.
///
/// Safe to call from any thread. Idempotent.
pub fn signal_shutdown_requested() {
  SHUTDOWN_REQUESTED.store(true, Ordering::Release);
}

/// Returns whether [`signal_shutdown_requested`] has been called.
///
/// Used by napi-rs internals; embedders typically don't need this directly.
#[inline]
pub fn shutdown_requested() -> bool {
  SHUTDOWN_REQUESTED.load(Ordering::Acquire)
}
