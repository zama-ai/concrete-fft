//! The standard API for Instant is not available in Wasm runtimes.
//! This module replaces the Instant type from std to a custom
//! type that accesses Javascript's Performance API.

#[cfg(target_family = "wasm")]
mod wasm;
#[cfg(not(target_family = "wasm"))]
pub use std::time::Instant;

#[cfg(target_family = "wasm")]
pub use wasm::Instant;
