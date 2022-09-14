//! Concrete-FFT is a pure Rust high performance fast Fourier transform library that processes
//! vectors of sizes that are powers of two.
//!
//! This library provides two FFT modules:
//!  - The ordered module FFT applies a forward/inverse FFT that takes its input in standard
//!  order, and outputs the result in standard order. For more detail on what the FFT
//!  computes, check the ordered module-level documentation.
//!  - The unordered module FFT applies a forward FFT that takes its input in standard order,
//!  and outputs the result in a certain permuted order that may depend on the FFT plan. On the
//!  other hand, the inverse FFT takes its input in that same permuted order and outputs its result
//!  in standard order. This is useful for cases where the order of the coefficients in the
//!  Fourier domain is not important. An example is using the Fourier transform for vector
//!  convolution. The only operations that are performed in the Fourier domain are elementwise, and
//!  so the order of the coefficients does not affect the results.
//!
//! # Features
//!
//!  - `std` (default): This enables runtime arch detection for accelerated SIMD instructions, and
//!  an FFT plan that measures the various implementations to choose the fastest one at runtime.
//!  - `serde`: This enables serialization and deserialization functions for the unordered plan.
//!  These allow for data in the Fourier domain to be serialized from the permuted order to the
//!  standard order, and deserialized from the standard order to the permuted order.
//!  This is needed since the inverse transform must be used with the same plan that
//!  computed/deserialized the forward transform (or more specifically, a plan with the same
//!  internal base FFT size).
//!
//!  - `nightly` (automatic feature): This feature is enabled on nightly builds. It enables
//!  unstable Rust features to further speed up the FFT, by activating AVX512F instructions on CPUs
//!  that support them.
//!
//! # Example
//!
#![cfg_attr(feature = "std", doc = "```")]
#![cfg_attr(not(feature = "std"), doc = "```ignore")]
//! use concrete_fft::c64;
//! use concrete_fft::ordered::{Plan, Method};
//! use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
//! use num_complex::ComplexFloat;
//! use std::time::Duration;
//!
//! const N: usize = 4;
//! let plan = Plan::new(4, Method::Measure(Duration::from_millis(10)));
//! let mut scratch_memory = GlobalMemBuffer::new(plan.fft_scratch().unwrap());
//! let mut stack = DynStack::new(&mut scratch_memory);
//!
//! let data = [
//!     c64::new(1.0, 0.0),
//!     c64::new(2.0, 0.0),
//!     c64::new(3.0, 0.0),
//!     c64::new(4.0, 0.0),
//! ];
//!
//! let mut transformed_fwd = data;
//! plan.fwd(&mut transformed_fwd, stack.rb_mut());
//!
//! let mut transformed_inv = transformed_fwd;
//! plan.inv(&mut transformed_inv, stack.rb_mut());
//!
//! for (actual, expected) in transformed_inv.iter().map(|z| z / N as f64).zip(data) {
//!     assert!((expected - actual).abs() < 1e-9);
//! }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::erasing_op,
    clippy::identity_op,
    clippy::zero_prefixed_literal,
    clippy::excessive_precision,
    clippy::type_complexity,
    clippy::too_many_arguments
)]
#![cfg_attr(feature = "nightly", feature(stdsimd, avx512_target_feature))]
#![cfg_attr(docsrs, feature(doc_cfg))]

use num_complex::Complex64;

mod fft_simd;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod x86;

pub(crate) mod dif16;
pub(crate) mod dif2;
pub(crate) mod dif4;
pub(crate) mod dif8;

pub(crate) mod dit16;
pub(crate) mod dit2;
pub(crate) mod dit4;
pub(crate) mod dit8;

pub mod ordered;
pub mod unordered;

/// 64-bit complex floating point type.
#[allow(non_camel_case_types)]
pub type c64 = Complex64;

type FnArray = [unsafe fn(*mut c64, *mut c64, *const c64); 17];

#[derive(Copy, Clone)]
struct FftImpl {
    fwd: FnArray,
    inv: FnArray,
}

#[cfg(feature = "std")]
macro_rules! x86_feature_detected {
    ($tt: tt) => {
        is_x86_feature_detected!($tt)
    };
}

#[cfg(not(feature = "std"))]
macro_rules! x86_feature_detected {
    ($tt: tt) => {
        cfg!(target_arch = $tt)
    };
}

pub(crate) use x86_feature_detected;
