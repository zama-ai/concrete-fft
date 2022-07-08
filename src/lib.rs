//! This crate computes the forward and inverse Fourier transform of a given array of some size
//! that is a small-ish power of `2`.
//!
//! # Example
//!
//! ```
//! use binfft::{c64, init_twiddles, fwd, inv, fft_scratch};
//! use dyn_stack::{uninit_mem_in_global, DynStack, ReborrowMut};
//! use num_complex::ComplexFloat;
//!
//! const N: usize = 4;
//!
//! let mut mem = uninit_mem_in_global(fft_scratch(N).unwrap());
//! let mut stack = DynStack::new(&mut mem);
//!
//! let mut twiddles = [c64::new(0.0, 0.0); 2 * N];
//! init_twiddles(N, &mut twiddles);
//!
//! let data = [
//!     c64::new(1.0, 0.0),
//!     c64::new(2.0, 0.0),
//!     c64::new(3.0, 0.0),
//!     c64::new(4.0, 0.0),
//! ];
//!
//! let mut transformed_fwd = data;
//! fwd(&mut transformed_fwd, &twiddles, stack.rb_mut());
//!
//! let mut transformed_inv = transformed_fwd;
//! inv(&mut transformed_inv, &twiddles, stack.rb_mut());
//!
//! for (expected, actual) in transformed_inv.iter().map(|z| z / N as f64).zip(data) {
//!     let diff = (expected - actual).abs();
//!     assert!(diff < 1e-9);
//! }
//! ```

use dyn_stack::{SizeOverflow, StackReq};
use num_complex::Complex;

mod fft_simd;
mod twiddles;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod x86;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "wasm32")]
mod wasm32;

pub mod dif4;

/// Complex type containing two `f64` values.
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;

/// Maximum exponent of the power of two size that we can process.
pub const MAX_EXP: usize = 17;

/// Scratch memory requirements for calling fft functions.
pub fn fft_scratch(n: usize) -> Result<StackReq, SizeOverflow> {
    StackReq::try_new_aligned::<c64>(n, 64)
}

/// Initialize twiddles for subsequent forward and inverse Fourier transforms of size `n`.
/// `twiddles` must be of length `2*n`.
pub fn init_twiddles(n: usize, twiddles: &mut [c64]) {
    assert!(n.is_power_of_two());
    let i = n.trailing_zeros() as usize;
    assert!(i < MAX_EXP);
    assert_eq!(twiddles.len(), 2 * n);

    unsafe {
        twiddles::init_wt(4, n, twiddles.as_mut_ptr());
    }
}

/// Execute forward Fourier transform using the computed twiddles.
pub fn fwd(data: &mut [c64], twiddles: &[c64], stack: dyn_stack::DynStack) {
    dif4::fwd(data, twiddles, stack);
}

/// Execute inverse Fourier transform using the computed twiddles.
pub fn inv(data: &mut [c64], twiddles: &[c64], stack: dyn_stack::DynStack) {
    dif4::inv(data, twiddles, stack);
}
