use dyn_stack::{SizeOverflow, StackReq};
use num_complex::Complex;

mod fft_simd;
mod twiddles;
mod x86;

pub mod dif4;

#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;
pub const MAX_EXP: usize = 17;

/// Scratch memory requirements for calling fft functions.
pub fn fft_scratch(n: usize) -> Result<StackReq, SizeOverflow> {
    StackReq::try_new_aligned::<c64>(n, 64)
}

/// Initialize twiddles. `twiddles` must be of length `2*n`.
pub fn init_twiddles(n: usize, twiddles: &mut [c64]) {
    assert!(n.is_power_of_two());
    let i = n.trailing_zeros() as usize;
    assert!(i < MAX_EXP);
    assert_eq!(twiddles.len(), 2 * n);

    unsafe {
        twiddles::init_wt(4, n, twiddles.as_mut_ptr());
    }
}

/// Execute forward fourier transform using the computed twiddles.
pub fn fwd(data: &mut [c64], twiddles: &[c64], stack: dyn_stack::DynStack) {
    dif4::fwd(data, twiddles, stack);
}

/// Execute inverse fourier transform using the computed twiddles.
pub fn inv(data: &mut [c64], twiddles: &[c64], stack: dyn_stack::DynStack) {
    dif4::inv(data, twiddles, stack);
}
