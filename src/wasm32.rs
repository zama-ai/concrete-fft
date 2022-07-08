use crate::c64;
use crate::fft_simd::{
    fft_simd2_emu, wrap_impl, FftSimd1, FftSimd2, FftSimd4, FftSimd8, FRAC_1_SQRT_2, H1X, H1Y,
};
use core::arch::wasm32::*;
use core::fmt::Debug;
use core::mem::{transmute, MaybeUninit};

#[derive(Debug, Copy, Clone)]
pub struct Simd128;

#[inline(always)]
unsafe fn swap_re_im(ab: v128) -> v128 {
    u64x2_shuffle::<1, 0>(ab)
}

impl FftSimd1 for Simd128 {
    type Xmm = v128;

    #[inline(always)]
    unsafe fn cmplx(x: f64, y: f64) -> Self::Xmm {
        f64x2(x, y)
    }

    #[inline(always)]
    unsafe fn getpz(z: &c64) -> Self::Xmm {
        v128_load(z as *const c64 as *const Xmm)
    }

    #[inline(always)]
    unsafe fn setpz(z: *mut c64, x: Self::Xmm) {
        v128_store(z as *mut Xmm, x);
    }

    #[inline(always)]
    unsafe fn swappz(x: &mut c64, y: &mut c64) {
        let z = Self::getpz(x);
        Self::setpz(x, Self::getpz(y));
        Self::setpz(y, z);
    }

    #[inline(always)]
    unsafe fn cnjpz(xy: Self::Xmm) -> Self::Xmm {
        let zm = Self::cmplx(0.0, -0.0);
        v128_xor(zm, xy)
    }

    #[inline(always)]
    unsafe fn jxpz(xy: Self::Xmm) -> Self::Xmm {
        let xmy = Self::cnjpz(xy);
        swap_re_im(xmy)
    }

    #[inline(always)]
    unsafe fn negpz(xy: Self::Xmm) -> Self::Xmm {
        let zm = Self::cmplx(-0.0, -0.0);
        v128_xor(zm, xy)
    }

    #[inline(always)]
    unsafe fn mjxpz(xy: Self::Xmm) -> Self::Xmm {
        let yx = swap_re_im(xy);
        Self::cnjpz(yx)
    }

    #[inline(always)]
    unsafe fn addpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm {
        f64x2_add(a, b)
    }

    #[inline(always)]
    unsafe fn subpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm {
        f64x2_sub(a, b)
    }

    #[inline(always)]
    unsafe fn mulpd(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm {
        f64x2_mul(a, b)
    }

    #[inline(always)]
    unsafe fn haddpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm {
        let ba = swap_re_im(ab);
        let yx = swap_re_im(yx);
        let apb = Self::addpz(ab, ba);
        let xpy = Self::addpz(xy, yx);
        u64x2_shuffle::<0, 3>(apb, xpy)
    }

    #[inline(always)]
    unsafe fn mulpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm {
        let aa = u64x2_shuffle::<0, 0>(ab, ab);
        let bb = u64x2_shuffle::<1, 1>(ab, ab);
        Self::addpz(Self::mulpd(aa, xy), Self::mulpd(bb, Self::jxpz(xy)))
    }

    #[inline(always)]
    unsafe fn v8xpz(xy: Self::Xmm) -> Self::Xmm {
        let rr = Self::cmplx(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        Self::mulpd(rr, Self::addpz(xy, Self::jxpz(xy)))
    }

    #[inline(always)]
    unsafe fn w8xpz(xy: Self::Xmm) -> Self::Xmm {
        let rr = Self::cmplx(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        let ymx = Self::cnjpz(swap_re_im(xy));
        Self::mulpz(rr, Self::addpz(xy, ymx))
    }

    #[inline(always)]
    unsafe fn h1xpz(xy: Self::Xmm) -> Self::Xmm {
        let h1 = Self::cmplx(H1X, H1Y);
        Self::mulpz(h1, xy)
    }

    #[inline(always)]
    unsafe fn h3xpz(xy: Self::Xmm) -> Self::Xmm {
        let h3 = Self::cmplx(-H1Y, -H1X);
        Self::mulpz(h3, xy)
    }

    #[inline(always)]
    unsafe fn hfxpz(xy: Self::Xmm) -> Self::Xmm {
        let hf = Self::cmplx(H1X, -H1Y);
        Self::mulpz(hf, xy)
    }

    #[inline(always)]
    unsafe fn hdxpz(xy: Self::Xmm) -> Self::Xmm {
        let hd = Self::cmplx(-H1Y, H1X);
        Self::mulpz(hd, xy)
    }

    fn is_feature_detected() -> bool {
        false
    }
}
