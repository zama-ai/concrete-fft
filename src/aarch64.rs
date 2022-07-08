use crate::c64;
use crate::fft_simd::{
    fft_simd2_emu, wrap_impl, FftSimd1, FftSimd2, FftSimd4, FftSimd8, FRAC_1_SQRT_2, H1X, H1Y,
};
use core::arch::aarch64::*;
use core::fmt::Debug;
use core::mem::{transmute, MaybeUninit};

#[derive(Debug, Copy, Clone)]
pub struct Neon;

#[inline(always)]
unsafe fn to_c64(ab: float64x2_t) -> c64 {
    let tmp = MaybeUninit::<c64>::uninit();
    Neon::setpz(&mut tmp as *mut MaybeUninit<c64> as *mut c64, ab);
    tmp.assume_init()
}

#[inline(always)]
unsafe fn swap_re_im(ab: float64x2_t) -> float64x2_t {
    let ab = to_c64(ab);
    Neon::cmplx(ab.im, ab.re)
}

#[inline(always)]
unsafe fn xor(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    let a = to_c64(a);
    let b = to_c64(b);

    let a_re: u64 = transmute(a.re);
    let a_im: u64 = transmute(a.im);

    let b_re: u64 = transmute(b.re);
    let b_im: u64 = transmute(b.im);

    let res_re = a_re ^ b_re;
    let res_im = a_im ^ b_im;

    Neon::cmplx(transmute(res_re), transmute(res_im))
}

impl FftSimd1 for Neon {
    type Xmm = float64x2_t;

    #[inline(always)]
    unsafe fn cmplx(x: f64, y: f64) -> Self::Xmm {
        let data = [x, y];
        vld1q_f64(data.as_ptr())
    }

    #[inline(always)]
    unsafe fn getpz(z: &c64) -> Self::Xmm {
        vld1q_f64(data as *const c64 as *const f64)
    }

    #[inline(always)]
    unsafe fn setpz(z: *mut c64, x: Self::Xmm) {
        vst1q_f64(z as *mut f64, x);
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
        xor(zm, xy)
    }

    #[inline(always)]
    unsafe fn jxpz(xy: Self::Xmm) -> Self::Xmm {
        let xmy = Self::cnjpz(xy);
        swap_re_im(xmy)
    }

    #[inline(always)]
    unsafe fn negpz(xy: Self::Xmm) -> Self::Xmm {
        let zm = Self::cmplx(-0.0, -0.0);
        xor(zm, xy)
    }

    #[inline(always)]
    unsafe fn mjxpz(xy: Self::Xmm) -> Self::Xmm {
        let yx = swap_re_im(xy);
        Self::cnjpz(yx)
    }

    #[inline(always)]
    unsafe fn addpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm {
        vaddq_f64(a, b)
    }

    #[inline(always)]
    unsafe fn subpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm {
        vsubq_f64(a, b)
    }

    #[inline(always)]
    unsafe fn mulpd(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm {
        vmulq_f64(a, b)
    }

    #[inline(always)]
    unsafe fn haddpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm {
        let ba = swap_re_im(ab);
        let yx = swap_re_im(yx);
        let apb = Self::addpz(ab, ba);
        let xpy = Self::addpz(xy, yx);
        let c64 { re, im: _ } = to_c64(apb);
        let c64 { re: _, im } = to_c64(xpy);
        Self::cmplx(re, im)
    }

    #[inline(always)]
    unsafe fn mulpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm {
        let aa = {
            let c64 { re, im: _ } = to_c64(ab);
            Self::cmplx(re, re)
        };
        let bb = {
            let c64 { re: _, im } = to_c64(ab);
            Self::cmplx(im, im)
        };
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
        is_x86_feature_detected!("neon")
    }
}
