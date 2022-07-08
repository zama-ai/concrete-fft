use crate::c64;
use crate::fft_simd::{fft_simd2_emu, wrap_impl, FftSimd1, FftSimd2, FRAC_1_SQRT_2, H1X, H1Y};
use core::fmt::Debug;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;

pub use crate::fft_simd::Scalar;
#[derive(Debug, Copy, Clone)]
pub struct Sse2;
#[derive(Debug, Copy, Clone)]
pub struct Sse3;
#[derive(Debug, Copy, Clone)]
pub struct Avx;
#[derive(Debug, Copy, Clone)]
pub struct Fma;

impl FftSimd1 for Sse2 {
    type Xmm = __m128d;

    #[inline(always)]
    unsafe fn cmplx(x: f64, y: f64) -> Self::Xmm {
        _mm_setr_pd(x, y)
    }

    #[inline(always)]
    unsafe fn getpz(z: &c64) -> Self::Xmm {
        _mm_loadu_pd(z as *const c64 as *const f64)
    }

    #[inline(always)]
    unsafe fn setpz(z: *mut c64, x: Self::Xmm) {
        _mm_storeu_pd(z as *mut c64 as *mut f64, x);
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
        _mm_xor_pd(zm, xy)
    }

    #[inline(always)]
    unsafe fn jxpz(xy: Self::Xmm) -> Self::Xmm {
        let xmy = Self::cnjpz(xy);
        _mm_shuffle_pd::<0b01>(xmy, xmy)
    }

    #[inline(always)]
    unsafe fn negpz(xy: Self::Xmm) -> Self::Xmm {
        let zm = Self::cmplx(-0.0, -0.0);
        _mm_xor_pd(zm, xy)
    }

    #[inline(always)]
    unsafe fn mjxpz(xy: Self::Xmm) -> Self::Xmm {
        let yx = _mm_shuffle_pd::<0b01>(xy, xy);
        Self::cnjpz(yx)
    }

    #[inline(always)]
    unsafe fn addpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm {
        _mm_add_pd(a, b)
    }

    #[inline(always)]
    unsafe fn subpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm {
        _mm_sub_pd(a, b)
    }

    #[inline(always)]
    unsafe fn mulpd(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm {
        _mm_mul_pd(a, b)
    }

    #[inline(always)]
    unsafe fn haddpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm {
        let ba = _mm_shuffle_pd::<0b01>(ab, ab);
        let yx = _mm_shuffle_pd::<0b01>(xy, xy);
        let apb = _mm_add_sd(ab, ba);
        let xpy = _mm_add_sd(xy, yx);
        _mm_shuffle_pd::<0b00>(apb, xpy)
    }

    #[inline(always)]
    unsafe fn mulpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm {
        let aa = _mm_unpacklo_pd(ab, ab);
        let bb = _mm_unpackhi_pd(ab, ab);
        _mm_add_pd(_mm_mul_pd(aa, xy), _mm_mul_pd(bb, Self::jxpz(xy)))
    }

    #[inline(always)]
    unsafe fn v8xpz(xy: Self::Xmm) -> Self::Xmm {
        let rr = Self::cmplx(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        _mm_mul_pd(rr, _mm_add_pd(xy, Self::jxpz(xy)))
    }

    #[inline(always)]
    unsafe fn w8xpz(xy: Self::Xmm) -> Self::Xmm {
        let rr = Self::cmplx(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        let ymx = Self::cnjpz(_mm_shuffle_pd::<0b01>(xy, xy));
        Self::mulpz(rr, _mm_add_pd(xy, ymx))
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
        is_x86_feature_detected!("sse2")
    }
}

impl FftSimd1 for Sse3 {
    type Xmm = __m128d;

    wrap_impl! {
    Sse2 =>
        unsafe fn cmplx(x: f64, b: f64) -> Self::Xmm;
        unsafe fn getpz(z: &c64) -> Self::Xmm;
        unsafe fn setpz(z: *mut c64, x: Self::Xmm);
        unsafe fn swappz(x: &mut c64, y: &mut c64);
        unsafe fn cnjpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn jxpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn negpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn mjxpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn addpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm;
        unsafe fn subpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm;
        unsafe fn mulpd(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm;
        unsafe fn w8xpz(xy: Self::Xmm) -> Self::Xmm;
    }

    #[inline(always)]
    unsafe fn haddpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm {
        _mm_hadd_pd(ab, xy)
    }

    #[inline(always)]
    unsafe fn mulpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm {
        let aa = _mm_unpacklo_pd(ab, ab);
        let bb = _mm_unpackhi_pd(ab, ab);
        let yx = _mm_shuffle_pd::<0b01>(xy, xy);
        _mm_addsub_pd(_mm_mul_pd(aa, xy), _mm_mul_pd(bb, yx))
    }

    #[inline(always)]
    unsafe fn v8xpz(xy: Self::Xmm) -> Self::Xmm {
        let rr = Self::cmplx(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        let yx = _mm_shuffle_pd::<0b01>(xy, xy);
        _mm_mul_pd(rr, _mm_addsub_pd(xy, yx))
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
        is_x86_feature_detected!("sse3")
    }
}

impl FftSimd1 for Avx {
    type Xmm = __m128d;

    wrap_impl! {
    Sse3 =>
        unsafe fn cmplx(x: f64, y: f64) -> Self::Xmm;
        unsafe fn getpz(z: &c64) -> Self::Xmm;
        unsafe fn setpz(z: *mut c64, x: Self::Xmm);
        unsafe fn swappz(x: &mut c64, y: &mut c64);
        unsafe fn cnjpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn jxpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn negpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn mjxpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn addpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm;
        unsafe fn subpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm;
        unsafe fn mulpd(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm;
        unsafe fn haddpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm;
        unsafe fn mulpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm;
        unsafe fn v8xpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn w8xpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn h1xpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn h3xpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn hfxpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn hdxpz(xy: Self::Xmm) -> Self::Xmm;
    }

    fn is_feature_detected() -> bool {
        is_x86_feature_detected!("avx")
    }
}

impl FftSimd1 for Fma {
    type Xmm = __m128d;

    wrap_impl! {
    Sse3 =>
        unsafe fn cmplx(x: f64, y: f64) -> Self::Xmm;
        unsafe fn getpz(z: &c64) -> Self::Xmm;
        unsafe fn setpz(z: *mut c64, x: Self::Xmm);
        unsafe fn swappz(x: &mut c64, y: &mut c64);
        unsafe fn cnjpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn jxpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn negpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn mjxpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn addpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm;
        unsafe fn subpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm;
        unsafe fn mulpd(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm;
        unsafe fn haddpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm;
        unsafe fn v8xpz(xy: Self::Xmm) -> Self::Xmm;
        unsafe fn w8xpz(xy: Self::Xmm) -> Self::Xmm;
    }

    #[inline(always)]
    unsafe fn mulpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm {
        let aa = _mm_unpacklo_pd(ab, ab);
        let bb = _mm_unpackhi_pd(ab, ab);
        let yx = _mm_shuffle_pd::<0b01>(xy, xy);
        _mm_fmaddsub_pd(aa, xy, _mm_mul_pd(bb, yx))
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
        is_x86_feature_detected!("fma")
    }
}

fft_simd2_emu!(Sse2);
fft_simd2_emu!(Sse3);

impl FftSimd2 for Avx {
    type Ymm = __m256d;

    #[inline(always)]
    unsafe fn cmplx2(a: f64, b: f64, c: f64, d: f64) -> Self::Ymm {
        _mm256_setr_pd(a, b, c, d)
    }

    #[inline(always)]
    unsafe fn getpz2(z: *const c64) -> Self::Ymm {
        _mm256_loadu_pd(z as *const f64)
    }

    #[inline(always)]
    unsafe fn setpz2(z: *mut c64, x: Self::Ymm) {
        _mm256_storeu_pd(z as *mut f64, x);
    }

    #[inline(always)]
    unsafe fn cnjpz2(xy: Self::Ymm) -> Self::Ymm {
        let zm = Self::cmplx2(0.0, -0.0, 0.0, -0.0);
        _mm256_xor_pd(zm, xy)
    }

    #[inline(always)]
    unsafe fn jxpz2(xy: Self::Ymm) -> Self::Ymm {
        let xmy = Self::cnjpz2(xy);
        _mm256_shuffle_pd::<0b0101>(xmy, xmy)
    }

    #[inline(always)]
    unsafe fn negpz2(xy: Self::Ymm) -> Self::Ymm {
        let mm = Self::cmplx2(-0.0, -0.0, -0.0, -0.0);
        _mm256_xor_pd(mm, xy)
    }

    #[inline(always)]
    unsafe fn addpz2(a: Self::Ymm, b: Self::Ymm) -> Self::Ymm {
        _mm256_add_pd(a, b)
    }

    #[inline(always)]
    unsafe fn subpz2(a: Self::Ymm, b: Self::Ymm) -> Self::Ymm {
        _mm256_sub_pd(a, b)
    }

    #[inline(always)]
    unsafe fn mulpd2(a: Self::Ymm, b: Self::Ymm) -> Self::Ymm {
        _mm256_mul_pd(a, b)
    }

    #[inline(always)]
    unsafe fn mulpz2(ab: Self::Ymm, xy: Self::Ymm) -> Self::Ymm {
        let aa = _mm256_unpacklo_pd(ab, ab);
        let bb = _mm256_unpackhi_pd(ab, ab);
        let yx = _mm256_shuffle_pd::<0b0101>(xy, xy);
        _mm256_addsub_pd(_mm256_mul_pd(aa, xy), _mm256_mul_pd(bb, yx))
    }

    #[inline(always)]
    unsafe fn v8xpz2(xy: Self::Ymm) -> Self::Ymm {
        let rr = Self::cmplx2(FRAC_1_SQRT_2, FRAC_1_SQRT_2, FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        let yx = _mm256_shuffle_pd::<0b0101>(xy, xy);
        _mm256_mul_pd(rr, _mm256_addsub_pd(xy, yx))
    }

    #[inline(always)]
    unsafe fn w8xpz2(xy: Self::Ymm) -> Self::Ymm {
        let rr = Self::cmplx2(FRAC_1_SQRT_2, FRAC_1_SQRT_2, FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        let ymx = Self::cnjpz2(_mm256_shuffle_pd::<0b0101>(xy, xy));
        _mm256_mul_pd(rr, _mm256_add_pd(xy, ymx))
    }

    #[inline(always)]
    unsafe fn h1xpz2(xy: Self::Ymm) -> Self::Ymm {
        let h1 = Self::cmplx2(H1X, H1Y, H1X, H1Y);
        Self::mulpz2(h1, xy)
    }

    #[inline(always)]
    unsafe fn h3xpz2(xy: Self::Ymm) -> Self::Ymm {
        let h3 = Self::cmplx2(-H1Y, -H1X, -H1Y, -H1X);
        Self::mulpz2(h3, xy)
    }

    #[inline(always)]
    unsafe fn hfxpz2(xy: Self::Ymm) -> Self::Ymm {
        let hf = Self::cmplx2(H1X, -H1Y, H1X, -H1Y);
        Self::mulpz2(hf, xy)
    }

    #[inline(always)]
    unsafe fn hdxpz2(xy: Self::Ymm) -> Self::Ymm {
        let hd = Self::cmplx2(-H1Y, H1X, -H1Y, H1X);
        Self::mulpz2(hd, xy)
    }

    #[inline(always)]
    unsafe fn duppz2(x: Self::Xmm) -> Self::Ymm {
        _mm256_broadcast_pd(&x)
    }

    #[inline(always)]
    unsafe fn duppz3(z: &c64) -> Self::Ymm {
        Self::duppz2(Self::getpz(z))
    }

    #[inline(always)]
    unsafe fn cat(a: Self::Xmm, b: Self::Xmm) -> Self::Ymm {
        let ax = _mm256_castpd128_pd256(a);
        _mm256_insertf128_pd(ax, b, 1)
    }

    #[inline(always)]
    unsafe fn catlo(ax: Self::Ymm, by: Self::Ymm) -> Self::Ymm {
        _mm256_permute2f128_pd::<0b00100000>(ax, by)
    }

    #[inline(always)]
    unsafe fn cathi(ax: Self::Ymm, by: Self::Ymm) -> Self::Ymm {
        _mm256_permute2f128_pd::<0b00110001>(ax, by)
    }

    #[inline(always)]
    unsafe fn swaplohi(ab: Self::Ymm) -> Self::Ymm {
        _mm256_permute2f128_pd::<0b00000001>(ab, ab)
    }

    #[inline(always)]
    unsafe fn getwp2(s: usize, w: *const c64, p: usize) -> Self::Ymm {
        let sp = s * p;
        let w0 = &*w.add(sp);
        let w1 = &*w.add(sp + s);
        Self::cat(Self::getpz(w0), Self::getpz(w1))
    }

    #[inline(always)]
    unsafe fn cnj_getwp2(s: usize, w: *const c64, p: usize) -> Self::Ymm {
        Self::cnjpz2(Self::getwp2(s, w, p))
    }

    #[inline(always)]
    unsafe fn getlo(a_b: Self::Ymm) -> Self::Xmm {
        _mm256_castpd256_pd128(a_b)
    }

    #[inline(always)]
    unsafe fn gethi(a_b: Self::Ymm) -> Self::Xmm {
        _mm256_extractf128_pd::<0b1>(a_b)
    }

    #[inline(always)]
    unsafe fn getpz3(s: usize, z: *const c64) -> Self::Ymm {
        let z0 = &*z;
        let z1 = &*z.add(s);
        Self::cat(Self::getpz(z0), Self::getpz(z1))
    }

    #[inline(always)]
    unsafe fn setpz3(s: usize, z: *mut c64, x: Self::Ymm) {
        Self::setpz(z, Self::getlo(x));
        Self::setpz(z.add(s), Self::gethi(x));
    }
}

impl FftSimd2 for Fma {
    type Ymm = __m256d;

    wrap_impl! {
    Avx =>
        unsafe fn cmplx2(a: f64, b: f64, c: f64, d: f64) -> Self::Ymm;
        unsafe fn getpz2(z: *const c64) -> Self::Ymm;
        unsafe fn setpz2(z: *mut c64, x: Self::Ymm);
        unsafe fn cnjpz2(xy: Self::Ymm) -> Self::Ymm;
        unsafe fn jxpz2(xy: Self::Ymm) -> Self::Ymm;
        unsafe fn negpz2(xy: Self::Ymm) -> Self::Ymm;
        unsafe fn addpz2(a: Self::Ymm, b: Self::Ymm) -> Self::Ymm;
        unsafe fn subpz2(a: Self::Ymm, b: Self::Ymm) -> Self::Ymm;
        unsafe fn mulpd2(a: Self::Ymm, b: Self::Ymm) -> Self::Ymm;
        unsafe fn v8xpz2(xy: Self::Ymm) -> Self::Ymm;
        unsafe fn w8xpz2(xy: Self::Ymm) -> Self::Ymm;
        unsafe fn duppz2(x: Self::Xmm) -> Self::Ymm;
        unsafe fn duppz3(z: &c64) -> Self::Ymm;
        unsafe fn cat(a: Self::Xmm, b: Self::Xmm) -> Self::Ymm;
        unsafe fn catlo(ax: Self::Ymm, by: Self::Ymm) -> Self::Ymm;
        unsafe fn cathi(ax: Self::Ymm, by: Self::Ymm) -> Self::Ymm;
        unsafe fn swaplohi(ab: Self::Ymm) -> Self::Ymm;
        unsafe fn getwp2(s: usize, w: *const c64, p: usize) -> Self::Ymm;
        unsafe fn cnj_getwp2(s: usize, w: *const c64, p: usize) -> Self::Ymm;
        unsafe fn getlo(a_b: Self::Ymm) -> Self::Xmm;
        unsafe fn gethi(a_b: Self::Ymm) -> Self::Xmm;
        unsafe fn getpz3(s: usize, z: *const c64) -> Self::Ymm;
        unsafe fn setpz3(s: usize, z: *mut c64, x: Self::Ymm);
    }

    #[inline(always)]
    unsafe fn mulpz2(ab: Self::Ymm, xy: Self::Ymm) -> Self::Ymm {
        let aa = _mm256_unpacklo_pd(ab, ab);
        let bb = _mm256_unpackhi_pd(ab, ab);
        let yx = _mm256_shuffle_pd::<0b0101>(xy, xy);
        _mm256_fmaddsub_pd(aa, xy, _mm256_mul_pd(bb, yx))
    }

    #[inline(always)]
    unsafe fn h1xpz2(xy: Self::Ymm) -> Self::Ymm {
        let h1 = Self::cmplx2(H1X, H1Y, H1X, H1Y);
        Self::mulpz2(h1, xy)
    }

    #[inline(always)]
    unsafe fn h3xpz2(xy: Self::Ymm) -> Self::Ymm {
        let h3 = Self::cmplx2(-H1Y, -H1X, -H1Y, -H1X);
        Self::mulpz2(h3, xy)
    }

    #[inline(always)]
    unsafe fn hfxpz2(xy: Self::Ymm) -> Self::Ymm {
        let hf = Self::cmplx2(H1X, -H1Y, H1X, -H1Y);
        Self::mulpz2(hf, xy)
    }

    #[inline(always)]
    unsafe fn hdxpz2(xy: Self::Ymm) -> Self::Ymm {
        let hd = Self::cmplx2(-H1Y, H1X, -H1Y, H1X);
        Self::mulpz2(hd, xy)
    }
}
