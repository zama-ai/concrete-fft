use crate::c64;
use core::arch::x86_64::*;
use core::fmt::Debug;

pub const H1X: f64 = 0.923879532511286762010323247995557949;
pub const H1Y: f64 = -0.382683432365089757574419179753100195;
pub use core::f64::consts::FRAC_1_SQRT_2;

#[derive(Debug, Copy, Clone)]
pub struct Scalar;
#[derive(Debug, Copy, Clone)]
pub struct Sse2;
#[derive(Debug, Copy, Clone)]
pub struct Sse3;
#[derive(Debug, Copy, Clone)]
pub struct Avx;
#[derive(Debug, Copy, Clone)]
pub struct Fma;

#[repr(C)]
pub struct XmmEmu(f64, f64);
#[repr(C)]
pub struct YmmEmu<S: FftSimd1>(S::Xmm, S::Xmm);
#[repr(C)]
pub struct ZmmEmu<S: FftSimd2>(S::Ymm, S::Ymm);
#[repr(C)]
pub struct AmmEmu<S: FftSimd4>(S::Zmm, S::Zmm);
#[repr(C)]
pub struct BmmEmu<S: FftSimd8>(S::Amm, S::Amm);

impl Copy for XmmEmu {}
impl<S: FftSimd1> Copy for YmmEmu<S> {}
impl<S: FftSimd2> Copy for ZmmEmu<S> {}
impl<S: FftSimd4> Copy for AmmEmu<S> {}
impl<S: FftSimd8> Copy for BmmEmu<S> {}

impl Clone for XmmEmu {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: FftSimd1> Clone for YmmEmu<S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: FftSimd2> Clone for ZmmEmu<S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: FftSimd4> Clone for AmmEmu<S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: FftSimd8> Clone for BmmEmu<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl Debug for XmmEmu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("XmmEmu")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}
impl<S: FftSimd1> Debug for YmmEmu<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("YmmEmu")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}
impl<S: FftSimd2> Debug for ZmmEmu<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ZmmEmu")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}
impl<S: FftSimd4> Debug for AmmEmu<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("AmmEmu")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}
impl<S: FftSimd8> Debug for BmmEmu<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("BmmEmu")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

pub trait FftSimd1 {
    type Xmm: Copy + Clone + Debug;

    fn is_feature_detected() -> bool;

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

pub trait FftSimd2: FftSimd1 {
    type Ymm: Copy + Clone + Debug;

    unsafe fn cmplx2(a: f64, b: f64, c: f64, d: f64) -> Self::Ymm;
    unsafe fn getpz2(z: *const c64) -> Self::Ymm;
    unsafe fn setpz2(z: *mut c64, x: Self::Ymm);
    unsafe fn cnjpz2(xy: Self::Ymm) -> Self::Ymm;
    unsafe fn jxpz2(xy: Self::Ymm) -> Self::Ymm;
    unsafe fn negpz2(xy: Self::Ymm) -> Self::Ymm;
    unsafe fn addpz2(a: Self::Ymm, b: Self::Ymm) -> Self::Ymm;
    unsafe fn subpz2(a: Self::Ymm, b: Self::Ymm) -> Self::Ymm;
    unsafe fn mulpd2(a: Self::Ymm, b: Self::Ymm) -> Self::Ymm;
    unsafe fn mulpz2(ab: Self::Ymm, xy: Self::Ymm) -> Self::Ymm;
    unsafe fn v8xpz2(xy: Self::Ymm) -> Self::Ymm;
    unsafe fn w8xpz2(xy: Self::Ymm) -> Self::Ymm;
    unsafe fn h1xpz2(xy: Self::Ymm) -> Self::Ymm;
    unsafe fn h3xpz2(xy: Self::Ymm) -> Self::Ymm;
    unsafe fn hfxpz2(xy: Self::Ymm) -> Self::Ymm;
    unsafe fn hdxpz2(xy: Self::Ymm) -> Self::Ymm;
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

pub trait FftSimd4: FftSimd2 {
    type Zmm: Copy + Clone + Debug;

    unsafe fn getpz4(z: *const c64) -> Self::Zmm;
    unsafe fn setpz4(a: *mut c64, z: Self::Zmm);
    unsafe fn cnjpz4(xy: Self::Zmm) -> Self::Zmm;
    unsafe fn jxpz4(xy: Self::Zmm) -> Self::Zmm;
    unsafe fn addpz4(a: Self::Zmm, b: Self::Zmm) -> Self::Zmm;
    unsafe fn subpz4(a: Self::Zmm, b: Self::Zmm) -> Self::Zmm;
    unsafe fn mulpd4(a: Self::Zmm, b: Self::Zmm) -> Self::Zmm;
    unsafe fn mulpz4(ab: Self::Zmm, xy: Self::Zmm) -> Self::Zmm;
    unsafe fn v8xpz4(xy: Self::Zmm) -> Self::Zmm;
    unsafe fn w8xpz4(xy: Self::Zmm) -> Self::Zmm;
    unsafe fn h1xpz4(xy: Self::Zmm) -> Self::Zmm;
    unsafe fn h3xpz4(xy: Self::Zmm) -> Self::Zmm;
    unsafe fn hfxpz4(xy: Self::Zmm) -> Self::Zmm;
    unsafe fn hdxpz4(xy: Self::Zmm) -> Self::Zmm;
    unsafe fn duppz4(x: Self::Xmm) -> Self::Zmm;
    unsafe fn duppz5(x: &c64) -> Self::Zmm;
}

pub trait FftSimd8: FftSimd4 {
    type Amm: Copy + Clone + Debug;

    unsafe fn getpz8(z: *const c64) -> Self::Amm;
    unsafe fn setpz8(a: *mut c64, z: Self::Amm);
    unsafe fn cnjpz8(xy: Self::Amm) -> Self::Amm;
    unsafe fn jxpz8(xy: Self::Amm) -> Self::Amm;
    unsafe fn addpz8(a: Self::Amm, b: Self::Amm) -> Self::Amm;
    unsafe fn subpz8(a: Self::Amm, b: Self::Amm) -> Self::Amm;
    unsafe fn mulpd8(a: Self::Amm, b: Self::Amm) -> Self::Amm;
    unsafe fn mulpz8(ab: Self::Amm, xy: Self::Amm) -> Self::Amm;
    unsafe fn v8xpz8(xy: Self::Amm) -> Self::Amm;
    unsafe fn w8xpz8(xy: Self::Amm) -> Self::Amm;
    unsafe fn h1xpz8(xy: Self::Amm) -> Self::Amm;
    unsafe fn h3xpz8(xy: Self::Amm) -> Self::Amm;
    unsafe fn hfxpz8(xy: Self::Amm) -> Self::Amm;
    unsafe fn hdxpz8(xy: Self::Amm) -> Self::Amm;
    unsafe fn duppz8(x: Self::Xmm) -> Self::Amm;
    unsafe fn duppz9(x: &c64) -> Self::Amm;
}

pub trait FftSimd16: FftSimd8 {
    type Bmm: Copy + Clone + Debug;

    unsafe fn getpz16(z: *const c64) -> Self::Bmm;
    unsafe fn setpz16(a: *mut c64, z: Self::Bmm);
    unsafe fn cnjpz16(xy: Self::Bmm) -> Self::Bmm;
    unsafe fn jxpz16(xy: Self::Bmm) -> Self::Bmm;
    unsafe fn addpz16(a: Self::Bmm, b: Self::Bmm) -> Self::Bmm;
    unsafe fn subpz16(a: Self::Bmm, b: Self::Bmm) -> Self::Bmm;
    unsafe fn mulpd16(a: Self::Bmm, b: Self::Bmm) -> Self::Bmm;
    unsafe fn mulpz16(ab: Self::Bmm, xy: Self::Bmm) -> Self::Bmm;
    unsafe fn v8xpz16(xy: Self::Bmm) -> Self::Bmm;
    unsafe fn w8xpz16(xy: Self::Bmm) -> Self::Bmm;
    unsafe fn h1xpz16(xy: Self::Bmm) -> Self::Bmm;
    unsafe fn h3xpz16(xy: Self::Bmm) -> Self::Bmm;
    unsafe fn hfxpz16(xy: Self::Bmm) -> Self::Bmm;
    unsafe fn hdxpz16(xy: Self::Bmm) -> Self::Bmm;
    unsafe fn duppz16(x: Self::Xmm) -> Self::Bmm;
    unsafe fn duppz17(x: &c64) -> Self::Bmm;
}

macro_rules! wrap_impl {
    ($other: ty =>
        $(unsafe fn $fn_name: ident($($arg_name: ident: $arg_ty: ty),* $(,)?) $(-> $return_ty: ty)?; )*) => {
        $(
            #[inline(always)]
            unsafe fn $fn_name($($arg_name: $arg_ty,)*) $(-> $return_ty)* {
                <$other>::$fn_name($($arg_name,)*)
            }
        )*
    };
}

impl FftSimd1 for Scalar {
    type Xmm = XmmEmu;

    #[inline(always)]
    unsafe fn cmplx(x: f64, y: f64) -> Self::Xmm {
        XmmEmu(x, y)
    }

    #[inline(always)]
    unsafe fn getpz(z: &c64) -> Self::Xmm {
        XmmEmu(z.re, z.im)
    }

    #[inline(always)]
    unsafe fn setpz(z: *mut c64, x: Self::Xmm) {
        (*z).re = x.0;
        (*z).im = x.1;
    }

    #[inline(always)]
    unsafe fn swappz(x: &mut c64, y: &mut c64) {
        let z = Self::getpz(x);
        Self::setpz(x, Self::getpz(y));
        Self::setpz(y, z);
    }

    #[inline(always)]
    unsafe fn cnjpz(xy: Self::Xmm) -> Self::Xmm {
        XmmEmu(xy.0, -xy.1)
    }

    #[inline(always)]
    unsafe fn jxpz(xy: Self::Xmm) -> Self::Xmm {
        XmmEmu(-xy.1, xy.0)
    }

    #[inline(always)]
    unsafe fn negpz(xy: Self::Xmm) -> Self::Xmm {
        XmmEmu(-xy.0, -xy.1)
    }

    #[inline(always)]
    unsafe fn mjxpz(xy: Self::Xmm) -> Self::Xmm {
        XmmEmu(xy.1, -xy.0)
    }

    #[inline(always)]
    unsafe fn addpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm {
        XmmEmu(a.0 + b.0, a.1 + b.1)
    }

    #[inline(always)]
    unsafe fn subpz(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm {
        XmmEmu(a.0 - b.0, a.1 - b.1)
    }

    #[inline(always)]
    unsafe fn mulpd(a: Self::Xmm, b: Self::Xmm) -> Self::Xmm {
        XmmEmu(a.0 * b.0, a.1 * b.1)
    }

    #[inline(always)]
    unsafe fn haddpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm {
        XmmEmu(ab.0 + ab.1, xy.0 + xy.1)
    }

    #[inline(always)]
    unsafe fn mulpz(ab: Self::Xmm, xy: Self::Xmm) -> Self::Xmm {
        XmmEmu(ab.0 * xy.0 - ab.1 * xy.1, ab.0 * xy.1 + ab.1 * xy.0)
    }

    #[inline(always)]
    unsafe fn v8xpz(xy: Self::Xmm) -> Self::Xmm {
        XmmEmu(FRAC_1_SQRT_2 * (xy.0 - xy.1), FRAC_1_SQRT_2 * (xy.0 + xy.1))
    }

    #[inline(always)]
    unsafe fn w8xpz(xy: Self::Xmm) -> Self::Xmm {
        XmmEmu(FRAC_1_SQRT_2 * (xy.0 + xy.1), FRAC_1_SQRT_2 * (xy.1 - xy.0))
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
        true
    }
}

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

macro_rules! fft_simd2_emu {
    ($ty: ty) => {
        impl FftSimd2 for $ty {
            type Ymm = YmmEmu<Self>;

            #[inline(always)]
            unsafe fn cmplx2(a: f64, b: f64, c: f64, d: f64) -> Self::Ymm {
                YmmEmu(Self::cmplx(a, b), Self::cmplx(c, d))
            }

            #[inline(always)]
            unsafe fn getpz2(z: *const c64) -> Self::Ymm {
                YmmEmu(Self::getpz(&*z.add(0)), Self::getpz(&*z.add(1)))
            }

            #[inline(always)]
            unsafe fn setpz2(z: *mut c64, x: Self::Ymm) {
                Self::setpz(z.add(0), x.0);
                Self::setpz(z.add(1), x.1);
            }

            #[inline(always)]
            unsafe fn cnjpz2(xy: Self::Ymm) -> Self::Ymm {
                YmmEmu(Self::cnjpz(xy.0), Self::cnjpz(xy.1))
            }

            #[inline(always)]
            unsafe fn jxpz2(xy: Self::Ymm) -> Self::Ymm {
                YmmEmu(Self::jxpz(xy.0), Self::jxpz(xy.1))
            }

            #[inline(always)]
            unsafe fn negpz2(xy: Self::Ymm) -> Self::Ymm {
                YmmEmu(Self::negpz(xy.0), Self::negpz(xy.1))
            }

            #[inline(always)]
            unsafe fn addpz2(a: Self::Ymm, b: Self::Ymm) -> Self::Ymm {
                YmmEmu(Self::addpz(a.0, b.0), Self::addpz(a.1, b.1))
            }

            #[inline(always)]
            unsafe fn subpz2(a: Self::Ymm, b: Self::Ymm) -> Self::Ymm {
                YmmEmu(Self::subpz(a.0, b.0), Self::subpz(a.1, b.1))
            }

            #[inline(always)]
            unsafe fn mulpd2(a: Self::Ymm, b: Self::Ymm) -> Self::Ymm {
                YmmEmu(Self::mulpd(a.0, b.0), Self::mulpd(a.1, b.1))
            }

            #[inline(always)]
            unsafe fn mulpz2(ab: Self::Ymm, xy: Self::Ymm) -> Self::Ymm {
                YmmEmu(Self::mulpz(ab.0, xy.0), Self::mulpz(ab.1, xy.1))
            }

            #[inline(always)]
            unsafe fn v8xpz2(xy: Self::Ymm) -> Self::Ymm {
                YmmEmu(Self::v8xpz(xy.0), Self::v8xpz(xy.1))
            }

            #[inline(always)]
            unsafe fn w8xpz2(xy: Self::Ymm) -> Self::Ymm {
                YmmEmu(Self::w8xpz(xy.0), Self::w8xpz(xy.1))
            }

            #[inline(always)]
            unsafe fn h1xpz2(xy: Self::Ymm) -> Self::Ymm {
                YmmEmu(Self::h1xpz(xy.0), Self::h1xpz(xy.1))
            }

            #[inline(always)]
            unsafe fn h3xpz2(xy: Self::Ymm) -> Self::Ymm {
                YmmEmu(Self::h3xpz(xy.0), Self::h3xpz(xy.1))
            }

            #[inline(always)]
            unsafe fn hfxpz2(xy: Self::Ymm) -> Self::Ymm {
                YmmEmu(Self::hfxpz(xy.0), Self::hfxpz(xy.1))
            }

            #[inline(always)]
            unsafe fn hdxpz2(xy: Self::Ymm) -> Self::Ymm {
                YmmEmu(Self::hdxpz(xy.0), Self::hdxpz(xy.1))
            }

            #[inline(always)]
            unsafe fn duppz2(x: Self::Xmm) -> Self::Ymm {
                YmmEmu(x, x)
            }

            #[inline(always)]
            unsafe fn duppz3(z: &c64) -> Self::Ymm {
                Self::duppz2(Self::getpz(z))
            }

            #[inline(always)]
            unsafe fn cat(a: Self::Xmm, b: Self::Xmm) -> Self::Ymm {
                YmmEmu(a, b)
            }

            #[inline(always)]
            unsafe fn catlo(ax: Self::Ymm, by: Self::Ymm) -> Self::Ymm {
                YmmEmu(ax.0, by.0)
            }

            #[inline(always)]
            unsafe fn cathi(ax: Self::Ymm, by: Self::Ymm) -> Self::Ymm {
                YmmEmu(ax.1, by.1)
            }

            #[inline(always)]
            unsafe fn swaplohi(ab: Self::Ymm) -> Self::Ymm {
                YmmEmu(ab.1, ab.0)
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
                a_b.0
            }

            #[inline(always)]
            unsafe fn gethi(a_b: Self::Ymm) -> Self::Xmm {
                a_b.1
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
    };
}

fft_simd2_emu!(Scalar);
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

impl<S: FftSimd2> FftSimd4 for S {
    type Zmm = ZmmEmu<S>;

    #[inline(always)]
    unsafe fn getpz4(z: *const c64) -> Self::Zmm {
        ZmmEmu(Self::getpz2(z), Self::getpz2(z.add(2)))
    }

    #[inline(always)]
    unsafe fn setpz4(a: *mut c64, z: Self::Zmm) {
        Self::setpz2(a, z.0);
        Self::setpz2(a.add(2), z.1);
    }

    #[inline(always)]
    unsafe fn cnjpz4(xy: Self::Zmm) -> Self::Zmm {
        ZmmEmu(Self::cnjpz2(xy.0), Self::cnjpz2(xy.1))
    }

    #[inline(always)]
    unsafe fn jxpz4(xy: Self::Zmm) -> Self::Zmm {
        ZmmEmu(Self::jxpz2(xy.0), Self::jxpz2(xy.1))
    }

    #[inline(always)]
    unsafe fn addpz4(a: Self::Zmm, b: Self::Zmm) -> Self::Zmm {
        ZmmEmu(Self::addpz2(a.0, b.0), Self::addpz2(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn subpz4(a: Self::Zmm, b: Self::Zmm) -> Self::Zmm {
        ZmmEmu(Self::subpz2(a.0, b.0), Self::subpz2(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn mulpd4(a: Self::Zmm, b: Self::Zmm) -> Self::Zmm {
        ZmmEmu(Self::mulpd2(a.0, b.0), Self::mulpd2(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn mulpz4(a: Self::Zmm, b: Self::Zmm) -> Self::Zmm {
        ZmmEmu(Self::mulpz2(a.0, b.0), Self::mulpz2(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn v8xpz4(xy: Self::Zmm) -> Self::Zmm {
        ZmmEmu(Self::v8xpz2(xy.0), Self::v8xpz2(xy.1))
    }

    #[inline(always)]
    unsafe fn w8xpz4(xy: Self::Zmm) -> Self::Zmm {
        ZmmEmu(Self::w8xpz2(xy.0), Self::w8xpz2(xy.1))
    }

    #[inline(always)]
    unsafe fn h1xpz4(xy: Self::Zmm) -> Self::Zmm {
        ZmmEmu(Self::h1xpz2(xy.0), Self::h1xpz2(xy.1))
    }

    #[inline(always)]
    unsafe fn h3xpz4(xy: Self::Zmm) -> Self::Zmm {
        ZmmEmu(Self::h3xpz2(xy.0), Self::h3xpz2(xy.1))
    }

    #[inline(always)]
    unsafe fn hfxpz4(xy: Self::Zmm) -> Self::Zmm {
        ZmmEmu(Self::hfxpz2(xy.0), Self::hfxpz2(xy.1))
    }

    #[inline(always)]
    unsafe fn hdxpz4(xy: Self::Zmm) -> Self::Zmm {
        ZmmEmu(Self::hdxpz2(xy.0), Self::hdxpz2(xy.1))
    }

    #[inline(always)]
    unsafe fn duppz4(x: Self::Xmm) -> Self::Zmm {
        let y = Self::duppz2(x);
        ZmmEmu(y, y)
    }

    #[inline(always)]
    unsafe fn duppz5(x: &c64) -> Self::Zmm {
        Self::duppz4(Self::getpz(x))
    }
}

impl<S: FftSimd4> FftSimd8 for S {
    type Amm = AmmEmu<S>;

    #[inline(always)]
    unsafe fn getpz8(z: *const c64) -> Self::Amm {
        AmmEmu(Self::getpz4(z), Self::getpz4(z.add(4)))
    }

    #[inline(always)]
    unsafe fn setpz8(a: *mut c64, z: Self::Amm) {
        Self::setpz4(a, z.0);
        Self::setpz4(a.add(4), z.1);
    }

    #[inline(always)]
    unsafe fn cnjpz8(xy: Self::Amm) -> Self::Amm {
        AmmEmu(Self::cnjpz4(xy.0), Self::cnjpz4(xy.1))
    }

    #[inline(always)]
    unsafe fn jxpz8(xy: Self::Amm) -> Self::Amm {
        AmmEmu(Self::jxpz4(xy.0), Self::jxpz4(xy.1))
    }

    #[inline(always)]
    unsafe fn addpz8(a: Self::Amm, b: Self::Amm) -> Self::Amm {
        AmmEmu(Self::addpz4(a.0, b.0), Self::addpz4(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn subpz8(a: Self::Amm, b: Self::Amm) -> Self::Amm {
        AmmEmu(Self::subpz4(a.0, b.0), Self::subpz4(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn mulpd8(a: Self::Amm, b: Self::Amm) -> Self::Amm {
        AmmEmu(Self::mulpd4(a.0, b.0), Self::mulpd4(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn mulpz8(a: Self::Amm, b: Self::Amm) -> Self::Amm {
        AmmEmu(Self::mulpz4(a.0, b.0), Self::mulpz4(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn v8xpz8(xy: Self::Amm) -> Self::Amm {
        AmmEmu(Self::v8xpz4(xy.0), Self::v8xpz4(xy.1))
    }

    #[inline(always)]
    unsafe fn w8xpz8(xy: Self::Amm) -> Self::Amm {
        AmmEmu(Self::w8xpz4(xy.0), Self::w8xpz4(xy.1))
    }

    #[inline(always)]
    unsafe fn h1xpz8(xy: Self::Amm) -> Self::Amm {
        AmmEmu(Self::h1xpz4(xy.0), Self::h1xpz4(xy.1))
    }

    #[inline(always)]
    unsafe fn h3xpz8(xy: Self::Amm) -> Self::Amm {
        AmmEmu(Self::h3xpz4(xy.0), Self::h3xpz4(xy.1))
    }

    #[inline(always)]
    unsafe fn hfxpz8(xy: Self::Amm) -> Self::Amm {
        AmmEmu(Self::hfxpz4(xy.0), Self::hfxpz4(xy.1))
    }

    #[inline(always)]
    unsafe fn hdxpz8(xy: Self::Amm) -> Self::Amm {
        AmmEmu(Self::hdxpz4(xy.0), Self::hdxpz4(xy.1))
    }

    #[inline(always)]
    unsafe fn duppz8(x: Self::Xmm) -> Self::Amm {
        let y = Self::duppz4(x);
        AmmEmu(y, y)
    }

    #[inline(always)]
    unsafe fn duppz9(x: &c64) -> Self::Amm {
        Self::duppz8(Self::getpz(x))
    }
}

impl<S: FftSimd8> FftSimd16 for S {
    type Bmm = BmmEmu<S>;

    #[inline(always)]
    unsafe fn getpz16(z: *const c64) -> Self::Bmm {
        BmmEmu(Self::getpz8(z), Self::getpz8(z.add(8)))
    }

    #[inline(always)]
    unsafe fn setpz16(a: *mut c64, z: Self::Bmm) {
        Self::setpz8(a, z.0);
        Self::setpz8(a.add(8), z.1);
    }

    #[inline(always)]
    unsafe fn cnjpz16(xy: Self::Bmm) -> Self::Bmm {
        BmmEmu(Self::cnjpz8(xy.0), Self::cnjpz8(xy.1))
    }

    #[inline(always)]
    unsafe fn jxpz16(xy: Self::Bmm) -> Self::Bmm {
        BmmEmu(Self::jxpz8(xy.0), Self::jxpz8(xy.1))
    }

    #[inline(always)]
    unsafe fn addpz16(a: Self::Bmm, b: Self::Bmm) -> Self::Bmm {
        BmmEmu(Self::addpz8(a.0, b.0), Self::addpz8(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn subpz16(a: Self::Bmm, b: Self::Bmm) -> Self::Bmm {
        BmmEmu(Self::subpz8(a.0, b.0), Self::subpz8(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn mulpd16(a: Self::Bmm, b: Self::Bmm) -> Self::Bmm {
        BmmEmu(Self::mulpd8(a.0, b.0), Self::mulpd8(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn mulpz16(a: Self::Bmm, b: Self::Bmm) -> Self::Bmm {
        BmmEmu(Self::mulpz8(a.0, b.0), Self::mulpz8(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn v8xpz16(xy: Self::Bmm) -> Self::Bmm {
        BmmEmu(Self::v8xpz8(xy.0), Self::v8xpz8(xy.1))
    }

    #[inline(always)]
    unsafe fn w8xpz16(xy: Self::Bmm) -> Self::Bmm {
        BmmEmu(Self::w8xpz8(xy.0), Self::w8xpz8(xy.1))
    }

    #[inline(always)]
    unsafe fn h1xpz16(xy: Self::Bmm) -> Self::Bmm {
        BmmEmu(Self::h1xpz8(xy.0), Self::h1xpz8(xy.1))
    }

    #[inline(always)]
    unsafe fn h3xpz16(xy: Self::Bmm) -> Self::Bmm {
        BmmEmu(Self::h3xpz8(xy.0), Self::h3xpz8(xy.1))
    }

    #[inline(always)]
    unsafe fn hfxpz16(xy: Self::Bmm) -> Self::Bmm {
        BmmEmu(Self::hfxpz8(xy.0), Self::hfxpz8(xy.1))
    }

    #[inline(always)]
    unsafe fn hdxpz16(xy: Self::Bmm) -> Self::Bmm {
        BmmEmu(Self::hdxpz8(xy.0), Self::hdxpz8(xy.1))
    }

    #[inline(always)]
    unsafe fn duppz16(x: Self::Xmm) -> Self::Bmm {
        let y = Self::duppz8(x);
        BmmEmu(y, y)
    }

    #[inline(always)]
    unsafe fn duppz17(x: &c64) -> Self::Bmm {
        Self::duppz16(Self::getpz(x))
    }
}
