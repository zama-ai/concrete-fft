use crate::c64;
use crate::fft_simd::{FftSimd64, FftSimd64X2};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[derive(Copy, Clone, Debug)]
pub struct AvxX2;
#[derive(Copy, Clone, Debug)]
pub struct AvxX1;

#[derive(Copy, Clone, Debug)]
pub struct FmaX2;
#[derive(Copy, Clone, Debug)]
pub struct FmaX1;

#[cfg(feature = "nightly")]
pub struct Avx512X4;
#[cfg(feature = "nightly")]
pub struct Avx512X2;
#[cfg(feature = "nightly")]
pub struct Avx512X1;

macro_rules! reimpl {
    (
        as $ty: ty:
        $(
            unsafe fn $name: ident($($arg_name: ident: $arg_ty: ty),* $(,)?) $(-> $ret: ty)?;
        )*
    ) => {
        $(
            #[inline(always)]
            unsafe fn $name($($arg_name: $arg_ty),*) $(-> $ret)? {
                <$ty>::$name($($arg_name),*)
            }
        )*
    };
}

impl FftSimd64 for AvxX1 {
    type Reg = __m128d;

    const COMPLEX_PER_REG: usize = 1;

    #[inline(always)]
    unsafe fn splat_re_im(ptr: *const f64) -> Self::Reg {
        _mm_set1_pd(*ptr)
    }

    #[inline(always)]
    unsafe fn splat(ptr: *const crate::c64) -> Self::Reg {
        Self::load(ptr)
    }

    #[inline(always)]
    unsafe fn load(ptr: *const crate::c64) -> Self::Reg {
        _mm_loadu_pd(ptr as _)
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut crate::c64, z: Self::Reg) {
        _mm_storeu_pd(ptr as _, z);
    }

    #[inline(always)]
    unsafe fn xor(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm_xor_pd(a, b)
    }

    #[inline(always)]
    unsafe fn swap_re_im(xy: Self::Reg) -> Self::Reg {
        _mm_permute_pd::<0b01>(xy)
    }

    #[inline(always)]
    unsafe fn add(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm_add_pd(a, b)
    }

    #[inline(always)]
    unsafe fn sub(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm_sub_pd(a, b)
    }

    #[inline(always)]
    unsafe fn cwise_mul(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm_mul_pd(a, b)
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        let ab = a;
        let xy = b;
        let aa = _mm_unpacklo_pd(ab, ab);
        let bb = _mm_unpackhi_pd(ab, ab);
        let yx = Self::swap_re_im(xy);
        _mm_addsub_pd(_mm_mul_pd(aa, xy), _mm_mul_pd(bb, yx))
    }
}

impl FftSimd64 for AvxX2 {
    type Reg = __m256d;

    const COMPLEX_PER_REG: usize = 2;

    #[inline(always)]
    unsafe fn splat_re_im(ptr: *const f64) -> Self::Reg {
        _mm256_set1_pd(*ptr)
    }

    #[inline(always)]
    unsafe fn splat(ptr: *const crate::c64) -> Self::Reg {
        let tmp = _mm_loadu_pd(ptr as _);
        _mm256_broadcast_pd(&tmp)
    }

    #[inline(always)]
    unsafe fn load(ptr: *const crate::c64) -> Self::Reg {
        _mm256_loadu_pd(ptr as _)
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut crate::c64, z: Self::Reg) {
        _mm256_storeu_pd(ptr as _, z);
    }

    #[inline(always)]
    unsafe fn xor(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm256_xor_pd(a, b)
    }

    #[inline(always)]
    unsafe fn swap_re_im(xy: Self::Reg) -> Self::Reg {
        _mm256_permute_pd::<0b0101>(xy)
    }

    #[inline(always)]
    unsafe fn add(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm256_add_pd(a, b)
    }

    #[inline(always)]
    unsafe fn sub(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm256_sub_pd(a, b)
    }

    #[inline(always)]
    unsafe fn cwise_mul(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm256_mul_pd(a, b)
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        let ab = a;
        let xy = b;
        let aa = _mm256_unpacklo_pd(ab, ab);
        let bb = _mm256_unpackhi_pd(ab, ab);
        let yx = Self::swap_re_im(xy);
        _mm256_addsub_pd(_mm256_mul_pd(aa, xy), _mm256_mul_pd(bb, yx))
    }
}

impl FftSimd64 for FmaX1 {
    type Reg = __m128d;

    const COMPLEX_PER_REG: usize = 1;

    reimpl! { as AvxX1:
        unsafe fn splat_re_im(ptr: *const f64) -> Self::Reg;
        unsafe fn splat(ptr: *const c64) -> Self::Reg;
        unsafe fn load(ptr: *const c64) -> Self::Reg;
        unsafe fn store(ptr: *mut c64, z: Self::Reg);
        unsafe fn xor(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn swap_re_im(xy: Self::Reg) -> Self::Reg;
        unsafe fn add(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn sub(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn cwise_mul(a: Self::Reg, b: Self::Reg) -> Self::Reg;
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        let ab = a;
        let xy = b;
        let aa = _mm_unpacklo_pd(ab, ab);
        let bb = _mm_unpackhi_pd(ab, ab);
        let yx = Self::swap_re_im(xy);
        _mm_fmaddsub_pd(aa, xy, _mm_mul_pd(bb, yx))
    }
}

impl FftSimd64 for FmaX2 {
    type Reg = __m256d;

    const COMPLEX_PER_REG: usize = 2;

    reimpl! { as AvxX2:
        unsafe fn splat_re_im(ptr: *const f64) -> Self::Reg;
        unsafe fn splat(ptr: *const c64) -> Self::Reg;
        unsafe fn load(ptr: *const c64) -> Self::Reg;
        unsafe fn store(ptr: *mut c64, z: Self::Reg);
        unsafe fn xor(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn swap_re_im(xy: Self::Reg) -> Self::Reg;
        unsafe fn add(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn sub(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn cwise_mul(a: Self::Reg, b: Self::Reg) -> Self::Reg;
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        let ab = a;
        let xy = b;
        let aa = _mm256_unpacklo_pd(ab, ab);
        let bb = _mm256_unpackhi_pd(ab, ab);
        let yx = Self::swap_re_im(xy);
        _mm256_fmaddsub_pd(aa, xy, _mm256_mul_pd(bb, yx))
    }
}

#[cfg(feature = "nightly")]
impl FftSimd64 for Avx512X1 {
    type Reg = __m128d;

    const COMPLEX_PER_REG: usize = 1;

    reimpl! { as FmaX1:
        unsafe fn splat_re_im(ptr: *const f64) -> Self::Reg;
        unsafe fn splat(ptr: *const c64) -> Self::Reg;
        unsafe fn load(ptr: *const c64) -> Self::Reg;
        unsafe fn store(ptr: *mut c64, z: Self::Reg);
        unsafe fn xor(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn swap_re_im(xy: Self::Reg) -> Self::Reg;
        unsafe fn add(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn sub(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn cwise_mul(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn mul(a: Self::Reg, b: Self::Reg) -> Self::Reg;
    }
}

#[cfg(feature = "nightly")]
impl FftSimd64 for Avx512X2 {
    type Reg = __m256d;

    const COMPLEX_PER_REG: usize = 2;

    reimpl! { as FmaX2:
        unsafe fn splat_re_im(ptr: *const f64) -> Self::Reg;
        unsafe fn splat(ptr: *const c64) -> Self::Reg;
        unsafe fn load(ptr: *const c64) -> Self::Reg;
        unsafe fn store(ptr: *mut c64, z: Self::Reg);
        unsafe fn xor(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn swap_re_im(xy: Self::Reg) -> Self::Reg;
        unsafe fn add(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn sub(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn cwise_mul(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn mul(a: Self::Reg, b: Self::Reg) -> Self::Reg;
    }
}

#[cfg(feature = "nightly")]
impl FftSimd64 for Avx512X4 {
    type Reg = __m512d;

    const COMPLEX_PER_REG: usize = 4;

    #[inline(always)]
    unsafe fn splat_re_im(ptr: *const f64) -> Self::Reg {
        _mm512_set1_pd(*ptr)
    }

    #[inline(always)]
    unsafe fn splat(ptr: *const crate::c64) -> Self::Reg {
        _mm512_castps_pd(_mm512_broadcast_f32x4(_mm_castpd_ps(_mm_loadu_pd(
            ptr as _,
        ))))
    }

    #[inline(always)]
    unsafe fn load(ptr: *const crate::c64) -> Self::Reg {
        _mm512_loadu_pd(ptr as _)
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut crate::c64, z: Self::Reg) {
        _mm512_storeu_pd(ptr as _, z);
    }

    #[inline(always)]
    unsafe fn xor(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm512_castsi512_pd(_mm512_xor_si512(
            _mm512_castpd_si512(a),
            _mm512_castpd_si512(b),
        ))
    }

    #[inline(always)]
    unsafe fn swap_re_im(xy: Self::Reg) -> Self::Reg {
        _mm512_permute_pd::<0b01010101>(xy)
    }

    #[inline(always)]
    unsafe fn add(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm512_add_pd(a, b)
    }

    #[inline(always)]
    unsafe fn sub(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm512_sub_pd(a, b)
    }

    #[inline(always)]
    unsafe fn cwise_mul(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm512_mul_pd(a, b)
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        let ab = a;
        let xy = b;
        let aa = _mm512_unpacklo_pd(ab, ab);
        let bb = _mm512_unpackhi_pd(ab, ab);
        let yx = Self::swap_re_im(xy);
        _mm512_fmaddsub_pd(aa, xy, _mm512_mul_pd(bb, yx))
    }
}

impl FftSimd64X2 for AvxX2 {
    #[inline(always)]
    unsafe fn catlo(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm256_permute2f128_pd::<0b00100000>(a, b)
    }

    #[inline(always)]
    unsafe fn cathi(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm256_permute2f128_pd::<0b00110001>(a, b)
    }
}

impl FftSimd64X2 for FmaX2 {
    reimpl! { as AvxX2:
        unsafe fn catlo(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn cathi(a: Self::Reg, b: Self::Reg) -> Self::Reg;
    }
}

#[cfg(feature = "nightly")]
impl FftSimd64X2 for Avx512X2 {
    reimpl! { as AvxX2:
        unsafe fn catlo(a: Self::Reg, b: Self::Reg) -> Self::Reg;
        unsafe fn cathi(a: Self::Reg, b: Self::Reg) -> Self::Reg;
    }
}

#[cfg(feature = "nightly")]
impl crate::fft_simd::FftSimd64X4 for Avx512X4 {
    #[inline(always)]
    unsafe fn transpose(
        r0: Self::Reg,
        r1: Self::Reg,
        r2: Self::Reg,
        r3: Self::Reg,
    ) -> (Self::Reg, Self::Reg, Self::Reg, Self::Reg) {
        let t0 = _mm512_shuffle_f64x2::<0b10001000>(r0, r1);
        let t1 = _mm512_shuffle_f64x2::<0b11011101>(r0, r1);
        let t2 = _mm512_shuffle_f64x2::<0b10001000>(r2, r3);
        let t3 = _mm512_shuffle_f64x2::<0b11011101>(r2, r3);

        let s0 = _mm512_shuffle_f64x2::<0b10001000>(t0, t2);
        let s1 = _mm512_shuffle_f64x2::<0b11011101>(t0, t2);
        let s2 = _mm512_shuffle_f64x2::<0b10001000>(t1, t3);
        let s3 = _mm512_shuffle_f64x2::<0b11011101>(t1, t3);

        (s0, s2, s1, s3)
    }
}
