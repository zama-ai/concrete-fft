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
        let yx = _mm_shuffle_pd::<0b0101>(xy, xy);
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
        let yx = _mm256_shuffle_pd::<0b0101>(xy, xy);
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
        let yx = _mm_shuffle_pd::<0b0101>(xy, xy);
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
        let yx = _mm256_shuffle_pd::<0b0101>(xy, xy);
        _mm256_fmaddsub_pd(aa, xy, _mm256_mul_pd(bb, yx))
    }
}

impl FftSimd64X2 for AvxX2 {
    unsafe fn catlo(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        _mm256_permute2f128_pd::<0b00100000>(a, b)
    }

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
