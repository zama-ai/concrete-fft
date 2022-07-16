use crate::c64;
use core::fmt::Debug;
use core::mem::transmute;

const H1X: f64 = 0.923879532511286762010323247995557949;
const H1Y: f64 = -0.382683432365089757574419179753100195;

pub trait FftSimd64 {
    type Reg: Copy + Debug;
    const COMPLEX_PER_REG: usize;

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

pub trait FftSimd64Ext: FftSimd64 {
    #[inline(always)]
    unsafe fn conj(xy: Self::Reg) -> Self::Reg {
        let mask = Self::splat(&c64 { re: 0.0, im: -0.0 });
        Self::xor(xy, mask)
    }

    #[inline(always)]
    unsafe fn xpj(fwd: bool, xy: Self::Reg) -> Self::Reg {
        if fwd {
            Self::swap_re_im(Self::conj(xy))
        } else {
            Self::conj(Self::swap_re_im(xy))
        }
    }

    #[inline(always)]
    unsafe fn xmj(fwd: bool, xy: Self::Reg) -> Self::Reg {
        Self::xpj(!fwd, xy)
    }

    #[inline(always)]
    unsafe fn xv8(fwd: bool, xy: Self::Reg) -> Self::Reg {
        let r = Self::splat_re_im(&core::f64::consts::FRAC_1_SQRT_2);
        Self::cwise_mul(r, Self::add(xy, Self::xpj(fwd, xy)))
    }

    #[inline(always)]
    unsafe fn xw8(fwd: bool, xy: Self::Reg) -> Self::Reg {
        Self::xv8(!fwd, xy)
    }

    #[inline(always)]
    unsafe fn xh1(fwd: bool, xy: Self::Reg) -> Self::Reg {
        if fwd {
            Self::mul(Self::splat(&c64 { re: H1X, im: H1Y }), xy)
        } else {
            Self::mul(Self::splat(&c64 { re: H1X, im: -H1Y }), xy)
        }
    }

    #[inline(always)]
    unsafe fn xh3(fwd: bool, xy: Self::Reg) -> Self::Reg {
        if fwd {
            Self::mul(Self::splat(&c64 { re: -H1Y, im: -H1X }), xy)
        } else {
            Self::mul(Self::splat(&c64 { re: -H1Y, im: H1X }), xy)
        }
    }

    #[inline(always)]
    unsafe fn xhf(fwd: bool, xy: Self::Reg) -> Self::Reg {
        Self::xh1(!fwd, xy)
    }

    #[inline(always)]
    unsafe fn xhd(fwd: bool, xy: Self::Reg) -> Self::Reg {
        Self::xh3(!fwd, xy)
    }
}

impl<T: FftSimd64> FftSimd64Ext for T {}

pub trait FftSimd64X2: FftSimd64 {
    unsafe fn catlo(a: Self::Reg, b: Self::Reg) -> Self::Reg;
    unsafe fn cathi(a: Self::Reg, b: Self::Reg) -> Self::Reg;
}

#[derive(Copy, Clone, Debug)]
pub struct Scalar;

impl FftSimd64 for Scalar {
    type Reg = c64;

    const COMPLEX_PER_REG: usize = 1;

    #[inline(always)]
    unsafe fn splat_re_im(ptr: *const f64) -> Self::Reg {
        c64 { re: *ptr, im: *ptr }
    }

    #[inline(always)]
    unsafe fn splat(ptr: *const c64) -> Self::Reg {
        *ptr
    }

    #[inline(always)]
    unsafe fn load(ptr: *const c64) -> Self::Reg {
        *ptr
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut c64, z: Self::Reg) {
        *ptr = z;
    }

    #[inline(always)]
    unsafe fn xor(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        transmute(transmute::<c64, u128>(a) ^ transmute::<c64, u128>(b))
    }

    #[inline(always)]
    unsafe fn swap_re_im(xy: Self::Reg) -> Self::Reg {
        Self::Reg {
            re: xy.im,
            im: xy.re,
        }
    }

    #[inline(always)]
    unsafe fn add(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        a + b
    }

    #[inline(always)]
    unsafe fn sub(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        a - b
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        a * b
    }

    #[inline(always)]
    unsafe fn cwise_mul(a: Self::Reg, b: Self::Reg) -> Self::Reg {
        Self::Reg {
            re: a.re * b.re,
            im: a.im * b.im,
        }
    }
}

#[inline(always)]
pub unsafe fn twid(r: usize, big_n: usize, k: usize, w: *const c64, p: usize) -> &'static c64 {
    &*w.add(p + (k - 1) * (big_n / r))
}

#[inline(always)]
pub unsafe fn twid_t(r: usize, big_n: usize, k: usize, w: *const c64, p: usize) -> &'static c64 {
    &*w.add(r * p + (big_n + k))
}

pub fn init_wt(forward: bool, r: usize, big_n: usize, w: &mut [c64]) {
    if big_n < r {
        return;
    }

    let nr = big_n / r;
    let theta = -2.0 * core::f64::consts::PI / big_n as f64;

    for i in 0..2 * big_n {
        w[i].re = f64::NAN;
        w[i].im = f64::NAN;
    }

    for p in 0..nr {
        for k in 1..r {
            let (s, c) = (theta * (k * p) as f64).sin_cos();
            let z = c64::new(c, if forward { s } else { -s });
            w[p + (k - 1) * nr] = z;
            w[big_n + r * p + k] = z;
        }
    }
}
