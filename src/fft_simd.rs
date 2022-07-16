use crate::c64;
use core::fmt::Debug;
use core::mem::transmute;
use hexf::hexf64;

// cos(-pi/8)
pub const H1X: f64 = hexf64!("0x0.ec835e79946a3p0");
// sin(-pi/8)
pub const H1Y: f64 = hexf64!("0x0.61f78a9abaa59p0") * -1.0;

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

// https://stackoverflow.com/a/42792940
fn sincospi64(mut a: f64) -> (f64, f64) {
    let fma = f64::mul_add;

    // must be evaluated with IEEE-754 semantics
    let az = a * 0.0;

    // for |a| >= 2**53, cospi(a) = 1.0, but cospi(Inf) = NaN
    a = if a.abs() < hexf64!("0x1.0p53") { a } else { az };

    // reduce argument to primary approximation interval (-0.25, 0.25)
    let mut r = (a + a).round();
    let i = r as i64;
    let t = f64::mul_add(-0.5, r, a);

    // compute core approximations
    let s = t * t;

    // approximate cos(pi*x) for x in [-0.25,0.25]

    r = -1.0369917389758117e-4;
    r = fma(r, s, 1.9294935641298806e-3);
    r = fma(r, s, -2.5806887942825395e-2);
    r = fma(r, s, 2.3533063028328211e-1);
    r = fma(r, s, -1.3352627688538006e+0);
    r = fma(r, s, 4.0587121264167623e+0);
    r = fma(r, s, -4.9348022005446790e+0);
    let mut c = fma(r, s, 1.0000000000000000e+0);

    // approximate sin(pi*x) for x in [-0.25,0.25]
    r = 4.6151442520157035e-4;
    r = fma(r, s, -7.3700183130883555e-3);
    r = fma(r, s, 8.2145868949323936e-2);
    r = fma(r, s, -5.9926452893214921e-1);
    r = fma(r, s, 2.5501640398732688e+0);
    r = fma(r, s, -5.1677127800499516e+0);
    let s = s * t;
    r = r * s;

    let mut s = fma(t, 3.1415926535897931e+0, r);
    // map results according to quadrant

    if (i & 2) != 0 {
        s = 0.0 - s; // must be evaluated with IEEE-754 semantics
        c = 0.0 - c; // must be evaluated with IEEE-754 semantics
    }
    if (i & 1) != 0 {
        let t = 0.0 - s; // must be evaluated with IEEE-754 semantics
        s = c;
        c = t;
    }
    // IEEE-754: sinPi(+n) is +0 and sinPi(-n) is -0 for positive integers n
    if a == a.floor() {
        s = az
    }
    (s, c)
}

pub fn init_wt(r: usize, big_n: usize, w: &mut [c64], w_inv: &mut [c64]) {
    if big_n < r {
        return;
    }

    let nr = big_n / r;
    let theta = -2.0 / big_n as f64;

    for i in 0..2 * big_n {
        w[i].re = f64::NAN;
        w[i].im = f64::NAN;
    }

    for p in 0..nr {
        for k in 1..r {
            let (s, c) = sincospi64(theta * (k * p) as f64);
            let z = c64::new(c, s);
            w[p + (k - 1) * nr] = z;
            w[big_n + r * p + k] = z;
            w_inv[p + (k - 1) * nr] = z.conj();
            w_inv[big_n + r * p + k] = z.conj();
        }
    }
}
