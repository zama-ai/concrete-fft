// Copyright (c) 2019 OK Ojisan(Takuya OKAHISA)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use crate::c64;
use crate::dif2::end_2;
use crate::dif4::end_4;
use crate::fft_simd::{twid, twid_t, FftSimd64, FftSimd64Ext, FftSimd64X2, FftSimd64X4, Scalar};
use crate::x86_feature_detected;

#[inline(always)]
#[rustfmt::skip]
unsafe fn core_<I: FftSimd64>(
    fwd: bool,
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    w: *const c64,
) {
    debug_assert_eq!(s % I::COMPLEX_PER_REG, 0);

    let m = n / 8;
    let big_n = n * s;
    let big_n0 = 0;
    let big_n1 = big_n / 8;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;
    let big_n4 = big_n1 * 4;
    let big_n5 = big_n1 * 5;
    let big_n6 = big_n1 * 6;
    let big_n7 = big_n1 * 7;

    for p in 0..m {
        let sp = s * p;
        let s8p = 8 * sp;
        let w1p = I::splat(twid_t(8, big_n, 1, w, sp));
        let w2p = I::splat(twid_t(8, big_n, 2, w, sp));
        let w3p = I::splat(twid_t(8, big_n, 3, w, sp));
        let w4p = I::splat(twid_t(8, big_n, 4, w, sp));
        let w5p = I::splat(twid_t(8, big_n, 5, w, sp));
        let w6p = I::splat(twid_t(8, big_n, 6, w, sp));
        let w7p = I::splat(twid_t(8, big_n, 7, w, sp));

        let mut q = 0;
        while q < s {
            let xq_sp = x.add(q + sp);
            let yq_s8p = y.add(q + s8p);

            let x0 = I::load(xq_sp.add(big_n0));
            let x1 = I::load(xq_sp.add(big_n1));
            let x2 = I::load(xq_sp.add(big_n2));
            let x3 = I::load(xq_sp.add(big_n3));
            let x4 = I::load(xq_sp.add(big_n4));
            let x5 = I::load(xq_sp.add(big_n5));
            let x6 = I::load(xq_sp.add(big_n6));
            let x7 = I::load(xq_sp.add(big_n7));

            let a04 = I::add(x0, x4);
            let s04 = I::sub(x0, x4);
            let a26 = I::add(x2, x6);
            let js26 = I::xpj(fwd, I::sub(x2, x6));
            let a15 = I::add(x1, x5);
            let s15 = I::sub(x1, x5);
            let a37 = I::add(x3, x7);
            let js37 = I::xpj(fwd, I::sub(x3, x7));
            let a04_p1_a26 = I::add(a04, a26);
            let s04_mj_s26 = I::sub(s04, js26);
            let a04_m1_a26 = I::sub(a04, a26);
            let s04_pj_s26 = I::add(s04, js26);
            let a15_p1_a37 = I::add(a15, a37);
            let w8_s15_mj_s37 = I::xw8(fwd, I::sub(s15, js37));
            let j_a15_m1_a37 = I::xpj(fwd, I::sub(a15, a37));
            let v8_s15_pj_s37 = I::xv8(fwd, I::add(s15, js37));

            I::store(yq_s8p.add(s * 0), I::add(a04_p1_a26, a15_p1_a37));
            I::store(yq_s8p.add(s * 1), I::mul(w1p, I::add(s04_mj_s26, w8_s15_mj_s37)));
            I::store(yq_s8p.add(s * 2), I::mul(w2p, I::sub(a04_m1_a26, j_a15_m1_a37)));
            I::store(yq_s8p.add(s * 3), I::mul(w3p, I::sub(s04_pj_s26, v8_s15_pj_s37)));
            I::store(yq_s8p.add(s * 4), I::mul(w4p, I::sub(a04_p1_a26, a15_p1_a37)));
            I::store(yq_s8p.add(s * 5), I::mul(w5p, I::sub(s04_mj_s26, w8_s15_mj_s37)));
            I::store(yq_s8p.add(s * 6), I::mul(w6p, I::add(a04_m1_a26, j_a15_m1_a37)));
            I::store(yq_s8p.add(s * 7), I::mul(w7p, I::add(s04_pj_s26, v8_s15_pj_s37)));

            q += I::COMPLEX_PER_REG;
        }
    }
}

#[inline(always)]
#[allow(dead_code)]
unsafe fn core_x2<I: FftSimd64X2>(
    fwd: bool,
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    w: *const c64,
) {
    debug_assert_eq!(s, 1);

    let big_n = n;
    let big_n0 = 0;
    let big_n1 = big_n / 8;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;
    let big_n4 = big_n1 * 4;
    let big_n5 = big_n1 * 5;
    let big_n6 = big_n1 * 6;
    let big_n7 = big_n1 * 7;

    debug_assert_eq!(big_n1 % 2, 0);
    let mut p = 0;
    while p < big_n1 {
        let x_p = x.add(p);
        let y_8p = y.add(8 * p);

        let x0 = I::load(x_p.add(big_n0));
        let x1 = I::load(x_p.add(big_n1));
        let x2 = I::load(x_p.add(big_n2));
        let x3 = I::load(x_p.add(big_n3));
        let x4 = I::load(x_p.add(big_n4));
        let x5 = I::load(x_p.add(big_n5));
        let x6 = I::load(x_p.add(big_n6));
        let x7 = I::load(x_p.add(big_n7));

        let a04 = I::add(x0, x4);
        let s04 = I::sub(x0, x4);
        let a26 = I::add(x2, x6);
        let js26 = I::xpj(fwd, I::sub(x2, x6));
        let a15 = I::add(x1, x5);
        let s15 = I::sub(x1, x5);
        let a37 = I::add(x3, x7);
        let js37 = I::xpj(fwd, I::sub(x3, x7));

        let a04_p1_a26 = I::add(a04, a26);
        let s04_mj_s26 = I::sub(s04, js26);
        let a04_m1_a26 = I::sub(a04, a26);
        let s04_pj_s26 = I::add(s04, js26);
        let a15_p1_a37 = I::add(a15, a37);
        let w8_s15_mj_s37 = I::xw8(fwd, I::sub(s15, js37));
        let j_a15_m1_a37 = I::xpj(fwd, I::sub(a15, a37));
        let v8_s15_pj_s37 = I::xv8(fwd, I::add(s15, js37));

        let w1p = I::load(twid(8, big_n, 1, w, p));
        let w2p = I::load(twid(8, big_n, 2, w, p));
        let w3p = I::load(twid(8, big_n, 3, w, p));
        let w4p = I::load(twid(8, big_n, 4, w, p));
        let w5p = I::load(twid(8, big_n, 5, w, p));
        let w6p = I::load(twid(8, big_n, 6, w, p));
        let w7p = I::load(twid(8, big_n, 7, w, p));

        let aa = I::add(a04_p1_a26, a15_p1_a37);
        let bb = I::mul(w1p, I::add(s04_mj_s26, w8_s15_mj_s37));
        let cc = I::mul(w2p, I::sub(a04_m1_a26, j_a15_m1_a37));
        let dd = I::mul(w3p, I::sub(s04_pj_s26, v8_s15_pj_s37));
        let ee = I::mul(w4p, I::sub(a04_p1_a26, a15_p1_a37));
        let ff = I::mul(w5p, I::sub(s04_mj_s26, w8_s15_mj_s37));
        let gg = I::mul(w6p, I::add(a04_m1_a26, j_a15_m1_a37));
        let hh = I::mul(w7p, I::add(s04_pj_s26, v8_s15_pj_s37));

        {
            let ab = I::catlo(aa, bb);
            I::store(y_8p.add(0), ab);
            let cd = I::catlo(cc, dd);
            I::store(y_8p.add(2), cd);
            let ef = I::catlo(ee, ff);
            I::store(y_8p.add(4), ef);
            let gh = I::catlo(gg, hh);
            I::store(y_8p.add(6), gh);
        }
        {
            let ab = I::cathi(aa, bb);
            I::store(y_8p.add(8), ab);
            let cd = I::cathi(cc, dd);
            I::store(y_8p.add(10), cd);
            let ef = I::cathi(ee, ff);
            I::store(y_8p.add(12), ef);
            let gh = I::cathi(gg, hh);
            I::store(y_8p.add(14), gh);
        }

        p += 2;
    }
}

#[inline(always)]
#[allow(dead_code)]
unsafe fn core_x4<I2: FftSimd64X2, I4: FftSimd64X4>(
    fwd: bool,
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    w: *const c64,
) {
    debug_assert_eq!(s, 1);
    if n == 16 {
        return core_x2::<I2>(fwd, n, s, x, y, w);
    }

    let big_n = n;
    let big_n0 = 0;
    let big_n1 = big_n / 8;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;
    let big_n4 = big_n1 * 4;
    let big_n5 = big_n1 * 5;
    let big_n6 = big_n1 * 6;
    let big_n7 = big_n1 * 7;

    debug_assert_eq!(big_n1 % 4, 0);
    let mut p = 0;
    while p < big_n1 {
        let x_p = x.add(p);
        let y_8p = y.add(8 * p);

        let x0 = I4::load(x_p.add(big_n0));
        let x1 = I4::load(x_p.add(big_n1));
        let x2 = I4::load(x_p.add(big_n2));
        let x3 = I4::load(x_p.add(big_n3));
        let x4 = I4::load(x_p.add(big_n4));
        let x5 = I4::load(x_p.add(big_n5));
        let x6 = I4::load(x_p.add(big_n6));
        let x7 = I4::load(x_p.add(big_n7));

        let a04 = I4::add(x0, x4);
        let s04 = I4::sub(x0, x4);
        let a26 = I4::add(x2, x6);
        let js26 = I4::xpj(fwd, I4::sub(x2, x6));
        let a15 = I4::add(x1, x5);
        let s15 = I4::sub(x1, x5);
        let a37 = I4::add(x3, x7);
        let js37 = I4::xpj(fwd, I4::sub(x3, x7));

        let a04_p1_a26 = I4::add(a04, a26);
        let s04_mj_s26 = I4::sub(s04, js26);
        let a04_m1_a26 = I4::sub(a04, a26);
        let s04_pj_s26 = I4::add(s04, js26);
        let a15_p1_a37 = I4::add(a15, a37);
        let w8_s15_mj_s37 = I4::xw8(fwd, I4::sub(s15, js37));
        let j_a15_m1_a37 = I4::xpj(fwd, I4::sub(a15, a37));
        let v8_s15_pj_s37 = I4::xv8(fwd, I4::add(s15, js37));

        let w1p = I4::load(twid(8, big_n, 1, w, p));
        let w2p = I4::load(twid(8, big_n, 2, w, p));
        let w3p = I4::load(twid(8, big_n, 3, w, p));
        let w4p = I4::load(twid(8, big_n, 4, w, p));
        let w5p = I4::load(twid(8, big_n, 5, w, p));
        let w6p = I4::load(twid(8, big_n, 6, w, p));
        let w7p = I4::load(twid(8, big_n, 7, w, p));

        let a = I4::add(a04_p1_a26, a15_p1_a37);
        let b = I4::mul(w1p, I4::add(s04_mj_s26, w8_s15_mj_s37));
        let c = I4::mul(w2p, I4::sub(a04_m1_a26, j_a15_m1_a37));
        let d = I4::mul(w3p, I4::sub(s04_pj_s26, v8_s15_pj_s37));
        let e = I4::mul(w4p, I4::sub(a04_p1_a26, a15_p1_a37));
        let f = I4::mul(w5p, I4::sub(s04_mj_s26, w8_s15_mj_s37));
        let g = I4::mul(w6p, I4::add(a04_m1_a26, j_a15_m1_a37));
        let h = I4::mul(w7p, I4::add(s04_pj_s26, v8_s15_pj_s37));

        let (abcd0, abcd1, abcd2, abcd3) = I4::transpose(a, b, c, d);
        let (efgh0, efgh1, efgh2, efgh3) = I4::transpose(e, f, g, h);
        I4::store(y_8p.add(0), abcd0);
        I4::store(y_8p.add(4), efgh0);
        I4::store(y_8p.add(8), abcd1);
        I4::store(y_8p.add(12), efgh1);
        I4::store(y_8p.add(16), abcd2);
        I4::store(y_8p.add(20), efgh2);
        I4::store(y_8p.add(24), abcd3);
        I4::store(y_8p.add(28), efgh3);

        p += 4;
    }
}

#[inline(always)]
pub(crate) unsafe fn end_8<I: FftSimd64>(
    fwd: bool,
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    eo: bool,
) {
    debug_assert_eq!(n, 8);
    debug_assert_eq!(s % I::COMPLEX_PER_REG, 0);

    let z = if eo { y } else { x };

    let mut q = 0;
    while q < s {
        let xq = x.add(q);
        let zq = z.add(q);

        let x0 = I::load(xq.add(s * 0));
        let x1 = I::load(xq.add(s * 1));
        let x2 = I::load(xq.add(s * 2));
        let x3 = I::load(xq.add(s * 3));
        let x4 = I::load(xq.add(s * 4));
        let x5 = I::load(xq.add(s * 5));
        let x6 = I::load(xq.add(s * 6));
        let x7 = I::load(xq.add(s * 7));

        let a04 = I::add(x0, x4);
        let s04 = I::sub(x0, x4);
        let a26 = I::add(x2, x6);
        let js26 = I::xpj(fwd, I::sub(x2, x6));
        let a15 = I::add(x1, x5);
        let s15 = I::sub(x1, x5);
        let a37 = I::add(x3, x7);
        let js37 = I::xpj(fwd, I::sub(x3, x7));

        let a04_p1_a26 = I::add(a04, a26);
        let s04_mj_s26 = I::sub(s04, js26);
        let a04_m1_a26 = I::sub(a04, a26);
        let s04_pj_s26 = I::add(s04, js26);
        let a15_p1_a37 = I::add(a15, a37);
        let w8_s15_mj_s37 = I::xw8(fwd, I::sub(s15, js37));
        let j_a15_m1_a37 = I::xpj(fwd, I::sub(a15, a37));
        let v8_s15_pj_s37 = I::xv8(fwd, I::add(s15, js37));

        I::store(zq.add(0), I::add(a04_p1_a26, a15_p1_a37));
        I::store(zq.add(s), I::add(s04_mj_s26, w8_s15_mj_s37));
        I::store(zq.add(s * 2), I::sub(a04_m1_a26, j_a15_m1_a37));
        I::store(zq.add(s * 3), I::sub(s04_pj_s26, v8_s15_pj_s37));
        I::store(zq.add(s * 4), I::sub(a04_p1_a26, a15_p1_a37));
        I::store(zq.add(s * 5), I::sub(s04_mj_s26, w8_s15_mj_s37));
        I::store(zq.add(s * 6), I::add(a04_m1_a26, j_a15_m1_a37));
        I::store(zq.add(s * 7), I::add(s04_pj_s26, v8_s15_pj_s37));

        q += I::COMPLEX_PER_REG;
    }
}

macro_rules! dif8_impl {
    (
        $(
            $(#[$attr: meta])*
            pub static $fft: ident = Fft {
                core_1: $core1______: expr,
                native: $xn: ty,
                x1: $x1: ty,
                $(target: $target: tt,)?
            };
        )*
    ) => {
        $(
            #[allow(missing_copy_implementations)]
            #[allow(non_camel_case_types)]
            #[allow(dead_code)]
            $(#[$attr])*
            struct $fft {
                __private: (),
            }
            #[allow(unused_variables)]
            #[allow(dead_code)]
            $(#[$attr])*
            impl $fft {
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_00<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {}
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_01<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$x1>(FWD, 1 << 1, 1 << 0, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_02<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$x1>(FWD, 1 << 2, 1 << 0, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_03<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_8::<$x1>(FWD, 1 << 3, 1 << 0, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_04<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 4, 1 << 0, x, y, w);
                    end_2::<$xn>(FWD, 1 << 1, 1 << 3, y, x, true);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_05<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 5, 1 << 0, x, y, w);
                    end_4::<$xn>(FWD, 1 << 2, 1 << 3, y, x, true);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_06<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 6, 1 << 0, x, y, w);
                    end_8::<$xn>(FWD, 1 << 3, 1 << 3, y, x, true);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_07<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 7, 1 << 0, x, y, w);
                    core_::<$xn>(FWD, 1 << 4, 1 << 3, y, x, w);
                    end_2::<$xn>(FWD, 1 << 1, 1 << 6, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_08<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 8, 1 << 0, x, y, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 3, y, x, w);
                    end_4::<$xn>(FWD, 1 << 2, 1 << 6, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_09<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 9, 1 << 0, x, y, w);
                    core_::<$xn>(FWD, 1 << 6, 1 << 3, y, x, w);
                    end_8::<$xn>(FWD, 1 << 3, 1 << 6, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_10<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 10, 1 << 0, x, y, w);
                    core_::<$xn>(FWD, 1 << 07, 1 << 3, y, x, w);
                    core_::<$xn>(FWD, 1 << 04, 1 << 6, x, y, w);
                    end_2::<$xn>(FWD, 1 << 01, 1 << 9, y, x, true);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_11<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 11, 1 << 00, x, y, w);
                    core_::<$xn>(FWD, 1 << 08, 1 << 03, y, x, w);
                    core_::<$xn>(FWD, 1 << 05, 1 << 06, x, y, w);
                    end_4::<$xn>(FWD, 1 << 02, 1 << 09, y, x, true);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_12<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 12, 1 << 00, x, y, w);
                    core_::<$xn>(FWD, 1 << 09, 1 << 03, y, x, w);
                    core_::<$xn>(FWD, 1 << 06, 1 << 06, x, y, w);
                    end_8::<$xn>(FWD, 1 << 03, 1 << 09, y, x, true);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_13<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 13, 1 << 00, x, y, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 03, y, x, w);
                    core_::<$xn>(FWD, 1 << 07, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 04, 1 << 09, y, x, w);
                    end_2::<$xn>(FWD, 1 << 01, 1 << 12, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_14<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 14, 1 << 00, x, y, w);
                    core_::<$xn>(FWD, 1 << 11, 1 << 03, y, x, w);
                    core_::<$xn>(FWD, 1 << 08, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 05, 1 << 09, y, x, w);
                    end_4::<$xn>(FWD, 1 << 02, 1 << 12, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_15<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 15, 1 << 00, x, y, w);
                    core_::<$xn>(FWD, 1 << 12, 1 << 03, y, x, w);
                    core_::<$xn>(FWD, 1 << 09, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 06, 1 << 09, y, x, w);
                    end_8::<$xn>(FWD, 1 << 03, 1 << 12, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_16<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 16, 1 << 00, x, y, w);
                    core_::<$xn>(FWD, 1 << 13, 1 << 03, y, x, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 07, 1 << 09, y, x, w);
                    core_::<$xn>(FWD, 1 << 04, 1 << 12, x, y, w);
                    end_2::<$xn>(FWD, 1 << 01, 1 << 15, y, x, true);
                }
            }
            $(#[$attr])*
            pub(crate) static $fft: crate::FftImpl = crate::FftImpl {
                fwd: [
                    <$fft>::fft_00::<true>,
                    <$fft>::fft_01::<true>,
                    <$fft>::fft_02::<true>,
                    <$fft>::fft_03::<true>,
                    <$fft>::fft_04::<true>,
                    <$fft>::fft_05::<true>,
                    <$fft>::fft_06::<true>,
                    <$fft>::fft_07::<true>,
                    <$fft>::fft_08::<true>,
                    <$fft>::fft_09::<true>,
                    <$fft>::fft_10::<true>,
                    <$fft>::fft_11::<true>,
                    <$fft>::fft_12::<true>,
                    <$fft>::fft_13::<true>,
                    <$fft>::fft_14::<true>,
                    <$fft>::fft_15::<true>,
                    <$fft>::fft_16::<true>,
                ],
                inv: [
                    <$fft>::fft_00::<false>,
                    <$fft>::fft_01::<false>,
                    <$fft>::fft_02::<false>,
                    <$fft>::fft_03::<false>,
                    <$fft>::fft_04::<false>,
                    <$fft>::fft_05::<false>,
                    <$fft>::fft_06::<false>,
                    <$fft>::fft_07::<false>,
                    <$fft>::fft_08::<false>,
                    <$fft>::fft_09::<false>,
                    <$fft>::fft_10::<false>,
                    <$fft>::fft_11::<false>,
                    <$fft>::fft_12::<false>,
                    <$fft>::fft_13::<false>,
                    <$fft>::fft_14::<false>,
                    <$fft>::fft_15::<false>,
                    <$fft>::fft_16::<false>,
                ],
            };
            )*
    };
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::x86::*;

dif8_impl! {
    pub static DIF8_SCALAR = Fft {
        core_1: core_::<Scalar>,
        native: Scalar,
        x1: Scalar,
    };

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub static DIF8_AVX = Fft {
        core_1: core_x2::<AvxX2>,
        native: AvxX2,
        x1: AvxX1,
        target: "avx",
    };

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub static DIF8_FMA = Fft {
        core_1: core_x2::<FmaX2>,
        native: FmaX2,
        x1: FmaX1,
        target: "fma",
    };

    #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
    pub static DIF8_AVX512 = Fft {
        core_1: core_x4::<Avx512X2, Avx512X4>,
        native: Avx512X4,
        x1: Avx512X1,
        target: "avx512f",
    };
}

pub(crate) fn runtime_fft() -> crate::FftImpl {
    #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
    if x86_feature_detected!("avx512f") {
        return DIF8_AVX512;
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if x86_feature_detected!("fma") {
        return DIF8_FMA;
    } else if x86_feature_detected!("avx") {
        return DIF8_AVX;
    }

    DIF8_SCALAR
}
