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
use crate::dif8::end_8;
use crate::fft_simd::{twid, twid_t, FftSimd64, FftSimd64Ext, FftSimd64X2, FftSimd64X4, Scalar};
use crate::x86_feature_detected;

#[inline(always)]
unsafe fn core_<I: FftSimd64>(
    fwd: bool,
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    w: *const c64,
) {
    debug_assert_eq!(s % I::COMPLEX_PER_REG, 0);

    let m = n / 16;
    let big_n = n * s;
    let big_n0 = 0;
    let big_n1 = big_n / 16;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;
    let big_n4 = big_n1 * 4;
    let big_n5 = big_n1 * 5;
    let big_n6 = big_n1 * 6;
    let big_n7 = big_n1 * 7;
    let big_n8 = big_n1 * 8;
    let big_n9 = big_n1 * 9;
    let big_na = big_n1 * 10;
    let big_nb = big_n1 * 11;
    let big_nc = big_n1 * 12;
    let big_nd = big_n1 * 13;
    let big_ne = big_n1 * 14;
    let big_nf = big_n1 * 15;

    for p in 0..m {
        let sp = s * p;
        let s16p = 16 * sp;

        let w1p = I::splat(twid_t(16, big_n, 0x1, w, sp));
        let w2p = I::splat(twid_t(16, big_n, 0x2, w, sp));
        let w3p = I::splat(twid_t(16, big_n, 0x3, w, sp));
        let w4p = I::splat(twid_t(16, big_n, 0x4, w, sp));
        let w5p = I::splat(twid_t(16, big_n, 0x5, w, sp));
        let w6p = I::splat(twid_t(16, big_n, 0x6, w, sp));
        let w7p = I::splat(twid_t(16, big_n, 0x7, w, sp));
        let w8p = I::splat(twid_t(16, big_n, 0x8, w, sp));
        let w9p = I::splat(twid_t(16, big_n, 0x9, w, sp));
        let wap = I::splat(twid_t(16, big_n, 0xa, w, sp));
        let wbp = I::splat(twid_t(16, big_n, 0xb, w, sp));
        let wcp = I::splat(twid_t(16, big_n, 0xc, w, sp));
        let wdp = I::splat(twid_t(16, big_n, 0xd, w, sp));
        let wep = I::splat(twid_t(16, big_n, 0xe, w, sp));
        let wfp = I::splat(twid_t(16, big_n, 0xf, w, sp));

        let mut q = 0;
        while q < s {
            let xq_sp = x.add(q + sp);
            let yq_s16p = y.add(q + s16p);

            let x0 = I::load(xq_sp.add(big_n0));
            let x1 = I::load(xq_sp.add(big_n1));
            let x2 = I::load(xq_sp.add(big_n2));
            let x3 = I::load(xq_sp.add(big_n3));
            let x4 = I::load(xq_sp.add(big_n4));
            let x5 = I::load(xq_sp.add(big_n5));
            let x6 = I::load(xq_sp.add(big_n6));
            let x7 = I::load(xq_sp.add(big_n7));
            let x8 = I::load(xq_sp.add(big_n8));
            let x9 = I::load(xq_sp.add(big_n9));
            let xa = I::load(xq_sp.add(big_na));
            let xb = I::load(xq_sp.add(big_nb));
            let xc = I::load(xq_sp.add(big_nc));
            let xd = I::load(xq_sp.add(big_nd));
            let xe = I::load(xq_sp.add(big_ne));
            let xf = I::load(xq_sp.add(big_nf));

            let a08 = I::add(x0, x8);
            let s08 = I::sub(x0, x8);
            let a4c = I::add(x4, xc);
            let s4c = I::sub(x4, xc);
            let a2a = I::add(x2, xa);
            let s2a = I::sub(x2, xa);
            let a6e = I::add(x6, xe);
            let s6e = I::sub(x6, xe);
            let a19 = I::add(x1, x9);
            let s19 = I::sub(x1, x9);
            let a5d = I::add(x5, xd);
            let s5d = I::sub(x5, xd);
            let a3b = I::add(x3, xb);
            let s3b = I::sub(x3, xb);
            let a7f = I::add(x7, xf);
            let s7f = I::sub(x7, xf);

            let js4c = I::xpj(fwd, s4c);
            let js6e = I::xpj(fwd, s6e);
            let js5d = I::xpj(fwd, s5d);
            let js7f = I::xpj(fwd, s7f);

            let a08p1a4c = I::add(a08, a4c);
            let s08mjs4c = I::sub(s08, js4c);
            let a08m1a4c = I::sub(a08, a4c);
            let s08pjs4c = I::add(s08, js4c);
            let a2ap1a6e = I::add(a2a, a6e);
            let s2amjs6e = I::sub(s2a, js6e);
            let a2am1a6e = I::sub(a2a, a6e);
            let s2apjs6e = I::add(s2a, js6e);
            let a19p1a5d = I::add(a19, a5d);
            let s19mjs5d = I::sub(s19, js5d);
            let a19m1a5d = I::sub(a19, a5d);
            let s19pjs5d = I::add(s19, js5d);
            let a3bp1a7f = I::add(a3b, a7f);
            let s3bmjs7f = I::sub(s3b, js7f);
            let a3bm1a7f = I::sub(a3b, a7f);
            let s3bpjs7f = I::add(s3b, js7f);

            let w8_s2amjs6e = I::xw8(fwd, s2amjs6e);
            let j_a2am1a6e = I::xpj(fwd, a2am1a6e);
            let v8_s2apjs6e = I::xv8(fwd, s2apjs6e);

            let a08p1a4c_p1_a2ap1a6e = I::add(a08p1a4c, a2ap1a6e);
            let s08mjs4c_pw_s2amjs6e = I::add(s08mjs4c, w8_s2amjs6e);
            let a08m1a4c_mj_a2am1a6e = I::sub(a08m1a4c, j_a2am1a6e);
            let s08pjs4c_mv_s2apjs6e = I::sub(s08pjs4c, v8_s2apjs6e);
            let a08p1a4c_m1_a2ap1a6e = I::sub(a08p1a4c, a2ap1a6e);
            let s08mjs4c_mw_s2amjs6e = I::sub(s08mjs4c, w8_s2amjs6e);
            let a08m1a4c_pj_a2am1a6e = I::add(a08m1a4c, j_a2am1a6e);
            let s08pjs4c_pv_s2apjs6e = I::add(s08pjs4c, v8_s2apjs6e);

            let w8_s3bmjs7f = I::xw8(fwd, s3bmjs7f);
            let j_a3bm1a7f = I::xpj(fwd, a3bm1a7f);
            let v8_s3bpjs7f = I::xv8(fwd, s3bpjs7f);

            let a19p1a5d_p1_a3bp1a7f = I::add(a19p1a5d, a3bp1a7f);
            let s19mjs5d_pw_s3bmjs7f = I::add(s19mjs5d, w8_s3bmjs7f);
            let a19m1a5d_mj_a3bm1a7f = I::sub(a19m1a5d, j_a3bm1a7f);
            let s19pjs5d_mv_s3bpjs7f = I::sub(s19pjs5d, v8_s3bpjs7f);
            let a19p1a5d_m1_a3bp1a7f = I::sub(a19p1a5d, a3bp1a7f);
            let s19mjs5d_mw_s3bmjs7f = I::sub(s19mjs5d, w8_s3bmjs7f);
            let a19m1a5d_pj_a3bm1a7f = I::add(a19m1a5d, j_a3bm1a7f);
            let s19pjs5d_pv_s3bpjs7f = I::add(s19pjs5d, v8_s3bpjs7f);

            let h1_s19mjs5d_pw_s3bmjs7f = I::xh1(fwd, s19mjs5d_pw_s3bmjs7f);
            let w8_a19m1a5d_mj_a3bm1a7f = I::xw8(fwd, a19m1a5d_mj_a3bm1a7f);
            let h3_s19pjs5d_mv_s3bpjs7f = I::xh3(fwd, s19pjs5d_mv_s3bpjs7f);
            let j_a19p1a5d_m1_a3bp1a7f = I::xpj(fwd, a19p1a5d_m1_a3bp1a7f);
            let hd_s19mjs5d_mw_s3bmjs7f = I::xhd(fwd, s19mjs5d_mw_s3bmjs7f);
            let v8_a19m1a5d_pj_a3bm1a7f = I::xv8(fwd, a19m1a5d_pj_a3bm1a7f);
            let hf_s19pjs5d_pv_s3bpjs7f = I::xhf(fwd, s19pjs5d_pv_s3bpjs7f);

            I::store(
                yq_s16p.add(0),
                I::add(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
            );
            I::store(
                yq_s16p.add(s),
                I::mul(w1p, I::add(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)),
            );
            I::store(
                yq_s16p.add(s * 0x2),
                I::mul(w2p, I::add(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)),
            );
            I::store(
                yq_s16p.add(s * 0x3),
                I::mul(w3p, I::add(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)),
            );
            I::store(
                yq_s16p.add(s * 0x4),
                I::mul(w4p, I::sub(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f)),
            );
            I::store(
                yq_s16p.add(s * 0x5),
                I::mul(w5p, I::sub(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)),
            );
            I::store(
                yq_s16p.add(s * 0x6),
                I::mul(w6p, I::sub(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)),
            );
            I::store(
                yq_s16p.add(s * 0x7),
                I::mul(w7p, I::sub(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)),
            );

            I::store(
                yq_s16p.add(s * 0x8),
                I::mul(w8p, I::sub(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f)),
            );
            I::store(
                yq_s16p.add(s * 0x9),
                I::mul(w9p, I::sub(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)),
            );
            I::store(
                yq_s16p.add(s * 0xa),
                I::mul(wap, I::sub(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)),
            );
            I::store(
                yq_s16p.add(s * 0xb),
                I::mul(wbp, I::sub(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)),
            );
            I::store(
                yq_s16p.add(s * 0xc),
                I::mul(wcp, I::add(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f)),
            );
            I::store(
                yq_s16p.add(s * 0xd),
                I::mul(wdp, I::add(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)),
            );
            I::store(
                yq_s16p.add(s * 0xe),
                I::mul(wep, I::add(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)),
            );
            I::store(
                yq_s16p.add(s * 0xf),
                I::mul(wfp, I::add(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)),
            );

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
    let big_n1 = big_n / 16;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;
    let big_n4 = big_n1 * 4;
    let big_n5 = big_n1 * 5;
    let big_n6 = big_n1 * 6;
    let big_n7 = big_n1 * 7;
    let big_n8 = big_n1 * 8;
    let big_n9 = big_n1 * 9;
    let big_na = big_n1 * 10;
    let big_nb = big_n1 * 11;
    let big_nc = big_n1 * 12;
    let big_nd = big_n1 * 13;
    let big_ne = big_n1 * 14;
    let big_nf = big_n1 * 15;

    let mut p = 0;
    while p < big_n1 {
        let x_p = x.add(p);
        let y_16p = y.add(16 * p);

        let x0 = I::load(x_p.add(big_n0));
        let x1 = I::load(x_p.add(big_n1));
        let x2 = I::load(x_p.add(big_n2));
        let x3 = I::load(x_p.add(big_n3));
        let x4 = I::load(x_p.add(big_n4));
        let x5 = I::load(x_p.add(big_n5));
        let x6 = I::load(x_p.add(big_n6));
        let x7 = I::load(x_p.add(big_n7));
        let x8 = I::load(x_p.add(big_n8));
        let x9 = I::load(x_p.add(big_n9));
        let xa = I::load(x_p.add(big_na));
        let xb = I::load(x_p.add(big_nb));
        let xc = I::load(x_p.add(big_nc));
        let xd = I::load(x_p.add(big_nd));
        let xe = I::load(x_p.add(big_ne));
        let xf = I::load(x_p.add(big_nf));

        let a08 = I::add(x0, x8);
        let s08 = I::sub(x0, x8);
        let a4c = I::add(x4, xc);
        let s4c = I::sub(x4, xc);
        let a2a = I::add(x2, xa);
        let s2a = I::sub(x2, xa);
        let a6e = I::add(x6, xe);
        let s6e = I::sub(x6, xe);
        let a19 = I::add(x1, x9);
        let s19 = I::sub(x1, x9);
        let a5d = I::add(x5, xd);
        let s5d = I::sub(x5, xd);
        let a3b = I::add(x3, xb);
        let s3b = I::sub(x3, xb);
        let a7f = I::add(x7, xf);
        let s7f = I::sub(x7, xf);

        let js4c = I::xpj(fwd, s4c);
        let js6e = I::xpj(fwd, s6e);
        let js5d = I::xpj(fwd, s5d);
        let js7f = I::xpj(fwd, s7f);

        let a08p1a4c = I::add(a08, a4c);
        let s08mjs4c = I::sub(s08, js4c);
        let a08m1a4c = I::sub(a08, a4c);
        let s08pjs4c = I::add(s08, js4c);
        let a2ap1a6e = I::add(a2a, a6e);
        let s2amjs6e = I::sub(s2a, js6e);
        let a2am1a6e = I::sub(a2a, a6e);
        let s2apjs6e = I::add(s2a, js6e);
        let a19p1a5d = I::add(a19, a5d);
        let s19mjs5d = I::sub(s19, js5d);
        let a19m1a5d = I::sub(a19, a5d);
        let s19pjs5d = I::add(s19, js5d);
        let a3bp1a7f = I::add(a3b, a7f);
        let s3bmjs7f = I::sub(s3b, js7f);
        let a3bm1a7f = I::sub(a3b, a7f);
        let s3bpjs7f = I::add(s3b, js7f);

        let w8_s2amjs6e = I::xw8(fwd, s2amjs6e);
        let j_a2am1a6e = I::xpj(fwd, a2am1a6e);
        let v8_s2apjs6e = I::xv8(fwd, s2apjs6e);

        let a08p1a4c_p1_a2ap1a6e = I::add(a08p1a4c, a2ap1a6e);
        let s08mjs4c_pw_s2amjs6e = I::add(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_mj_a2am1a6e = I::sub(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_mv_s2apjs6e = I::sub(s08pjs4c, v8_s2apjs6e);
        let a08p1a4c_m1_a2ap1a6e = I::sub(a08p1a4c, a2ap1a6e);
        let s08mjs4c_mw_s2amjs6e = I::sub(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_pj_a2am1a6e = I::add(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_pv_s2apjs6e = I::add(s08pjs4c, v8_s2apjs6e);

        let w8_s3bmjs7f = I::xw8(fwd, s3bmjs7f);
        let j_a3bm1a7f = I::xpj(fwd, a3bm1a7f);
        let v8_s3bpjs7f = I::xv8(fwd, s3bpjs7f);

        let a19p1a5d_p1_a3bp1a7f = I::add(a19p1a5d, a3bp1a7f);
        let s19mjs5d_pw_s3bmjs7f = I::add(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_mj_a3bm1a7f = I::sub(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_mv_s3bpjs7f = I::sub(s19pjs5d, v8_s3bpjs7f);
        let a19p1a5d_m1_a3bp1a7f = I::sub(a19p1a5d, a3bp1a7f);
        let s19mjs5d_mw_s3bmjs7f = I::sub(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_pj_a3bm1a7f = I::add(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_pv_s3bpjs7f = I::add(s19pjs5d, v8_s3bpjs7f);

        let h1_s19mjs5d_pw_s3bmjs7f = I::xh1(fwd, s19mjs5d_pw_s3bmjs7f);
        let w8_a19m1a5d_mj_a3bm1a7f = I::xw8(fwd, a19m1a5d_mj_a3bm1a7f);
        let h3_s19pjs5d_mv_s3bpjs7f = I::xh3(fwd, s19pjs5d_mv_s3bpjs7f);
        let j_a19p1a5d_m1_a3bp1a7f = I::xpj(fwd, a19p1a5d_m1_a3bp1a7f);
        let hd_s19mjs5d_mw_s3bmjs7f = I::xhd(fwd, s19mjs5d_mw_s3bmjs7f);
        let v8_a19m1a5d_pj_a3bm1a7f = I::xv8(fwd, a19m1a5d_pj_a3bm1a7f);
        let hf_s19pjs5d_pv_s3bpjs7f = I::xhf(fwd, s19pjs5d_pv_s3bpjs7f);

        let w1p = I::load(twid(16, big_n, 1, w, p));
        let w2p = I::load(twid(16, big_n, 2, w, p));
        let w3p = I::load(twid(16, big_n, 3, w, p));
        let w4p = I::load(twid(16, big_n, 4, w, p));
        let w5p = I::load(twid(16, big_n, 5, w, p));
        let w6p = I::load(twid(16, big_n, 6, w, p));
        let w7p = I::load(twid(16, big_n, 7, w, p));
        let w8p = I::load(twid(16, big_n, 8, w, p));
        let w9p = I::load(twid(16, big_n, 9, w, p));
        let wap = I::load(twid(16, big_n, 10, w, p));
        let wbp = I::load(twid(16, big_n, 11, w, p));
        let wcp = I::load(twid(16, big_n, 12, w, p));
        let wdp = I::load(twid(16, big_n, 13, w, p));
        let wep = I::load(twid(16, big_n, 14, w, p));
        let wfp = I::load(twid(16, big_n, 15, w, p));

        let aa = I::add(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f);
        let bb = I::mul(w1p, I::add(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        let cc = I::mul(w2p, I::add(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
        let dd = I::mul(w3p, I::add(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
        let ee = I::mul(w4p, I::sub(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f));
        let ff = I::mul(w5p, I::sub(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
        let gg = I::mul(w6p, I::sub(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
        let hh = I::mul(w7p, I::sub(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

        let ii = I::mul(w8p, I::sub(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f));
        let jj = I::mul(w9p, I::sub(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        let kk = I::mul(wap, I::sub(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
        let ll = I::mul(wbp, I::sub(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
        let mm = I::mul(wcp, I::add(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f));
        let nn = I::mul(wdp, I::add(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
        let oo = I::mul(wep, I::add(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
        let pp = I::mul(wfp, I::add(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

        {
            let ab = I::catlo(aa, bb);
            I::store(y_16p.add(0x00), ab);
            let cd = I::catlo(cc, dd);
            I::store(y_16p.add(0x02), cd);
            let ef = I::catlo(ee, ff);
            I::store(y_16p.add(0x04), ef);
            let gh = I::catlo(gg, hh);
            I::store(y_16p.add(0x06), gh);
            let ij = I::catlo(ii, jj);
            I::store(y_16p.add(0x08), ij);
            let kl = I::catlo(kk, ll);
            I::store(y_16p.add(0x0a), kl);
            let mn = I::catlo(mm, nn);
            I::store(y_16p.add(0x0c), mn);
            let op = I::catlo(oo, pp);
            I::store(y_16p.add(0x0e), op);
        }
        {
            let ab = I::cathi(aa, bb);
            I::store(y_16p.add(0x10), ab);
            let cd = I::cathi(cc, dd);
            I::store(y_16p.add(0x12), cd);
            let ef = I::cathi(ee, ff);
            I::store(y_16p.add(0x14), ef);
            let gh = I::cathi(gg, hh);
            I::store(y_16p.add(0x16), gh);
            let ij = I::cathi(ii, jj);
            I::store(y_16p.add(0x18), ij);
            let kl = I::cathi(kk, ll);
            I::store(y_16p.add(0x1a), kl);
            let mn = I::cathi(mm, nn);
            I::store(y_16p.add(0x1c), mn);
            let op = I::cathi(oo, pp);
            I::store(y_16p.add(0x1e), op);
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
    if n == 32 {
        return core_x2::<I2>(fwd, n, s, x, y, w);
    }

    let big_n = n;
    let big_n0 = 0;
    let big_n1 = big_n / 16;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;
    let big_n4 = big_n1 * 4;
    let big_n5 = big_n1 * 5;
    let big_n6 = big_n1 * 6;
    let big_n7 = big_n1 * 7;
    let big_n8 = big_n1 * 8;
    let big_n9 = big_n1 * 9;
    let big_na = big_n1 * 10;
    let big_nb = big_n1 * 11;
    let big_nc = big_n1 * 12;
    let big_nd = big_n1 * 13;
    let big_ne = big_n1 * 14;
    let big_nf = big_n1 * 15;

    debug_assert_eq!(big_n1 % 4, 0);
    let mut p = 0;
    while p < big_n1 {
        let x_p = x.add(p);
        let y_16p = y.add(16 * p);

        let x0 = I4::load(x_p.add(big_n0));
        let x1 = I4::load(x_p.add(big_n1));
        let x2 = I4::load(x_p.add(big_n2));
        let x3 = I4::load(x_p.add(big_n3));
        let x4 = I4::load(x_p.add(big_n4));
        let x5 = I4::load(x_p.add(big_n5));
        let x6 = I4::load(x_p.add(big_n6));
        let x7 = I4::load(x_p.add(big_n7));
        let x8 = I4::load(x_p.add(big_n8));
        let x9 = I4::load(x_p.add(big_n9));
        let xa = I4::load(x_p.add(big_na));
        let xb = I4::load(x_p.add(big_nb));
        let xc = I4::load(x_p.add(big_nc));
        let xd = I4::load(x_p.add(big_nd));
        let xe = I4::load(x_p.add(big_ne));
        let xf = I4::load(x_p.add(big_nf));

        let a08 = I4::add(x0, x8);
        let s08 = I4::sub(x0, x8);
        let a4c = I4::add(x4, xc);
        let s4c = I4::sub(x4, xc);
        let a2a = I4::add(x2, xa);
        let s2a = I4::sub(x2, xa);
        let a6e = I4::add(x6, xe);
        let s6e = I4::sub(x6, xe);
        let a19 = I4::add(x1, x9);
        let s19 = I4::sub(x1, x9);
        let a5d = I4::add(x5, xd);
        let s5d = I4::sub(x5, xd);
        let a3b = I4::add(x3, xb);
        let s3b = I4::sub(x3, xb);
        let a7f = I4::add(x7, xf);
        let s7f = I4::sub(x7, xf);

        let js4c = I4::xpj(fwd, s4c);
        let js6e = I4::xpj(fwd, s6e);
        let js5d = I4::xpj(fwd, s5d);
        let js7f = I4::xpj(fwd, s7f);

        let a08p1a4c = I4::add(a08, a4c);
        let s08mjs4c = I4::sub(s08, js4c);
        let a08m1a4c = I4::sub(a08, a4c);
        let s08pjs4c = I4::add(s08, js4c);
        let a2ap1a6e = I4::add(a2a, a6e);
        let s2amjs6e = I4::sub(s2a, js6e);
        let a2am1a6e = I4::sub(a2a, a6e);
        let s2apjs6e = I4::add(s2a, js6e);
        let a19p1a5d = I4::add(a19, a5d);
        let s19mjs5d = I4::sub(s19, js5d);
        let a19m1a5d = I4::sub(a19, a5d);
        let s19pjs5d = I4::add(s19, js5d);
        let a3bp1a7f = I4::add(a3b, a7f);
        let s3bmjs7f = I4::sub(s3b, js7f);
        let a3bm1a7f = I4::sub(a3b, a7f);
        let s3bpjs7f = I4::add(s3b, js7f);

        let w8_s2amjs6e = I4::xw8(fwd, s2amjs6e);
        let j_a2am1a6e = I4::xpj(fwd, a2am1a6e);
        let v8_s2apjs6e = I4::xv8(fwd, s2apjs6e);

        let a08p1a4c_p1_a2ap1a6e = I4::add(a08p1a4c, a2ap1a6e);
        let s08mjs4c_pw_s2amjs6e = I4::add(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_mj_a2am1a6e = I4::sub(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_mv_s2apjs6e = I4::sub(s08pjs4c, v8_s2apjs6e);
        let a08p1a4c_m1_a2ap1a6e = I4::sub(a08p1a4c, a2ap1a6e);
        let s08mjs4c_mw_s2amjs6e = I4::sub(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_pj_a2am1a6e = I4::add(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_pv_s2apjs6e = I4::add(s08pjs4c, v8_s2apjs6e);

        let w8_s3bmjs7f = I4::xw8(fwd, s3bmjs7f);
        let j_a3bm1a7f = I4::xpj(fwd, a3bm1a7f);
        let v8_s3bpjs7f = I4::xv8(fwd, s3bpjs7f);

        let a19p1a5d_p1_a3bp1a7f = I4::add(a19p1a5d, a3bp1a7f);
        let s19mjs5d_pw_s3bmjs7f = I4::add(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_mj_a3bm1a7f = I4::sub(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_mv_s3bpjs7f = I4::sub(s19pjs5d, v8_s3bpjs7f);
        let a19p1a5d_m1_a3bp1a7f = I4::sub(a19p1a5d, a3bp1a7f);
        let s19mjs5d_mw_s3bmjs7f = I4::sub(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_pj_a3bm1a7f = I4::add(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_pv_s3bpjs7f = I4::add(s19pjs5d, v8_s3bpjs7f);

        let h1_s19mjs5d_pw_s3bmjs7f = I4::xh1(fwd, s19mjs5d_pw_s3bmjs7f);
        let w8_a19m1a5d_mj_a3bm1a7f = I4::xw8(fwd, a19m1a5d_mj_a3bm1a7f);
        let h3_s19pjs5d_mv_s3bpjs7f = I4::xh3(fwd, s19pjs5d_mv_s3bpjs7f);
        let j_a19p1a5d_m1_a3bp1a7f = I4::xpj(fwd, a19p1a5d_m1_a3bp1a7f);
        let hd_s19mjs5d_mw_s3bmjs7f = I4::xhd(fwd, s19mjs5d_mw_s3bmjs7f);
        let v8_a19m1a5d_pj_a3bm1a7f = I4::xv8(fwd, a19m1a5d_pj_a3bm1a7f);
        let hf_s19pjs5d_pv_s3bpjs7f = I4::xhf(fwd, s19pjs5d_pv_s3bpjs7f);

        let w1p = I4::load(twid(16, big_n, 1, w, p));
        let w2p = I4::load(twid(16, big_n, 2, w, p));
        let w3p = I4::load(twid(16, big_n, 3, w, p));
        let w4p = I4::load(twid(16, big_n, 4, w, p));
        let w5p = I4::load(twid(16, big_n, 5, w, p));
        let w6p = I4::load(twid(16, big_n, 6, w, p));
        let w7p = I4::load(twid(16, big_n, 7, w, p));
        let w8p = I4::load(twid(16, big_n, 8, w, p));
        let w9p = I4::load(twid(16, big_n, 9, w, p));
        let wap = I4::load(twid(16, big_n, 10, w, p));
        let wbp = I4::load(twid(16, big_n, 11, w, p));
        let wcp = I4::load(twid(16, big_n, 12, w, p));
        let wdp = I4::load(twid(16, big_n, 13, w, p));
        let wep = I4::load(twid(16, big_n, 14, w, p));
        let wfp = I4::load(twid(16, big_n, 15, w, p));

        let a_ = I4::add(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f);
        let b_ = I4::mul(w1p, I4::add(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        let c_ = I4::mul(w2p, I4::add(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
        let d_ = I4::mul(w3p, I4::add(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
        let e_ = I4::mul(w4p, I4::sub(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f));
        let f_ = I4::mul(w5p, I4::sub(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
        let g_ = I4::mul(w6p, I4::sub(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
        let h_ = I4::mul(w7p, I4::sub(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

        let i_ = I4::mul(w8p, I4::sub(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f));
        let j_ = I4::mul(w9p, I4::sub(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        let k_ = I4::mul(wap, I4::sub(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
        let l_ = I4::mul(wbp, I4::sub(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
        let m_ = I4::mul(wcp, I4::add(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f));
        let n_ = I4::mul(wdp, I4::add(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
        let o_ = I4::mul(wep, I4::add(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
        let p_ = I4::mul(wfp, I4::add(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

        let (abcd0, abcd1, abcd2, abcd3) = I4::transpose(a_, b_, c_, d_);
        let (efgh0, efgh1, efgh2, efgh3) = I4::transpose(e_, f_, g_, h_);
        let (ijkl0, ijkl1, ijkl2, ijkl3) = I4::transpose(i_, j_, k_, l_);
        let (mnop0, mnop1, mnop2, mnop3) = I4::transpose(m_, n_, o_, p_);

        I4::store(y_16p.add(0x00), abcd0);
        I4::store(y_16p.add(0x04), efgh0);
        I4::store(y_16p.add(0x08), ijkl0);
        I4::store(y_16p.add(0x0c), mnop0);

        I4::store(y_16p.add(0x10), abcd1);
        I4::store(y_16p.add(0x14), efgh1);
        I4::store(y_16p.add(0x18), ijkl1);
        I4::store(y_16p.add(0x1c), mnop1);

        I4::store(y_16p.add(0x20), abcd2);
        I4::store(y_16p.add(0x24), efgh2);
        I4::store(y_16p.add(0x28), ijkl2);
        I4::store(y_16p.add(0x2c), mnop2);

        I4::store(y_16p.add(0x30), abcd3);
        I4::store(y_16p.add(0x34), efgh3);
        I4::store(y_16p.add(0x38), ijkl3);
        I4::store(y_16p.add(0x3c), mnop3);

        p += 4;
    }
}

#[inline(always)]
pub(crate) unsafe fn end16<I: FftSimd64>(
    fwd: bool,
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    eo: bool,
) {
    debug_assert_eq!(n, 16);
    debug_assert_eq!(s % I::COMPLEX_PER_REG, 0);

    let z = if eo { y } else { x };

    let mut q = 0;
    while q < s {
        let xq = x.add(q);
        let zq = z.add(q);

        let x0 = I::load(xq.add(s * 0x0));
        let x1 = I::load(xq.add(s * 0x1));
        let x2 = I::load(xq.add(s * 0x2));
        let x3 = I::load(xq.add(s * 0x3));
        let x4 = I::load(xq.add(s * 0x4));
        let x5 = I::load(xq.add(s * 0x5));
        let x6 = I::load(xq.add(s * 0x6));
        let x7 = I::load(xq.add(s * 0x7));
        let x8 = I::load(xq.add(s * 0x8));
        let x9 = I::load(xq.add(s * 0x9));
        let xa = I::load(xq.add(s * 0xa));
        let xb = I::load(xq.add(s * 0xb));
        let xc = I::load(xq.add(s * 0xc));
        let xd = I::load(xq.add(s * 0xd));
        let xe = I::load(xq.add(s * 0xe));
        let xf = I::load(xq.add(s * 0xf));

        let a08 = I::add(x0, x8);
        let s08 = I::sub(x0, x8);
        let a4c = I::add(x4, xc);
        let s4c = I::sub(x4, xc);
        let a2a = I::add(x2, xa);
        let s2a = I::sub(x2, xa);
        let a6e = I::add(x6, xe);
        let s6e = I::sub(x6, xe);
        let a19 = I::add(x1, x9);
        let s19 = I::sub(x1, x9);
        let a5d = I::add(x5, xd);
        let s5d = I::sub(x5, xd);
        let a3b = I::add(x3, xb);
        let s3b = I::sub(x3, xb);
        let a7f = I::add(x7, xf);
        let s7f = I::sub(x7, xf);

        let js4c = I::xpj(fwd, s4c);
        let js6e = I::xpj(fwd, s6e);
        let js5d = I::xpj(fwd, s5d);
        let js7f = I::xpj(fwd, s7f);

        let a08p1a4c = I::add(a08, a4c);
        let s08mjs4c = I::sub(s08, js4c);
        let a08m1a4c = I::sub(a08, a4c);
        let s08pjs4c = I::add(s08, js4c);
        let a2ap1a6e = I::add(a2a, a6e);
        let s2amjs6e = I::sub(s2a, js6e);
        let a2am1a6e = I::sub(a2a, a6e);
        let s2apjs6e = I::add(s2a, js6e);
        let a19p1a5d = I::add(a19, a5d);
        let s19mjs5d = I::sub(s19, js5d);
        let a19m1a5d = I::sub(a19, a5d);
        let s19pjs5d = I::add(s19, js5d);
        let a3bp1a7f = I::add(a3b, a7f);
        let s3bmjs7f = I::sub(s3b, js7f);
        let a3bm1a7f = I::sub(a3b, a7f);
        let s3bpjs7f = I::add(s3b, js7f);

        let w8_s2amjs6e = I::xw8(fwd, s2amjs6e);
        let j_a2am1a6e = I::xpj(fwd, a2am1a6e);
        let v8_s2apjs6e = I::xv8(fwd, s2apjs6e);

        let a08p1a4c_p1_a2ap1a6e = I::add(a08p1a4c, a2ap1a6e);
        let s08mjs4c_pw_s2amjs6e = I::add(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_mj_a2am1a6e = I::sub(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_mv_s2apjs6e = I::sub(s08pjs4c, v8_s2apjs6e);
        let a08p1a4c_m1_a2ap1a6e = I::sub(a08p1a4c, a2ap1a6e);
        let s08mjs4c_mw_s2amjs6e = I::sub(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_pj_a2am1a6e = I::add(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_pv_s2apjs6e = I::add(s08pjs4c, v8_s2apjs6e);

        let w8_s3bmjs7f = I::xw8(fwd, s3bmjs7f);
        let j_a3bm1a7f = I::xpj(fwd, a3bm1a7f);
        let v8_s3bpjs7f = I::xv8(fwd, s3bpjs7f);

        let a19p1a5d_p1_a3bp1a7f = I::add(a19p1a5d, a3bp1a7f);
        let s19mjs5d_pw_s3bmjs7f = I::add(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_mj_a3bm1a7f = I::sub(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_mv_s3bpjs7f = I::sub(s19pjs5d, v8_s3bpjs7f);
        let a19p1a5d_m1_a3bp1a7f = I::sub(a19p1a5d, a3bp1a7f);
        let s19mjs5d_mw_s3bmjs7f = I::sub(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_pj_a3bm1a7f = I::add(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_pv_s3bpjs7f = I::add(s19pjs5d, v8_s3bpjs7f);

        let h1_s19mjs5d_pw_s3bmjs7f = I::xh1(fwd, s19mjs5d_pw_s3bmjs7f);
        let w8_a19m1a5d_mj_a3bm1a7f = I::xw8(fwd, a19m1a5d_mj_a3bm1a7f);
        let h3_s19pjs5d_mv_s3bpjs7f = I::xh3(fwd, s19pjs5d_mv_s3bpjs7f);
        let j_a19p1a5d_m1_a3bp1a7f = I::xpj(fwd, a19p1a5d_m1_a3bp1a7f);
        let hd_s19mjs5d_mw_s3bmjs7f = I::xhd(fwd, s19mjs5d_mw_s3bmjs7f);
        let v8_a19m1a5d_pj_a3bm1a7f = I::xv8(fwd, a19m1a5d_pj_a3bm1a7f);
        let hf_s19pjs5d_pv_s3bpjs7f = I::xhf(fwd, s19pjs5d_pv_s3bpjs7f);

        I::store(
            zq.add(0),
            I::add(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );
        I::store(
            zq.add(s),
            I::add(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );
        I::store(
            zq.add(s * 0x2),
            I::add(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );
        I::store(
            zq.add(s * 0x3),
            I::add(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );
        I::store(
            zq.add(s * 0x4),
            I::sub(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );
        I::store(
            zq.add(s * 0x5),
            I::sub(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );
        I::store(
            zq.add(s * 0x6),
            I::sub(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );
        I::store(
            zq.add(s * 0x7),
            I::sub(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );

        I::store(
            zq.add(s * 0x8),
            I::sub(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );
        I::store(
            zq.add(s * 0x9),
            I::sub(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );
        I::store(
            zq.add(s * 0xa),
            I::sub(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );
        I::store(
            zq.add(s * 0xb),
            I::sub(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );
        I::store(
            zq.add(s * 0xc),
            I::add(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );
        I::store(
            zq.add(s * 0xd),
            I::add(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );
        I::store(
            zq.add(s * 0xe),
            I::add(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );
        I::store(
            zq.add(s * 0xf),
            I::add(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );

        q += I::COMPLEX_PER_REG;
    }
}

macro_rules! dif16_impl {
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
                    end16::<$x1>(FWD, 1 << 4, 1 << 0, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_05<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 5, 1 << 0, x, y, w);
                    end_2::<$xn>(FWD, 1 << 1, 1 << 4, y, x, true);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_06<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 6, 1 << 0, x, y, w);
                    end_4::<$xn>(FWD, 1 << 2, 1 << 4, y, x, true);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_07<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 7, 1 << 0, x, y, w);
                    end_8::<$xn>(FWD, 1 << 3, 1 << 4, y, x, true);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_08<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 8, 1 << 0, x, y, w);
                    end16::<$xn>(FWD, 1 << 4, 1 << 4, y, x, true);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_09<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 9, 1 << 0, x, y, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 4, y, x, w);
                    end_2::<$xn>(FWD, 1 << 1, 1 << 8, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_10<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 10, 1 << 0, x, y, w);
                    core_::<$xn>(FWD, 1 << 06, 1 << 4, y, x, w);
                    end_4::<$xn>(FWD, 1 << 02, 1 << 8, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_11<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 11, 1 << 00, x, y, w);
                    core_::<$xn>(FWD, 1 << 07, 1 << 04, y, x, w);
                    end_8::<$xn>(FWD, 1 << 03, 1 << 08, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_12<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 12, 1 << 00, x, y, w);
                    core_::<$xn>(FWD, 1 << 08, 1 << 04, y, x, w);
                    end16::<$xn>(FWD, 1 << 04, 1 << 08, x, y, false);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_13<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 13, 1 << 00, x, y, w);
                    core_::<$xn>(FWD, 1 << 09, 1 << 04, y, x, w);
                    core_::<$xn>(FWD, 1 << 05, 1 << 08, x, y, w);
                    end_2::<$xn>(FWD, 1 << 01, 1 << 12, y, x, true);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_14<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 14, 1 << 00, x, y, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 04, y, x, w);
                    core_::<$xn>(FWD, 1 << 06, 1 << 08, x, y, w);
                    end_4::<$xn>(FWD, 1 << 02, 1 << 12, y, x, true);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_15<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 15, 1 << 00, x, y, w);
                    core_::<$xn>(FWD, 1 << 11, 1 << 04, y, x, w);
                    core_::<$xn>(FWD, 1 << 07, 1 << 08, x, y, w);
                    end_8::<$xn>(FWD, 1 << 03, 1 << 12, y, x, true);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_16<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    $core1______(FWD, 1 << 16, 1 << 00, x, y, w);
                    core_::<$xn>(FWD, 1 << 12, 1 << 04, y, x, w);
                    core_::<$xn>(FWD, 1 << 08, 1 << 08, x, y, w);
                    end16::<$xn>(FWD, 1 << 04, 1 << 12, y, x, true);
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

dif16_impl! {
    pub static DIF16_SCALAR = Fft {
        core_1: core_::<Scalar>,
        native: Scalar,
        x1: Scalar,
    };

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub static DIF16_AVX = Fft {
        core_1: core_x2::<AvxX2>,
        native: AvxX2,
        x1: AvxX1,
        target: "avx",
    };

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub static DIF16_FMA = Fft {
        core_1: core_x2::<FmaX2>,
        native: FmaX2,
        x1: FmaX1,
        target: "fma",
    };

    #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
    pub static DIF16_AVX512 = Fft {
        core_1: core_x4::<Avx512X2, Avx512X4>,
        native: Avx512X4,
        x1: Avx512X1,
        target: "avx512f",
    };
}

pub(crate) fn runtime_fft() -> crate::FftImpl {
    #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
    if x86_feature_detected!("avx512f") {
        return DIF16_AVX512;
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if x86_feature_detected!("fma") {
        return DIF16_FMA;
    } else if x86_feature_detected!("avx") {
        return DIF16_AVX;
    }

    DIF16_SCALAR
}
