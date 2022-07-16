use crate::c64;
use crate::dit4::{end_2, end_4};
use crate::dit8::end_8;
use crate::fft_simd::{twid, twid_t, FftSimd64, FftSimd64Ext, FftSimd64X2, Scalar};

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

            let y0 = I::load(yq_s16p.add(0));
            let y1 = I::mul(w1p, I::load(yq_s16p.add(s)));
            let y2 = I::mul(w2p, I::load(yq_s16p.add(s * 0x2)));
            let y3 = I::mul(w3p, I::load(yq_s16p.add(s * 0x3)));
            let y4 = I::mul(w4p, I::load(yq_s16p.add(s * 0x4)));
            let y5 = I::mul(w5p, I::load(yq_s16p.add(s * 0x5)));
            let y6 = I::mul(w6p, I::load(yq_s16p.add(s * 0x6)));
            let y7 = I::mul(w7p, I::load(yq_s16p.add(s * 0x7)));
            let y8 = I::mul(w8p, I::load(yq_s16p.add(s * 0x8)));
            let y9 = I::mul(w9p, I::load(yq_s16p.add(s * 0x9)));
            let ya = I::mul(wap, I::load(yq_s16p.add(s * 0xa)));
            let yb = I::mul(wbp, I::load(yq_s16p.add(s * 0xb)));
            let yc = I::mul(wcp, I::load(yq_s16p.add(s * 0xc)));
            let yd = I::mul(wdp, I::load(yq_s16p.add(s * 0xd)));
            let ye = I::mul(wep, I::load(yq_s16p.add(s * 0xe)));
            let yf = I::mul(wfp, I::load(yq_s16p.add(s * 0xf)));

            let a08 = I::add(y0, y8);
            let s08 = I::sub(y0, y8);
            let a4c = I::add(y4, yc);
            let s4c = I::sub(y4, yc);
            let a2a = I::add(y2, ya);
            let s2a = I::sub(y2, ya);
            let a6e = I::add(y6, ye);
            let s6e = I::sub(y6, ye);
            let a19 = I::add(y1, y9);
            let s19 = I::sub(y1, y9);
            let a5d = I::add(y5, yd);
            let s5d = I::sub(y5, yd);
            let a3b = I::add(y3, yb);
            let s3b = I::sub(y3, yb);
            let a7f = I::add(y7, yf);
            let s7f = I::sub(y7, yf);

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

            I::store(
                xq_sp.add(big_n0),
                I::add(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
            );
            I::store(
                xq_sp.add(big_n8),
                I::sub(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
            );

            let h1_s19mjs5d_pw_s3bmjs7f = I::xh1(fwd, s19mjs5d_pw_s3bmjs7f);
            I::store(
                xq_sp.add(big_n1),
                I::add(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
            );
            I::store(
                xq_sp.add(big_n9),
                I::sub(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
            );

            let w8_a19m1a5d_mj_a3bm1a7f = I::xw8(fwd, a19m1a5d_mj_a3bm1a7f);
            I::store(
                xq_sp.add(big_n2),
                I::add(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
            );
            I::store(
                xq_sp.add(big_na),
                I::sub(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
            );

            let h3_s19pjs5d_mv_s3bpjs7f = I::xh3(fwd, s19pjs5d_mv_s3bpjs7f);
            I::store(
                xq_sp.add(big_n3),
                I::add(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
            );
            I::store(
                xq_sp.add(big_nb),
                I::sub(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
            );

            let j_a19p1a5d_m1_a3bp1a7f = I::xpj(fwd, a19p1a5d_m1_a3bp1a7f);
            I::store(
                xq_sp.add(big_n4),
                I::sub(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
            );
            I::store(
                xq_sp.add(big_nc),
                I::add(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
            );

            let hd_s19mjs5d_mw_s3bmjs7f = I::xhd(fwd, s19mjs5d_mw_s3bmjs7f);
            I::store(
                xq_sp.add(big_n5),
                I::sub(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
            );
            I::store(
                xq_sp.add(big_nd),
                I::add(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
            );

            let v8_a19m1a5d_pj_a3bm1a7f = I::xv8(fwd, a19m1a5d_pj_a3bm1a7f);
            I::store(
                xq_sp.add(big_n6),
                I::sub(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
            );
            I::store(
                xq_sp.add(big_ne),
                I::add(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
            );

            let hf_s19pjs5d_pv_s3bpjs7f = I::xhf(fwd, s19pjs5d_pv_s3bpjs7f);
            I::store(
                xq_sp.add(big_n7),
                I::sub(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
            );
            I::store(
                xq_sp.add(big_nf),
                I::add(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
            );

            q += I::COMPLEX_PER_REG;
        }
    }
}

#[inline(always)]
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

        let ab_0 = I::load(y_16p.add(0x00));
        let cd_0 = I::load(y_16p.add(0x02));
        let ef_0 = I::load(y_16p.add(0x04));
        let gh_0 = I::load(y_16p.add(0x06));
        let ij_0 = I::load(y_16p.add(0x08));
        let kl_0 = I::load(y_16p.add(0x0a));
        let mn_0 = I::load(y_16p.add(0x0c));
        let op_0 = I::load(y_16p.add(0x0e));
        let ab_1 = I::load(y_16p.add(0x10));
        let cd_1 = I::load(y_16p.add(0x12));
        let ef_1 = I::load(y_16p.add(0x14));
        let gh_1 = I::load(y_16p.add(0x16));
        let ij_1 = I::load(y_16p.add(0x18));
        let kl_1 = I::load(y_16p.add(0x1a));
        let mn_1 = I::load(y_16p.add(0x1c));
        let op_1 = I::load(y_16p.add(0x1e));

        let y0 = I::catlo(ab_0, ab_1);
        let y1 = I::mul(w1p, I::cathi(ab_0, ab_1));
        let y2 = I::mul(w2p, I::catlo(cd_0, cd_1));
        let y3 = I::mul(w3p, I::cathi(cd_0, cd_1));
        let y4 = I::mul(w4p, I::catlo(ef_0, ef_1));
        let y5 = I::mul(w5p, I::cathi(ef_0, ef_1));
        let y6 = I::mul(w6p, I::catlo(gh_0, gh_1));
        let y7 = I::mul(w7p, I::cathi(gh_0, gh_1));

        let y8 = I::mul(w8p, I::catlo(ij_0, ij_1));
        let y9 = I::mul(w9p, I::cathi(ij_0, ij_1));
        let ya = I::mul(wap, I::catlo(kl_0, kl_1));
        let yb = I::mul(wbp, I::cathi(kl_0, kl_1));
        let yc = I::mul(wcp, I::catlo(mn_0, mn_1));
        let yd = I::mul(wdp, I::cathi(mn_0, mn_1));
        let ye = I::mul(wep, I::catlo(op_0, op_1));
        let yf = I::mul(wfp, I::cathi(op_0, op_1));

        let a08 = I::add(y0, y8);
        let s08 = I::sub(y0, y8);
        let a4c = I::add(y4, yc);
        let s4c = I::sub(y4, yc);
        let a2a = I::add(y2, ya);
        let s2a = I::sub(y2, ya);
        let a6e = I::add(y6, ye);
        let s6e = I::sub(y6, ye);
        let a19 = I::add(y1, y9);
        let s19 = I::sub(y1, y9);
        let a5d = I::add(y5, yd);
        let s5d = I::sub(y5, yd);
        let a3b = I::add(y3, yb);
        let s3b = I::sub(y3, yb);
        let a7f = I::add(y7, yf);
        let s7f = I::sub(y7, yf);

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

        I::store(
            x_p.add(big_n0),
            I::add(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );
        I::store(
            x_p.add(big_n8),
            I::sub(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );

        let h1_s19mjs5d_pw_s3bmjs7f = I::xh1(fwd, s19mjs5d_pw_s3bmjs7f);
        I::store(
            x_p.add(big_n1),
            I::add(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );
        I::store(
            x_p.add(big_n9),
            I::sub(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );

        let w8_a19m1a5d_mj_a3bm1a7f = I::xw8(fwd, a19m1a5d_mj_a3bm1a7f);
        I::store(
            x_p.add(big_n2),
            I::add(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );
        I::store(
            x_p.add(big_na),
            I::sub(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );

        let h3_s19pjs5d_mv_s3bpjs7f = I::xh3(fwd, s19pjs5d_mv_s3bpjs7f);
        I::store(
            x_p.add(big_n3),
            I::add(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );
        I::store(
            x_p.add(big_nb),
            I::sub(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );

        let j_a19p1a5d_m1_a3bp1a7f = I::xpj(fwd, a19p1a5d_m1_a3bp1a7f);
        I::store(
            x_p.add(big_n4),
            I::sub(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );
        I::store(
            x_p.add(big_nc),
            I::add(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );

        let hd_s19mjs5d_mw_s3bmjs7f = I::xhd(fwd, s19mjs5d_mw_s3bmjs7f);
        I::store(
            x_p.add(big_n5),
            I::sub(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );
        I::store(
            x_p.add(big_nd),
            I::add(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );

        let v8_a19m1a5d_pj_a3bm1a7f = I::xv8(fwd, a19m1a5d_pj_a3bm1a7f);
        I::store(
            x_p.add(big_n6),
            I::sub(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );
        I::store(
            x_p.add(big_ne),
            I::add(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );

        let hf_s19pjs5d_pv_s3bpjs7f = I::xhf(fwd, s19pjs5d_pv_s3bpjs7f);
        I::store(
            x_p.add(big_n7),
            I::sub(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );
        I::store(
            x_p.add(big_nf),
            I::add(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );

        p += 2;
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

        let z0 = I::load(zq.add(0x0));
        let z1 = I::load(zq.add(s));
        let z2 = I::load(zq.add(s * 0x2));
        let z3 = I::load(zq.add(s * 0x3));
        let z4 = I::load(zq.add(s * 0x4));
        let z5 = I::load(zq.add(s * 0x5));
        let z6 = I::load(zq.add(s * 0x6));
        let z7 = I::load(zq.add(s * 0x7));
        let z8 = I::load(zq.add(s * 0x8));
        let z9 = I::load(zq.add(s * 0x9));
        let za = I::load(zq.add(s * 0xa));
        let zb = I::load(zq.add(s * 0xb));
        let zc = I::load(zq.add(s * 0xc));
        let zd = I::load(zq.add(s * 0xd));
        let ze = I::load(zq.add(s * 0xe));
        let zf = I::load(zq.add(s * 0xf));

        let a08 = I::add(z0, z8);
        let s08 = I::sub(z0, z8);
        let a4c = I::add(z4, zc);
        let s4c = I::sub(z4, zc);
        let a2a = I::add(z2, za);
        let s2a = I::sub(z2, za);
        let a6e = I::add(z6, ze);
        let s6e = I::sub(z6, ze);
        let a19 = I::add(z1, z9);
        let s19 = I::sub(z1, z9);
        let a5d = I::add(z5, zd);
        let s5d = I::sub(z5, zd);
        let a3b = I::add(z3, zb);
        let s3b = I::sub(z3, zb);
        let a7f = I::add(z7, zf);
        let s7f = I::sub(z7, zf);

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
            xq.add(0x0),
            I::add(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );
        I::store(
            xq.add(s),
            I::add(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );
        I::store(
            xq.add(s * 0x2),
            I::add(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );
        I::store(
            xq.add(s * 0x3),
            I::add(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );
        I::store(
            xq.add(s * 0x4),
            I::sub(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );
        I::store(
            xq.add(s * 0x5),
            I::sub(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );
        I::store(
            xq.add(s * 0x6),
            I::sub(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );
        I::store(
            xq.add(s * 0x7),
            I::sub(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );

        I::store(
            xq.add(s * 0x8),
            I::sub(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );
        I::store(
            xq.add(s * 0x9),
            I::sub(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );
        I::store(
            xq.add(s * 0xa),
            I::sub(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );
        I::store(
            xq.add(s * 0xb),
            I::sub(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );
        I::store(
            xq.add(s * 0xc),
            I::add(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );
        I::store(
            xq.add(s * 0xd),
            I::add(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );
        I::store(
            xq.add(s * 0xe),
            I::add(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );
        I::store(
            xq.add(s * 0xf),
            I::add(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );

        q += I::COMPLEX_PER_REG;
    }
}

macro_rules! dit16_impl {
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
                    end_2::<$xn>(FWD, 1 << 1, 1 << 4, y, x, true);
                    $core1______(FWD, 1 << 5, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_06<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 2, 1 << 4, y, x, true);
                    $core1______(FWD, 1 << 6, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_07<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_8::<$xn>(FWD, 1 << 3, 1 << 4, y, x, true);
                    $core1______(FWD, 1 << 7, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_08<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end16::<$xn>(FWD, 1 << 4, 1 << 4, y, x, true);
                    $core1______(FWD, 1 << 8, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_09<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 8, x, y, false);
                    core_::<$xn>(FWD, 1 << 5, 1 << 4, y, x, w);
                    $core1______(FWD, 1 << 9, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_10<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 02, 1 << 8, x, y, false);
                    core_::<$xn>(FWD, 1 << 06, 1 << 4, y, x, w);
                    $core1______(FWD, 1 << 10, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_11<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_8::<$xn>(FWD, 1 << 03, 1 << 08, x, y, false);
                    core_::<$xn>(FWD, 1 << 07, 1 << 04, y, x, w);
                    $core1______(FWD, 1 << 11, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_12<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end16::<$xn>(FWD, 1 << 04, 1 << 08, x, y, false);
                    core_::<$xn>(FWD, 1 << 08, 1 << 04, y, x, w);
                    $core1______(FWD, 1 << 12, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_13<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 01, 1 << 12, y, x, true);
                    core_::<$xn>(FWD, 1 << 05, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 09, 1 << 04, y, x, w);
                    $core1______(FWD, 1 << 13, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_14<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 02, 1 << 12, y, x, true);
                    core_::<$xn>(FWD, 1 << 06, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 04, y, x, w);
                    $core1______(FWD, 1 << 14, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_15<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_8::<$xn>(FWD, 1 << 03, 1 << 12, y, x, true);
                    core_::<$xn>(FWD, 1 << 07, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 11, 1 << 04, y, x, w);
                    $core1______(FWD, 1 << 15, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_16<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end16::<$xn>(FWD, 1 << 04, 1 << 12, y, x, true);
                    core_::<$xn>(FWD, 1 << 08, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 12, 1 << 04, y, x, w);
                    $core1______(FWD, 1 << 16, 1 << 00, x, y, w);
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

dit16_impl! {
    pub static DIT16_SCALAR = Fft {
        core_1: core_::<Scalar>,
        native: Scalar,
        x1: Scalar,
    };

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub static DIT16_AVX = Fft {
        core_1: core_x2::<AvxX2>,
        native: AvxX2,
        x1: AvxX1,
        target: "avx",
    };

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub static DIT16_FMA = Fft {
        core_1: core_x2::<FmaX2>,
        native: FmaX2,
        x1: FmaX1,
        target: "fma",
    };

    #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
    pub static DIT16_AVX512 = Fft {
        core_1: core_x2::<Avx512X2>,
        native: Avx512X4,
        x1: Avx512X1,
        target: "avx512f",
    };
}

pub(crate) fn runtime_fft() -> crate::FftImpl {
    #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
    if is_x86_feature_detected!("avx512f") {
        return DIT16_AVX512;
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if is_x86_feature_detected!("fma") {
        return DIT16_FMA;
    } else if is_x86_feature_detected!("avx") {
        return DIT16_AVX;
    }

    DIT16_SCALAR
}
