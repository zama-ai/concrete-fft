use crate::fft_simd::FftSimd16;
use crate::impl_main_fn;
use crate::twiddles::{twid, twid_t};
use crate::{c64, MAX_EXP};
use dyn_stack::DynStack;

pub(crate) use crate::dit8::fwdend_2_1;
pub(crate) use crate::dit8::fwdend_2_s;
pub(crate) use crate::dit8::fwdend_4_1;
pub(crate) use crate::dit8::fwdend_4_s;
pub(crate) use crate::dit8::fwdend_8_1;
pub(crate) use crate::dit8::fwdend_8_s;
pub(crate) use crate::dit8::invend_2_1;
pub(crate) use crate::dit8::invend_2_s;
pub(crate) use crate::dit8::invend_4_1;
pub(crate) use crate::dit8::invend_4_s;
pub(crate) use crate::dit8::invend_8_1;
pub(crate) use crate::dit8::invend_8_s;

// forward butterfly
#[inline(always)]
unsafe fn fwdcore_s<S: FftSimd16>(n: usize, s: usize, x: *mut c64, y: *mut c64, w: *const c64) {
    debug_assert_ne!(s, 1);

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

        let w1p = S::duppz3(&*twid_t(16, big_n, 1, w, sp));
        let w2p = S::duppz3(&*twid_t(16, big_n, 2, w, sp));
        let w3p = S::duppz3(&*twid_t(16, big_n, 3, w, sp));
        let w4p = S::duppz3(&*twid_t(16, big_n, 4, w, sp));
        let w5p = S::duppz3(&*twid_t(16, big_n, 5, w, sp));
        let w6p = S::duppz3(&*twid_t(16, big_n, 6, w, sp));
        let w7p = S::duppz3(&*twid_t(16, big_n, 7, w, sp));
        let w8p = S::duppz3(&*twid_t(16, big_n, 8, w, sp));
        let w9p = S::duppz3(&*twid_t(16, big_n, 9, w, sp));
        let wap = S::duppz3(&*twid_t(16, big_n, 10, w, sp));
        let wbp = S::duppz3(&*twid_t(16, big_n, 11, w, sp));
        let wcp = S::duppz3(&*twid_t(16, big_n, 12, w, sp));
        let wdp = S::duppz3(&*twid_t(16, big_n, 13, w, sp));
        let wep = S::duppz3(&*twid_t(16, big_n, 14, w, sp));
        let wfp = S::duppz3(&*twid_t(16, big_n, 15, w, sp));

        let mut q = 0;
        while q < s {
            let xq_sp = x.add(q + sp);
            let yq_s16p = y.add(q + s16p);

            let y0 = S::getpz2(yq_s16p.add(0));
            let y1 = S::mulpz2(w1p, S::getpz2(yq_s16p.add(s)));
            let y2 = S::mulpz2(w2p, S::getpz2(yq_s16p.add(s * 0x2)));
            let y3 = S::mulpz2(w3p, S::getpz2(yq_s16p.add(s * 0x3)));
            let y4 = S::mulpz2(w4p, S::getpz2(yq_s16p.add(s * 0x4)));
            let y5 = S::mulpz2(w5p, S::getpz2(yq_s16p.add(s * 0x5)));
            let y6 = S::mulpz2(w6p, S::getpz2(yq_s16p.add(s * 0x6)));
            let y7 = S::mulpz2(w7p, S::getpz2(yq_s16p.add(s * 0x7)));
            let y8 = S::mulpz2(w8p, S::getpz2(yq_s16p.add(s * 0x8)));
            let y9 = S::mulpz2(w9p, S::getpz2(yq_s16p.add(s * 0x9)));
            let ya = S::mulpz2(wap, S::getpz2(yq_s16p.add(s * 0xa)));
            let yb = S::mulpz2(wbp, S::getpz2(yq_s16p.add(s * 0xb)));
            let yc = S::mulpz2(wcp, S::getpz2(yq_s16p.add(s * 0xc)));
            let yd = S::mulpz2(wdp, S::getpz2(yq_s16p.add(s * 0xd)));
            let ye = S::mulpz2(wep, S::getpz2(yq_s16p.add(s * 0xe)));
            let yf = S::mulpz2(wfp, S::getpz2(yq_s16p.add(s * 0xf)));

            let a08 = S::addpz2(y0, y8);
            let s08 = S::subpz2(y0, y8);
            let a4c = S::addpz2(y4, yc);
            let s4c = S::subpz2(y4, yc);
            let a2a = S::addpz2(y2, ya);
            let s2a = S::subpz2(y2, ya);
            let a6e = S::addpz2(y6, ye);
            let s6e = S::subpz2(y6, ye);
            let a19 = S::addpz2(y1, y9);
            let s19 = S::subpz2(y1, y9);
            let a5d = S::addpz2(y5, yd);
            let s5d = S::subpz2(y5, yd);
            let a3b = S::addpz2(y3, yb);
            let s3b = S::subpz2(y3, yb);
            let a7f = S::addpz2(y7, yf);
            let s7f = S::subpz2(y7, yf);

            let js4c = S::jxpz2(s4c);
            let js6e = S::jxpz2(s6e);
            let js5d = S::jxpz2(s5d);
            let js7f = S::jxpz2(s7f);

            let a08p1a4c = S::addpz2(a08, a4c);
            let s08mjs4c = S::subpz2(s08, js4c);
            let a08m1a4c = S::subpz2(a08, a4c);
            let s08pjs4c = S::addpz2(s08, js4c);
            let a2ap1a6e = S::addpz2(a2a, a6e);
            let s2amjs6e = S::subpz2(s2a, js6e);
            let a2am1a6e = S::subpz2(a2a, a6e);
            let s2apjs6e = S::addpz2(s2a, js6e);
            let a19p1a5d = S::addpz2(a19, a5d);
            let s19mjs5d = S::subpz2(s19, js5d);
            let a19m1a5d = S::subpz2(a19, a5d);
            let s19pjs5d = S::addpz2(s19, js5d);
            let a3bp1a7f = S::addpz2(a3b, a7f);
            let s3bmjs7f = S::subpz2(s3b, js7f);
            let a3bm1a7f = S::subpz2(a3b, a7f);
            let s3bpjs7f = S::addpz2(s3b, js7f);

            let w8_s2amjs6e = S::w8xpz2(s2amjs6e);
            let j_a2am1a6e = S::jxpz2(a2am1a6e);
            let v8_s2apjs6e = S::v8xpz2(s2apjs6e);

            let a08p1a4c_p1_a2ap1a6e = S::addpz2(a08p1a4c, a2ap1a6e);
            let s08mjs4c_pw_s2amjs6e = S::addpz2(s08mjs4c, w8_s2amjs6e);
            let a08m1a4c_mj_a2am1a6e = S::subpz2(a08m1a4c, j_a2am1a6e);
            let s08pjs4c_mv_s2apjs6e = S::subpz2(s08pjs4c, v8_s2apjs6e);
            let a08p1a4c_m1_a2ap1a6e = S::subpz2(a08p1a4c, a2ap1a6e);
            let s08mjs4c_mw_s2amjs6e = S::subpz2(s08mjs4c, w8_s2amjs6e);
            let a08m1a4c_pj_a2am1a6e = S::addpz2(a08m1a4c, j_a2am1a6e);
            let s08pjs4c_pv_s2apjs6e = S::addpz2(s08pjs4c, v8_s2apjs6e);

            let w8_s3bmjs7f = S::w8xpz2(s3bmjs7f);
            let j_a3bm1a7f = S::jxpz2(a3bm1a7f);
            let v8_s3bpjs7f = S::v8xpz2(s3bpjs7f);

            let a19p1a5d_p1_a3bp1a7f = S::addpz2(a19p1a5d, a3bp1a7f);
            let s19mjs5d_pw_s3bmjs7f = S::addpz2(s19mjs5d, w8_s3bmjs7f);
            let a19m1a5d_mj_a3bm1a7f = S::subpz2(a19m1a5d, j_a3bm1a7f);
            let s19pjs5d_mv_s3bpjs7f = S::subpz2(s19pjs5d, v8_s3bpjs7f);
            let a19p1a5d_m1_a3bp1a7f = S::subpz2(a19p1a5d, a3bp1a7f);
            let s19mjs5d_mw_s3bmjs7f = S::subpz2(s19mjs5d, w8_s3bmjs7f);
            let a19m1a5d_pj_a3bm1a7f = S::addpz2(a19m1a5d, j_a3bm1a7f);
            let s19pjs5d_pv_s3bpjs7f = S::addpz2(s19pjs5d, v8_s3bpjs7f);

            S::setpz2(
                xq_sp.add(big_n0),
                S::addpz2(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
            );
            S::setpz2(
                xq_sp.add(big_n8),
                S::subpz2(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
            );

            let h1_s19mjs5d_pw_s3bmjs7f = S::h1xpz2(s19mjs5d_pw_s3bmjs7f);
            S::setpz2(
                xq_sp.add(big_n1),
                S::addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
            );
            S::setpz2(
                xq_sp.add(big_n9),
                S::subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
            );

            let w8_a19m1a5d_mj_a3bm1a7f = S::w8xpz2(a19m1a5d_mj_a3bm1a7f);
            S::setpz2(
                xq_sp.add(big_n2),
                S::addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
            );
            S::setpz2(
                xq_sp.add(big_na),
                S::subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
            );

            let h3_s19pjs5d_mv_s3bpjs7f = S::h3xpz2(s19pjs5d_mv_s3bpjs7f);
            S::setpz2(
                xq_sp.add(big_n3),
                S::addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
            );
            S::setpz2(
                xq_sp.add(big_nb),
                S::subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
            );

            let j_a19p1a5d_m1_a3bp1a7f = S::jxpz2(a19p1a5d_m1_a3bp1a7f);
            S::setpz2(
                xq_sp.add(big_n4),
                S::subpz2(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
            );
            S::setpz2(
                xq_sp.add(big_nc),
                S::addpz2(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
            );

            let hd_s19mjs5d_mw_s3bmjs7f = S::hdxpz2(s19mjs5d_mw_s3bmjs7f);
            S::setpz2(
                xq_sp.add(big_n5),
                S::subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
            );
            S::setpz2(
                xq_sp.add(big_nd),
                S::addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
            );

            let v8_a19m1a5d_pj_a3bm1a7f = S::v8xpz2(a19m1a5d_pj_a3bm1a7f);
            S::setpz2(
                xq_sp.add(big_n6),
                S::subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
            );
            S::setpz2(
                xq_sp.add(big_ne),
                S::addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
            );

            let hf_s19pjs5d_pv_s3bpjs7f = S::hfxpz2(s19pjs5d_pv_s3bpjs7f);
            S::setpz2(
                xq_sp.add(big_n7),
                S::subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
            );
            S::setpz2(
                xq_sp.add(big_nf),
                S::addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
            );

            q += 2;
        }
    }
}

#[inline(always)]
unsafe fn fwdcore_1<S: FftSimd16>(big_n: usize, s: usize, x: *mut c64, y: *mut c64, w: *const c64) {
    debug_assert_eq!(s, 1);

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

        let w1p = S::getpz2(twid(16, big_n, 1, w, p));
        let w2p = S::getpz2(twid(16, big_n, 2, w, p));
        let w3p = S::getpz2(twid(16, big_n, 3, w, p));
        let w4p = S::getpz2(twid(16, big_n, 4, w, p));
        let w5p = S::getpz2(twid(16, big_n, 5, w, p));
        let w6p = S::getpz2(twid(16, big_n, 6, w, p));
        let w7p = S::getpz2(twid(16, big_n, 7, w, p));
        let w8p = S::getpz2(twid(16, big_n, 8, w, p));
        let w9p = S::getpz2(twid(16, big_n, 9, w, p));
        let wap = S::getpz2(twid(16, big_n, 10, w, p));
        let wbp = S::getpz2(twid(16, big_n, 11, w, p));
        let wcp = S::getpz2(twid(16, big_n, 12, w, p));
        let wdp = S::getpz2(twid(16, big_n, 13, w, p));
        let wep = S::getpz2(twid(16, big_n, 14, w, p));
        let wfp = S::getpz2(twid(16, big_n, 15, w, p));

        let ab_0 = S::getpz2(y_16p.add(0x00));
        let cd_0 = S::getpz2(y_16p.add(0x02));
        let ef_0 = S::getpz2(y_16p.add(0x04));
        let gh_0 = S::getpz2(y_16p.add(0x06));
        let ij_0 = S::getpz2(y_16p.add(0x08));
        let kl_0 = S::getpz2(y_16p.add(0x0a));
        let mn_0 = S::getpz2(y_16p.add(0x0c));
        let op_0 = S::getpz2(y_16p.add(0x0e));
        let ab_1 = S::getpz2(y_16p.add(0x10));
        let cd_1 = S::getpz2(y_16p.add(0x12));
        let ef_1 = S::getpz2(y_16p.add(0x14));
        let gh_1 = S::getpz2(y_16p.add(0x16));
        let ij_1 = S::getpz2(y_16p.add(0x18));
        let kl_1 = S::getpz2(y_16p.add(0x1a));
        let mn_1 = S::getpz2(y_16p.add(0x1c));
        let op_1 = S::getpz2(y_16p.add(0x1e));

        let y0 = S::catlo(ab_0, ab_1);
        let y1 = S::mulpz2(w1p, S::cathi(ab_0, ab_1));
        let y2 = S::mulpz2(w2p, S::catlo(cd_0, cd_1));
        let y3 = S::mulpz2(w3p, S::cathi(cd_0, cd_1));
        let y4 = S::mulpz2(w4p, S::catlo(ef_0, ef_1));
        let y5 = S::mulpz2(w5p, S::cathi(ef_0, ef_1));
        let y6 = S::mulpz2(w6p, S::catlo(gh_0, gh_1));
        let y7 = S::mulpz2(w7p, S::cathi(gh_0, gh_1));

        let y8 = S::mulpz2(w8p, S::catlo(ij_0, ij_1));
        let y9 = S::mulpz2(w9p, S::cathi(ij_0, ij_1));
        let ya = S::mulpz2(wap, S::catlo(kl_0, kl_1));
        let yb = S::mulpz2(wbp, S::cathi(kl_0, kl_1));
        let yc = S::mulpz2(wcp, S::catlo(mn_0, mn_1));
        let yd = S::mulpz2(wdp, S::cathi(mn_0, mn_1));
        let ye = S::mulpz2(wep, S::catlo(op_0, op_1));
        let yf = S::mulpz2(wfp, S::cathi(op_0, op_1));

        let a08 = S::addpz2(y0, y8);
        let s08 = S::subpz2(y0, y8);
        let a4c = S::addpz2(y4, yc);
        let s4c = S::subpz2(y4, yc);
        let a2a = S::addpz2(y2, ya);
        let s2a = S::subpz2(y2, ya);
        let a6e = S::addpz2(y6, ye);
        let s6e = S::subpz2(y6, ye);
        let a19 = S::addpz2(y1, y9);
        let s19 = S::subpz2(y1, y9);
        let a5d = S::addpz2(y5, yd);
        let s5d = S::subpz2(y5, yd);
        let a3b = S::addpz2(y3, yb);
        let s3b = S::subpz2(y3, yb);
        let a7f = S::addpz2(y7, yf);
        let s7f = S::subpz2(y7, yf);

        let js4c = S::jxpz2(s4c);
        let js6e = S::jxpz2(s6e);
        let js5d = S::jxpz2(s5d);
        let js7f = S::jxpz2(s7f);

        let a08p1a4c = S::addpz2(a08, a4c);
        let s08mjs4c = S::subpz2(s08, js4c);
        let a08m1a4c = S::subpz2(a08, a4c);
        let s08pjs4c = S::addpz2(s08, js4c);
        let a2ap1a6e = S::addpz2(a2a, a6e);
        let s2amjs6e = S::subpz2(s2a, js6e);
        let a2am1a6e = S::subpz2(a2a, a6e);
        let s2apjs6e = S::addpz2(s2a, js6e);
        let a19p1a5d = S::addpz2(a19, a5d);
        let s19mjs5d = S::subpz2(s19, js5d);
        let a19m1a5d = S::subpz2(a19, a5d);
        let s19pjs5d = S::addpz2(s19, js5d);
        let a3bp1a7f = S::addpz2(a3b, a7f);
        let s3bmjs7f = S::subpz2(s3b, js7f);
        let a3bm1a7f = S::subpz2(a3b, a7f);
        let s3bpjs7f = S::addpz2(s3b, js7f);

        let w8_s2amjs6e = S::w8xpz2(s2amjs6e);
        let j_a2am1a6e = S::jxpz2(a2am1a6e);
        let v8_s2apjs6e = S::v8xpz2(s2apjs6e);

        let a08p1a4c_p1_a2ap1a6e = S::addpz2(a08p1a4c, a2ap1a6e);
        let s08mjs4c_pw_s2amjs6e = S::addpz2(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_mj_a2am1a6e = S::subpz2(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_mv_s2apjs6e = S::subpz2(s08pjs4c, v8_s2apjs6e);
        let a08p1a4c_m1_a2ap1a6e = S::subpz2(a08p1a4c, a2ap1a6e);
        let s08mjs4c_mw_s2amjs6e = S::subpz2(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_pj_a2am1a6e = S::addpz2(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_pv_s2apjs6e = S::addpz2(s08pjs4c, v8_s2apjs6e);

        let w8_s3bmjs7f = S::w8xpz2(s3bmjs7f);
        let j_a3bm1a7f = S::jxpz2(a3bm1a7f);
        let v8_s3bpjs7f = S::v8xpz2(s3bpjs7f);

        let a19p1a5d_p1_a3bp1a7f = S::addpz2(a19p1a5d, a3bp1a7f);
        let s19mjs5d_pw_s3bmjs7f = S::addpz2(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_mj_a3bm1a7f = S::subpz2(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_mv_s3bpjs7f = S::subpz2(s19pjs5d, v8_s3bpjs7f);
        let a19p1a5d_m1_a3bp1a7f = S::subpz2(a19p1a5d, a3bp1a7f);
        let s19mjs5d_mw_s3bmjs7f = S::subpz2(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_pj_a3bm1a7f = S::addpz2(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_pv_s3bpjs7f = S::addpz2(s19pjs5d, v8_s3bpjs7f);

        S::setpz2(
            x_p.add(big_n0),
            S::addpz2(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );
        S::setpz2(
            x_p.add(big_n8),
            S::subpz2(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );

        let h1_s19mjs5d_pw_s3bmjs7f = S::h1xpz2(s19mjs5d_pw_s3bmjs7f);
        S::setpz2(
            x_p.add(big_n1),
            S::addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );
        S::setpz2(
            x_p.add(big_n9),
            S::subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );

        let w8_a19m1a5d_mj_a3bm1a7f = S::w8xpz2(a19m1a5d_mj_a3bm1a7f);
        S::setpz2(
            x_p.add(big_n2),
            S::addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );
        S::setpz2(
            x_p.add(big_na),
            S::subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );

        let h3_s19pjs5d_mv_s3bpjs7f = S::h3xpz2(s19pjs5d_mv_s3bpjs7f);
        S::setpz2(
            x_p.add(big_n3),
            S::addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );
        S::setpz2(
            x_p.add(big_nb),
            S::subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );

        let j_a19p1a5d_m1_a3bp1a7f = S::jxpz2(a19p1a5d_m1_a3bp1a7f);
        S::setpz2(
            x_p.add(big_n4),
            S::subpz2(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );
        S::setpz2(
            x_p.add(big_nc),
            S::addpz2(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );

        let hd_s19mjs5d_mw_s3bmjs7f = S::hdxpz2(s19mjs5d_mw_s3bmjs7f);
        S::setpz2(
            x_p.add(big_n5),
            S::subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );
        S::setpz2(
            x_p.add(big_nd),
            S::addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );

        let v8_a19m1a5d_pj_a3bm1a7f = S::v8xpz2(a19m1a5d_pj_a3bm1a7f);
        S::setpz2(
            x_p.add(big_n6),
            S::subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );
        S::setpz2(
            x_p.add(big_ne),
            S::addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );

        let hf_s19pjs5d_pv_s3bpjs7f = S::hfxpz2(s19pjs5d_pv_s3bpjs7f);
        S::setpz2(
            x_p.add(big_n7),
            S::subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );
        S::setpz2(
            x_p.add(big_nf),
            S::addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );

        p += 2;
    }
}

// backward butterfly
#[inline(always)]
unsafe fn invcore_s<S: FftSimd16>(n: usize, s: usize, x: *mut c64, y: *mut c64, w: *const c64) {
    debug_assert_ne!(s, 1);

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

        let w1p = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 1, w, sp)));
        let w2p = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 2, w, sp)));
        let w3p = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 3, w, sp)));
        let w4p = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 4, w, sp)));
        let w5p = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 5, w, sp)));
        let w6p = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 6, w, sp)));
        let w7p = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 7, w, sp)));
        let w8p = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 8, w, sp)));
        let w9p = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 9, w, sp)));
        let wap = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 10, w, sp)));
        let wbp = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 11, w, sp)));
        let wcp = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 12, w, sp)));
        let wdp = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 13, w, sp)));
        let wep = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 14, w, sp)));
        let wfp = S::cnjpz2(S::duppz3(&*twid_t(16, big_n, 15, w, sp)));

        let mut q = 0;
        while q < s {
            let xq_sp = x.add(q + sp);
            let yq_s16p = y.add(q + s16p);

            let y0 = S::getpz2(yq_s16p.add(0x0));
            let y1 = S::mulpz2(w1p, S::getpz2(yq_s16p.add(s)));
            let y2 = S::mulpz2(w2p, S::getpz2(yq_s16p.add(s * 0x2)));
            let y3 = S::mulpz2(w3p, S::getpz2(yq_s16p.add(s * 0x3)));
            let y4 = S::mulpz2(w4p, S::getpz2(yq_s16p.add(s * 0x4)));
            let y5 = S::mulpz2(w5p, S::getpz2(yq_s16p.add(s * 0x5)));
            let y6 = S::mulpz2(w6p, S::getpz2(yq_s16p.add(s * 0x6)));
            let y7 = S::mulpz2(w7p, S::getpz2(yq_s16p.add(s * 0x7)));
            let y8 = S::mulpz2(w8p, S::getpz2(yq_s16p.add(s * 0x8)));
            let y9 = S::mulpz2(w9p, S::getpz2(yq_s16p.add(s * 0x9)));
            let ya = S::mulpz2(wap, S::getpz2(yq_s16p.add(s * 0xa)));
            let yb = S::mulpz2(wbp, S::getpz2(yq_s16p.add(s * 0xb)));
            let yc = S::mulpz2(wcp, S::getpz2(yq_s16p.add(s * 0xc)));
            let yd = S::mulpz2(wdp, S::getpz2(yq_s16p.add(s * 0xd)));
            let ye = S::mulpz2(wep, S::getpz2(yq_s16p.add(s * 0xe)));
            let yf = S::mulpz2(wfp, S::getpz2(yq_s16p.add(s * 0xf)));

            let a08 = S::addpz2(y0, y8);
            let s08 = S::subpz2(y0, y8);
            let a4c = S::addpz2(y4, yc);
            let s4c = S::subpz2(y4, yc);
            let a2a = S::addpz2(y2, ya);
            let s2a = S::subpz2(y2, ya);
            let a6e = S::addpz2(y6, ye);
            let s6e = S::subpz2(y6, ye);
            let a19 = S::addpz2(y1, y9);
            let s19 = S::subpz2(y1, y9);
            let a5d = S::addpz2(y5, yd);
            let s5d = S::subpz2(y5, yd);
            let a3b = S::addpz2(y3, yb);
            let s3b = S::subpz2(y3, yb);
            let a7f = S::addpz2(y7, yf);
            let s7f = S::subpz2(y7, yf);

            let js4c = S::jxpz2(s4c);
            let js6e = S::jxpz2(s6e);
            let js5d = S::jxpz2(s5d);
            let js7f = S::jxpz2(s7f);

            let a08p1a4c = S::addpz2(a08, a4c);
            let s08mjs4c = S::subpz2(s08, js4c);
            let a08m1a4c = S::subpz2(a08, a4c);
            let s08pjs4c = S::addpz2(s08, js4c);
            let a2ap1a6e = S::addpz2(a2a, a6e);
            let s2amjs6e = S::subpz2(s2a, js6e);
            let a2am1a6e = S::subpz2(a2a, a6e);
            let s2apjs6e = S::addpz2(s2a, js6e);
            let a19p1a5d = S::addpz2(a19, a5d);
            let s19mjs5d = S::subpz2(s19, js5d);
            let a19m1a5d = S::subpz2(a19, a5d);
            let s19pjs5d = S::addpz2(s19, js5d);
            let a3bp1a7f = S::addpz2(a3b, a7f);
            let s3bmjs7f = S::subpz2(s3b, js7f);
            let a3bm1a7f = S::subpz2(a3b, a7f);
            let s3bpjs7f = S::addpz2(s3b, js7f);

            let w8_s2amjs6e = S::w8xpz2(s2amjs6e);
            let j_a2am1a6e = S::jxpz2(a2am1a6e);
            let v8_s2apjs6e = S::v8xpz2(s2apjs6e);

            let a08p1a4c_p1_a2ap1a6e = S::addpz2(a08p1a4c, a2ap1a6e);
            let s08mjs4c_pw_s2amjs6e = S::addpz2(s08mjs4c, w8_s2amjs6e);
            let a08m1a4c_mj_a2am1a6e = S::subpz2(a08m1a4c, j_a2am1a6e);
            let s08pjs4c_mv_s2apjs6e = S::subpz2(s08pjs4c, v8_s2apjs6e);
            let a08p1a4c_m1_a2ap1a6e = S::subpz2(a08p1a4c, a2ap1a6e);
            let s08mjs4c_mw_s2amjs6e = S::subpz2(s08mjs4c, w8_s2amjs6e);
            let a08m1a4c_pj_a2am1a6e = S::addpz2(a08m1a4c, j_a2am1a6e);
            let s08pjs4c_pv_s2apjs6e = S::addpz2(s08pjs4c, v8_s2apjs6e);

            let w8_s3bmjs7f = S::w8xpz2(s3bmjs7f);
            let j_a3bm1a7f = S::jxpz2(a3bm1a7f);
            let v8_s3bpjs7f = S::v8xpz2(s3bpjs7f);

            let a19p1a5d_p1_a3bp1a7f = S::addpz2(a19p1a5d, a3bp1a7f);
            let s19mjs5d_pw_s3bmjs7f = S::addpz2(s19mjs5d, w8_s3bmjs7f);
            let a19m1a5d_mj_a3bm1a7f = S::subpz2(a19m1a5d, j_a3bm1a7f);
            let s19pjs5d_mv_s3bpjs7f = S::subpz2(s19pjs5d, v8_s3bpjs7f);
            let a19p1a5d_m1_a3bp1a7f = S::subpz2(a19p1a5d, a3bp1a7f);
            let s19mjs5d_mw_s3bmjs7f = S::subpz2(s19mjs5d, w8_s3bmjs7f);
            let a19m1a5d_pj_a3bm1a7f = S::addpz2(a19m1a5d, j_a3bm1a7f);
            let s19pjs5d_pv_s3bpjs7f = S::addpz2(s19pjs5d, v8_s3bpjs7f);

            S::setpz2(
                xq_sp.add(big_n0),
                S::addpz2(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
            );
            S::setpz2(
                xq_sp.add(big_n8),
                S::subpz2(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
            );

            let hf_s19pjs5d_pv_s3bpjs7f = S::hfxpz2(s19pjs5d_pv_s3bpjs7f);
            S::setpz2(
                xq_sp.add(big_n1),
                S::addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
            );
            S::setpz2(
                xq_sp.add(big_n9),
                S::subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
            );

            let v8_a19m1a5d_pj_a3bm1a7f = S::v8xpz2(a19m1a5d_pj_a3bm1a7f);
            S::setpz2(
                xq_sp.add(big_n2),
                S::addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
            );
            S::setpz2(
                xq_sp.add(big_na),
                S::subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
            );

            let hd_s19mjs5d_mw_s3bmjs7f = S::hdxpz2(s19mjs5d_mw_s3bmjs7f);
            S::setpz2(
                xq_sp.add(big_n3),
                S::addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
            );
            S::setpz2(
                xq_sp.add(big_nb),
                S::subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
            );

            let j_a19p1a5d_m1_a3bp1a7f = S::jxpz2(a19p1a5d_m1_a3bp1a7f);
            S::setpz2(
                xq_sp.add(big_n4),
                S::addpz2(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
            );
            S::setpz2(
                xq_sp.add(big_nc),
                S::subpz2(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
            );

            let h3_s19pjs5d_mv_s3bpjs7f = S::h3xpz2(s19pjs5d_mv_s3bpjs7f);
            S::setpz2(
                xq_sp.add(big_n5),
                S::subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
            );
            S::setpz2(
                xq_sp.add(big_nd),
                S::addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
            );

            let w8_a19m1a5d_mj_a3bm1a7f = S::w8xpz2(a19m1a5d_mj_a3bm1a7f);
            S::setpz2(
                xq_sp.add(big_n6),
                S::subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
            );
            S::setpz2(
                xq_sp.add(big_ne),
                S::addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
            );

            let h1_s19mjs5d_pw_s3bmjs7f = S::h1xpz2(s19mjs5d_pw_s3bmjs7f);
            S::setpz2(
                xq_sp.add(big_n7),
                S::subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
            );
            S::setpz2(
                xq_sp.add(big_nf),
                S::addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
            );

            q += 2;
        }
    }
}

#[inline(always)]
unsafe fn invcore_1<S: FftSimd16>(big_n: usize, s: usize, x: *mut c64, y: *mut c64, w: *const c64) {
    debug_assert_eq!(s, 1);

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

        let w1p = S::cnjpz2(S::getpz2(twid(16, big_n, 1, w, p)));
        let w2p = S::cnjpz2(S::getpz2(twid(16, big_n, 2, w, p)));
        let w3p = S::cnjpz2(S::getpz2(twid(16, big_n, 3, w, p)));
        let w4p = S::cnjpz2(S::getpz2(twid(16, big_n, 4, w, p)));
        let w5p = S::cnjpz2(S::getpz2(twid(16, big_n, 5, w, p)));
        let w6p = S::cnjpz2(S::getpz2(twid(16, big_n, 6, w, p)));
        let w7p = S::cnjpz2(S::getpz2(twid(16, big_n, 7, w, p)));
        let w8p = S::cnjpz2(S::getpz2(twid(16, big_n, 8, w, p)));
        let w9p = S::cnjpz2(S::getpz2(twid(16, big_n, 9, w, p)));
        let wap = S::cnjpz2(S::getpz2(twid(16, big_n, 10, w, p)));
        let wbp = S::cnjpz2(S::getpz2(twid(16, big_n, 11, w, p)));
        let wcp = S::cnjpz2(S::getpz2(twid(16, big_n, 12, w, p)));
        let wdp = S::cnjpz2(S::getpz2(twid(16, big_n, 13, w, p)));
        let wep = S::cnjpz2(S::getpz2(twid(16, big_n, 14, w, p)));
        let wfp = S::cnjpz2(S::getpz2(twid(16, big_n, 15, w, p)));

        let ab_0 = S::getpz2(y_16p.add(0x00));
        let cd_0 = S::getpz2(y_16p.add(0x02));
        let ef_0 = S::getpz2(y_16p.add(0x04));
        let gh_0 = S::getpz2(y_16p.add(0x06));
        let ij_0 = S::getpz2(y_16p.add(0x08));
        let kl_0 = S::getpz2(y_16p.add(0x0a));
        let mn_0 = S::getpz2(y_16p.add(0x0c));
        let op_0 = S::getpz2(y_16p.add(0x0e));
        let ab_1 = S::getpz2(y_16p.add(0x10));
        let cd_1 = S::getpz2(y_16p.add(0x12));
        let ef_1 = S::getpz2(y_16p.add(0x14));
        let gh_1 = S::getpz2(y_16p.add(0x16));
        let ij_1 = S::getpz2(y_16p.add(0x18));
        let kl_1 = S::getpz2(y_16p.add(0x1a));
        let mn_1 = S::getpz2(y_16p.add(0x1c));
        let op_1 = S::getpz2(y_16p.add(0x1e));

        let y0 = S::catlo(ab_0, ab_1);
        let y1 = S::mulpz2(w1p, S::cathi(ab_0, ab_1));
        let y2 = S::mulpz2(w2p, S::catlo(cd_0, cd_1));
        let y3 = S::mulpz2(w3p, S::cathi(cd_0, cd_1));
        let y4 = S::mulpz2(w4p, S::catlo(ef_0, ef_1));
        let y5 = S::mulpz2(w5p, S::cathi(ef_0, ef_1));
        let y6 = S::mulpz2(w6p, S::catlo(gh_0, gh_1));
        let y7 = S::mulpz2(w7p, S::cathi(gh_0, gh_1));

        let y8 = S::mulpz2(w8p, S::catlo(ij_0, ij_1));
        let y9 = S::mulpz2(w9p, S::cathi(ij_0, ij_1));
        let ya = S::mulpz2(wap, S::catlo(kl_0, kl_1));
        let yb = S::mulpz2(wbp, S::cathi(kl_0, kl_1));
        let yc = S::mulpz2(wcp, S::catlo(mn_0, mn_1));
        let yd = S::mulpz2(wdp, S::cathi(mn_0, mn_1));
        let ye = S::mulpz2(wep, S::catlo(op_0, op_1));
        let yf = S::mulpz2(wfp, S::cathi(op_0, op_1));

        let a08 = S::addpz2(y0, y8);
        let s08 = S::subpz2(y0, y8);
        let a4c = S::addpz2(y4, yc);
        let s4c = S::subpz2(y4, yc);
        let a2a = S::addpz2(y2, ya);
        let s2a = S::subpz2(y2, ya);
        let a6e = S::addpz2(y6, ye);
        let s6e = S::subpz2(y6, ye);
        let a19 = S::addpz2(y1, y9);
        let s19 = S::subpz2(y1, y9);
        let a5d = S::addpz2(y5, yd);
        let s5d = S::subpz2(y5, yd);
        let a3b = S::addpz2(y3, yb);
        let s3b = S::subpz2(y3, yb);
        let a7f = S::addpz2(y7, yf);
        let s7f = S::subpz2(y7, yf);

        let js4c = S::jxpz2(s4c);
        let js6e = S::jxpz2(s6e);
        let js5d = S::jxpz2(s5d);
        let js7f = S::jxpz2(s7f);

        let a08p1a4c = S::addpz2(a08, a4c);
        let s08mjs4c = S::subpz2(s08, js4c);
        let a08m1a4c = S::subpz2(a08, a4c);
        let s08pjs4c = S::addpz2(s08, js4c);
        let a2ap1a6e = S::addpz2(a2a, a6e);
        let s2amjs6e = S::subpz2(s2a, js6e);
        let a2am1a6e = S::subpz2(a2a, a6e);
        let s2apjs6e = S::addpz2(s2a, js6e);
        let a19p1a5d = S::addpz2(a19, a5d);
        let s19mjs5d = S::subpz2(s19, js5d);
        let a19m1a5d = S::subpz2(a19, a5d);
        let s19pjs5d = S::addpz2(s19, js5d);
        let a3bp1a7f = S::addpz2(a3b, a7f);
        let s3bmjs7f = S::subpz2(s3b, js7f);
        let a3bm1a7f = S::subpz2(a3b, a7f);
        let s3bpjs7f = S::addpz2(s3b, js7f);

        let w8_s2amjs6e = S::w8xpz2(s2amjs6e);
        let j_a2am1a6e = S::jxpz2(a2am1a6e);
        let v8_s2apjs6e = S::v8xpz2(s2apjs6e);

        let a08p1a4c_p1_a2ap1a6e = S::addpz2(a08p1a4c, a2ap1a6e);
        let s08mjs4c_pw_s2amjs6e = S::addpz2(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_mj_a2am1a6e = S::subpz2(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_mv_s2apjs6e = S::subpz2(s08pjs4c, v8_s2apjs6e);
        let a08p1a4c_m1_a2ap1a6e = S::subpz2(a08p1a4c, a2ap1a6e);
        let s08mjs4c_mw_s2amjs6e = S::subpz2(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_pj_a2am1a6e = S::addpz2(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_pv_s2apjs6e = S::addpz2(s08pjs4c, v8_s2apjs6e);

        let w8_s3bmjs7f = S::w8xpz2(s3bmjs7f);
        let j_a3bm1a7f = S::jxpz2(a3bm1a7f);
        let v8_s3bpjs7f = S::v8xpz2(s3bpjs7f);

        let a19p1a5d_p1_a3bp1a7f = S::addpz2(a19p1a5d, a3bp1a7f);
        let s19mjs5d_pw_s3bmjs7f = S::addpz2(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_mj_a3bm1a7f = S::subpz2(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_mv_s3bpjs7f = S::subpz2(s19pjs5d, v8_s3bpjs7f);
        let a19p1a5d_m1_a3bp1a7f = S::subpz2(a19p1a5d, a3bp1a7f);
        let s19mjs5d_mw_s3bmjs7f = S::subpz2(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_pj_a3bm1a7f = S::addpz2(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_pv_s3bpjs7f = S::addpz2(s19pjs5d, v8_s3bpjs7f);

        S::setpz2(
            x_p.add(big_n0),
            S::addpz2(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );
        S::setpz2(
            x_p.add(big_n8),
            S::subpz2(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );

        let hf_s19pjs5d_pv_s3bpjs7f = S::hfxpz2(s19pjs5d_pv_s3bpjs7f);
        S::setpz2(
            x_p.add(big_n1),
            S::addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );
        S::setpz2(
            x_p.add(big_n9),
            S::subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );

        let v8_a19m1a5d_pj_a3bm1a7f = S::v8xpz2(a19m1a5d_pj_a3bm1a7f);
        S::setpz2(
            x_p.add(big_n2),
            S::addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );
        S::setpz2(
            x_p.add(big_na),
            S::subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );

        let hd_s19mjs5d_mw_s3bmjs7f = S::hdxpz2(s19mjs5d_mw_s3bmjs7f);
        S::setpz2(
            x_p.add(big_n3),
            S::addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );
        S::setpz2(
            x_p.add(big_nb),
            S::subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );

        let j_a19p1a5d_m1_a3bp1a7f = S::jxpz2(a19p1a5d_m1_a3bp1a7f);
        S::setpz2(
            x_p.add(big_n4),
            S::addpz2(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );
        S::setpz2(
            x_p.add(big_nc),
            S::subpz2(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );

        let h3_s19pjs5d_mv_s3bpjs7f = S::h3xpz2(s19pjs5d_mv_s3bpjs7f);
        S::setpz2(
            x_p.add(big_n5),
            S::subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );
        S::setpz2(
            x_p.add(big_nd),
            S::addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );

        let w8_a19m1a5d_mj_a3bm1a7f = S::w8xpz2(a19m1a5d_mj_a3bm1a7f);
        S::setpz2(
            x_p.add(big_n6),
            S::subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );
        S::setpz2(
            x_p.add(big_ne),
            S::addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );

        let h1_s19mjs5d_pw_s3bmjs7f = S::h1xpz2(s19mjs5d_pw_s3bmjs7f);
        S::setpz2(
            x_p.add(big_n7),
            S::subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );
        S::setpz2(
            x_p.add(big_nf),
            S::addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );

        p += 2;
    }
}

#[inline(always)]
pub(crate) unsafe fn fwdend_16_s<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 16);
    debug_assert_ne!(s, 1);

    let z = if eo { y } else { x };

    let mut q = 0;
    while q < s {
        let xq = x.add(q);
        let zq = z.add(q);

        let z0 = S::getpz2(zq.add(0x0));
        let z1 = S::getpz2(zq.add(s));
        let z2 = S::getpz2(zq.add(s * 0x2));
        let z3 = S::getpz2(zq.add(s * 0x3));
        let z4 = S::getpz2(zq.add(s * 0x4));
        let z5 = S::getpz2(zq.add(s * 0x5));
        let z6 = S::getpz2(zq.add(s * 0x6));
        let z7 = S::getpz2(zq.add(s * 0x7));
        let z8 = S::getpz2(zq.add(s * 0x8));
        let z9 = S::getpz2(zq.add(s * 0x9));
        let za = S::getpz2(zq.add(s * 0xa));
        let zb = S::getpz2(zq.add(s * 0xb));
        let zc = S::getpz2(zq.add(s * 0xc));
        let zd = S::getpz2(zq.add(s * 0xd));
        let ze = S::getpz2(zq.add(s * 0xe));
        let zf = S::getpz2(zq.add(s * 0xf));

        let a08 = S::addpz2(z0, z8);
        let s08 = S::subpz2(z0, z8);
        let a4c = S::addpz2(z4, zc);
        let s4c = S::subpz2(z4, zc);
        let a2a = S::addpz2(z2, za);
        let s2a = S::subpz2(z2, za);
        let a6e = S::addpz2(z6, ze);
        let s6e = S::subpz2(z6, ze);
        let a19 = S::addpz2(z1, z9);
        let s19 = S::subpz2(z1, z9);
        let a5d = S::addpz2(z5, zd);
        let s5d = S::subpz2(z5, zd);
        let a3b = S::addpz2(z3, zb);
        let s3b = S::subpz2(z3, zb);
        let a7f = S::addpz2(z7, zf);
        let s7f = S::subpz2(z7, zf);

        let js4c = S::jxpz2(s4c);
        let js6e = S::jxpz2(s6e);
        let js5d = S::jxpz2(s5d);
        let js7f = S::jxpz2(s7f);

        let a08p1a4c = S::addpz2(a08, a4c);
        let s08mjs4c = S::subpz2(s08, js4c);
        let a08m1a4c = S::subpz2(a08, a4c);
        let s08pjs4c = S::addpz2(s08, js4c);
        let a2ap1a6e = S::addpz2(a2a, a6e);
        let s2amjs6e = S::subpz2(s2a, js6e);
        let a2am1a6e = S::subpz2(a2a, a6e);
        let s2apjs6e = S::addpz2(s2a, js6e);
        let a19p1a5d = S::addpz2(a19, a5d);
        let s19mjs5d = S::subpz2(s19, js5d);
        let a19m1a5d = S::subpz2(a19, a5d);
        let s19pjs5d = S::addpz2(s19, js5d);
        let a3bp1a7f = S::addpz2(a3b, a7f);
        let s3bmjs7f = S::subpz2(s3b, js7f);
        let a3bm1a7f = S::subpz2(a3b, a7f);
        let s3bpjs7f = S::addpz2(s3b, js7f);

        let w8_s2amjs6e = S::w8xpz2(s2amjs6e);
        let j_a2am1a6e = S::jxpz2(a2am1a6e);
        let v8_s2apjs6e = S::v8xpz2(s2apjs6e);

        let a08p1a4c_p1_a2ap1a6e = S::addpz2(a08p1a4c, a2ap1a6e);
        let s08mjs4c_pw_s2amjs6e = S::addpz2(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_mj_a2am1a6e = S::subpz2(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_mv_s2apjs6e = S::subpz2(s08pjs4c, v8_s2apjs6e);
        let a08p1a4c_m1_a2ap1a6e = S::subpz2(a08p1a4c, a2ap1a6e);
        let s08mjs4c_mw_s2amjs6e = S::subpz2(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_pj_a2am1a6e = S::addpz2(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_pv_s2apjs6e = S::addpz2(s08pjs4c, v8_s2apjs6e);

        let w8_s3bmjs7f = S::w8xpz2(s3bmjs7f);
        let j_a3bm1a7f = S::jxpz2(a3bm1a7f);
        let v8_s3bpjs7f = S::v8xpz2(s3bpjs7f);

        let a19p1a5d_p1_a3bp1a7f = S::addpz2(a19p1a5d, a3bp1a7f);
        let s19mjs5d_pw_s3bmjs7f = S::addpz2(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_mj_a3bm1a7f = S::subpz2(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_mv_s3bpjs7f = S::subpz2(s19pjs5d, v8_s3bpjs7f);
        let a19p1a5d_m1_a3bp1a7f = S::subpz2(a19p1a5d, a3bp1a7f);
        let s19mjs5d_mw_s3bmjs7f = S::subpz2(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_pj_a3bm1a7f = S::addpz2(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_pv_s3bpjs7f = S::addpz2(s19pjs5d, v8_s3bpjs7f);

        let h1_s19mjs5d_pw_s3bmjs7f = S::h1xpz2(s19mjs5d_pw_s3bmjs7f);
        let w8_a19m1a5d_mj_a3bm1a7f = S::w8xpz2(a19m1a5d_mj_a3bm1a7f);
        let h3_s19pjs5d_mv_s3bpjs7f = S::h3xpz2(s19pjs5d_mv_s3bpjs7f);
        let j_a19p1a5d_m1_a3bp1a7f = S::jxpz2(a19p1a5d_m1_a3bp1a7f);
        let hd_s19mjs5d_mw_s3bmjs7f = S::hdxpz2(s19mjs5d_mw_s3bmjs7f);
        let v8_a19m1a5d_pj_a3bm1a7f = S::v8xpz2(a19m1a5d_pj_a3bm1a7f);
        let hf_s19pjs5d_pv_s3bpjs7f = S::hfxpz2(s19pjs5d_pv_s3bpjs7f);

        S::setpz2(
            xq.add(0x0),
            S::addpz2(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );
        S::setpz2(
            xq.add(s),
            S::addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );
        S::setpz2(
            xq.add(s * 0x2),
            S::addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );
        S::setpz2(
            xq.add(s * 0x3),
            S::addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );
        S::setpz2(
            xq.add(s * 0x4),
            S::subpz2(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );
        S::setpz2(
            xq.add(s * 0x5),
            S::subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );
        S::setpz2(
            xq.add(s * 0x6),
            S::subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );
        S::setpz2(
            xq.add(s * 0x7),
            S::subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );

        S::setpz2(
            xq.add(s * 0x8),
            S::subpz2(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );
        S::setpz2(
            xq.add(s * 0x9),
            S::subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );
        S::setpz2(
            xq.add(s * 0xa),
            S::subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );
        S::setpz2(
            xq.add(s * 0xb),
            S::subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );
        S::setpz2(
            xq.add(s * 0xc),
            S::addpz2(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );
        S::setpz2(
            xq.add(s * 0xd),
            S::addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );
        S::setpz2(
            xq.add(s * 0xe),
            S::addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );
        S::setpz2(
            xq.add(s * 0xf),
            S::addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );

        q += 2;
    }
}

#[inline(always)]
pub(crate) unsafe fn fwdend_16_1<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 16);
    debug_assert_eq!(s, 1);

    let z = if eo { y } else { x };

    let z0 = S::getpz(&*z.add(0x0));
    let z1 = S::getpz(&*z.add(0x1));
    let z2 = S::getpz(&*z.add(0x2));
    let z3 = S::getpz(&*z.add(0x3));
    let z4 = S::getpz(&*z.add(0x4));
    let z5 = S::getpz(&*z.add(0x5));
    let z6 = S::getpz(&*z.add(0x6));
    let z7 = S::getpz(&*z.add(0x7));
    let z8 = S::getpz(&*z.add(0x8));
    let z9 = S::getpz(&*z.add(0x9));
    let za = S::getpz(&*z.add(0xa));
    let zb = S::getpz(&*z.add(0xb));
    let zc = S::getpz(&*z.add(0xc));
    let zd = S::getpz(&*z.add(0xd));
    let ze = S::getpz(&*z.add(0xe));
    let zf = S::getpz(&*z.add(0xf));

    let a08 = S::addpz(z0, z8);
    let s08 = S::subpz(z0, z8);
    let a4c = S::addpz(z4, zc);
    let s4c = S::subpz(z4, zc);
    let a2a = S::addpz(z2, za);
    let s2a = S::subpz(z2, za);
    let a6e = S::addpz(z6, ze);
    let s6e = S::subpz(z6, ze);
    let a19 = S::addpz(z1, z9);
    let s19 = S::subpz(z1, z9);
    let a5d = S::addpz(z5, zd);
    let s5d = S::subpz(z5, zd);
    let a3b = S::addpz(z3, zb);
    let s3b = S::subpz(z3, zb);
    let a7f = S::addpz(z7, zf);
    let s7f = S::subpz(z7, zf);

    let js4c = S::jxpz(s4c);
    let js6e = S::jxpz(s6e);
    let js5d = S::jxpz(s5d);
    let js7f = S::jxpz(s7f);

    let a08p1a4c = S::addpz(a08, a4c);
    let s08mjs4c = S::subpz(s08, js4c);
    let a08m1a4c = S::subpz(a08, a4c);
    let s08pjs4c = S::addpz(s08, js4c);
    let a2ap1a6e = S::addpz(a2a, a6e);
    let s2amjs6e = S::subpz(s2a, js6e);
    let a2am1a6e = S::subpz(a2a, a6e);
    let s2apjs6e = S::addpz(s2a, js6e);
    let a19p1a5d = S::addpz(a19, a5d);
    let s19mjs5d = S::subpz(s19, js5d);
    let a19m1a5d = S::subpz(a19, a5d);
    let s19pjs5d = S::addpz(s19, js5d);
    let a3bp1a7f = S::addpz(a3b, a7f);
    let s3bmjs7f = S::subpz(s3b, js7f);
    let a3bm1a7f = S::subpz(a3b, a7f);
    let s3bpjs7f = S::addpz(s3b, js7f);

    let w8_s2amjs6e = S::w8xpz(s2amjs6e);
    let j_a2am1a6e = S::jxpz(a2am1a6e);
    let v8_s2apjs6e = S::v8xpz(s2apjs6e);

    let a08p1a4c_p1_a2ap1a6e = S::addpz(a08p1a4c, a2ap1a6e);
    let s08mjs4c_pw_s2amjs6e = S::addpz(s08mjs4c, w8_s2amjs6e);
    let a08m1a4c_mj_a2am1a6e = S::subpz(a08m1a4c, j_a2am1a6e);
    let s08pjs4c_mv_s2apjs6e = S::subpz(s08pjs4c, v8_s2apjs6e);
    let a08p1a4c_m1_a2ap1a6e = S::subpz(a08p1a4c, a2ap1a6e);
    let s08mjs4c_mw_s2amjs6e = S::subpz(s08mjs4c, w8_s2amjs6e);
    let a08m1a4c_pj_a2am1a6e = S::addpz(a08m1a4c, j_a2am1a6e);
    let s08pjs4c_pv_s2apjs6e = S::addpz(s08pjs4c, v8_s2apjs6e);

    let w8_s3bmjs7f = S::w8xpz(s3bmjs7f);
    let j_a3bm1a7f = S::jxpz(a3bm1a7f);
    let v8_s3bpjs7f = S::v8xpz(s3bpjs7f);

    let a19p1a5d_p1_a3bp1a7f = S::addpz(a19p1a5d, a3bp1a7f);
    let s19mjs5d_pw_s3bmjs7f = S::addpz(s19mjs5d, w8_s3bmjs7f);
    let a19m1a5d_mj_a3bm1a7f = S::subpz(a19m1a5d, j_a3bm1a7f);
    let s19pjs5d_mv_s3bpjs7f = S::subpz(s19pjs5d, v8_s3bpjs7f);
    let a19p1a5d_m1_a3bp1a7f = S::subpz(a19p1a5d, a3bp1a7f);
    let s19mjs5d_mw_s3bmjs7f = S::subpz(s19mjs5d, w8_s3bmjs7f);
    let a19m1a5d_pj_a3bm1a7f = S::addpz(a19m1a5d, j_a3bm1a7f);
    let s19pjs5d_pv_s3bpjs7f = S::addpz(s19pjs5d, v8_s3bpjs7f);

    let h1_s19mjs5d_pw_s3bmjs7f = S::h1xpz(s19mjs5d_pw_s3bmjs7f);
    let w8_a19m1a5d_mj_a3bm1a7f = S::w8xpz(a19m1a5d_mj_a3bm1a7f);
    let h3_s19pjs5d_mv_s3bpjs7f = S::h3xpz(s19pjs5d_mv_s3bpjs7f);
    let j_a19p1a5d_m1_a3bp1a7f = S::jxpz(a19p1a5d_m1_a3bp1a7f);
    let hd_s19mjs5d_mw_s3bmjs7f = S::hdxpz(s19mjs5d_mw_s3bmjs7f);
    let v8_a19m1a5d_pj_a3bm1a7f = S::v8xpz(a19m1a5d_pj_a3bm1a7f);
    let hf_s19pjs5d_pv_s3bpjs7f = S::hfxpz(s19pjs5d_pv_s3bpjs7f);

    S::setpz(
        x.add(0x0),
        S::addpz(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
    );
    S::setpz(
        x.add(0x1),
        S::addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
    );
    S::setpz(
        x.add(0x2),
        S::addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
    );
    S::setpz(
        x.add(0x3),
        S::addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
    );
    S::setpz(
        x.add(0x4),
        S::subpz(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
    );
    S::setpz(
        x.add(0x5),
        S::subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
    );
    S::setpz(
        x.add(0x6),
        S::subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
    );
    S::setpz(
        x.add(0x7),
        S::subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
    );

    S::setpz(
        x.add(0x8),
        S::subpz(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
    );
    S::setpz(
        x.add(0x9),
        S::subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
    );
    S::setpz(
        x.add(0xa),
        S::subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
    );
    S::setpz(
        x.add(0xb),
        S::subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
    );
    S::setpz(
        x.add(0xc),
        S::addpz(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
    );
    S::setpz(
        x.add(0xd),
        S::addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
    );
    S::setpz(
        x.add(0xe),
        S::addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
    );
    S::setpz(
        x.add(0xf),
        S::addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
    );
}

#[inline(always)]
pub(crate) unsafe fn invend_16_s<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 16);
    debug_assert_ne!(s, 1);

    let z = if eo { y } else { x };

    let mut q = 0;
    while q < s {
        let xq = x.add(q);
        let zq = z.add(q);

        let z0 = S::getpz2(zq.add(0x0));
        let z1 = S::getpz2(zq.add(s));
        let z2 = S::getpz2(zq.add(s * 0x2));
        let z3 = S::getpz2(zq.add(s * 0x3));
        let z4 = S::getpz2(zq.add(s * 0x4));
        let z5 = S::getpz2(zq.add(s * 0x5));
        let z6 = S::getpz2(zq.add(s * 0x6));
        let z7 = S::getpz2(zq.add(s * 0x7));
        let z8 = S::getpz2(zq.add(s * 0x8));
        let z9 = S::getpz2(zq.add(s * 0x9));
        let za = S::getpz2(zq.add(s * 0xa));
        let zb = S::getpz2(zq.add(s * 0xb));
        let zc = S::getpz2(zq.add(s * 0xc));
        let zd = S::getpz2(zq.add(s * 0xd));
        let ze = S::getpz2(zq.add(s * 0xe));
        let zf = S::getpz2(zq.add(s * 0xf));

        let a08 = S::addpz2(z0, z8);
        let s08 = S::subpz2(z0, z8);
        let a4c = S::addpz2(z4, zc);
        let s4c = S::subpz2(z4, zc);
        let a2a = S::addpz2(z2, za);
        let s2a = S::subpz2(z2, za);
        let a6e = S::addpz2(z6, ze);
        let s6e = S::subpz2(z6, ze);
        let a19 = S::addpz2(z1, z9);
        let s19 = S::subpz2(z1, z9);
        let a5d = S::addpz2(z5, zd);
        let s5d = S::subpz2(z5, zd);
        let a3b = S::addpz2(z3, zb);
        let s3b = S::subpz2(z3, zb);
        let a7f = S::addpz2(z7, zf);
        let s7f = S::subpz2(z7, zf);

        let js4c = S::jxpz2(s4c);
        let js6e = S::jxpz2(s6e);
        let js5d = S::jxpz2(s5d);
        let js7f = S::jxpz2(s7f);

        let a08p1a4c = S::addpz2(a08, a4c);
        let s08mjs4c = S::subpz2(s08, js4c);
        let a08m1a4c = S::subpz2(a08, a4c);
        let s08pjs4c = S::addpz2(s08, js4c);
        let a2ap1a6e = S::addpz2(a2a, a6e);
        let s2amjs6e = S::subpz2(s2a, js6e);
        let a2am1a6e = S::subpz2(a2a, a6e);
        let s2apjs6e = S::addpz2(s2a, js6e);
        let a19p1a5d = S::addpz2(a19, a5d);
        let s19mjs5d = S::subpz2(s19, js5d);
        let a19m1a5d = S::subpz2(a19, a5d);
        let s19pjs5d = S::addpz2(s19, js5d);
        let a3bp1a7f = S::addpz2(a3b, a7f);
        let s3bmjs7f = S::subpz2(s3b, js7f);
        let a3bm1a7f = S::subpz2(a3b, a7f);
        let s3bpjs7f = S::addpz2(s3b, js7f);

        let w8_s2amjs6e = S::w8xpz2(s2amjs6e);
        let j_a2am1a6e = S::jxpz2(a2am1a6e);
        let v8_s2apjs6e = S::v8xpz2(s2apjs6e);

        let a08p1a4c_p1_a2ap1a6e = S::addpz2(a08p1a4c, a2ap1a6e);
        let s08mjs4c_pw_s2amjs6e = S::addpz2(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_mj_a2am1a6e = S::subpz2(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_mv_s2apjs6e = S::subpz2(s08pjs4c, v8_s2apjs6e);
        let a08p1a4c_m1_a2ap1a6e = S::subpz2(a08p1a4c, a2ap1a6e);
        let s08mjs4c_mw_s2amjs6e = S::subpz2(s08mjs4c, w8_s2amjs6e);
        let a08m1a4c_pj_a2am1a6e = S::addpz2(a08m1a4c, j_a2am1a6e);
        let s08pjs4c_pv_s2apjs6e = S::addpz2(s08pjs4c, v8_s2apjs6e);

        let w8_s3bmjs7f = S::w8xpz2(s3bmjs7f);
        let j_a3bm1a7f = S::jxpz2(a3bm1a7f);
        let v8_s3bpjs7f = S::v8xpz2(s3bpjs7f);

        let a19p1a5d_p1_a3bp1a7f = S::addpz2(a19p1a5d, a3bp1a7f);
        let s19mjs5d_pw_s3bmjs7f = S::addpz2(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_mj_a3bm1a7f = S::subpz2(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_mv_s3bpjs7f = S::subpz2(s19pjs5d, v8_s3bpjs7f);
        let a19p1a5d_m1_a3bp1a7f = S::subpz2(a19p1a5d, a3bp1a7f);
        let s19mjs5d_mw_s3bmjs7f = S::subpz2(s19mjs5d, w8_s3bmjs7f);
        let a19m1a5d_pj_a3bm1a7f = S::addpz2(a19m1a5d, j_a3bm1a7f);
        let s19pjs5d_pv_s3bpjs7f = S::addpz2(s19pjs5d, v8_s3bpjs7f);

        let h1_s19mjs5d_pw_s3bmjs7f = S::h1xpz2(s19mjs5d_pw_s3bmjs7f);
        let w8_a19m1a5d_mj_a3bm1a7f = S::w8xpz2(a19m1a5d_mj_a3bm1a7f);
        let h3_s19pjs5d_mv_s3bpjs7f = S::h3xpz2(s19pjs5d_mv_s3bpjs7f);
        let j_a19p1a5d_m1_a3bp1a7f = S::jxpz2(a19p1a5d_m1_a3bp1a7f);
        let hd_s19mjs5d_mw_s3bmjs7f = S::hdxpz2(s19mjs5d_mw_s3bmjs7f);
        let v8_a19m1a5d_pj_a3bm1a7f = S::v8xpz2(a19m1a5d_pj_a3bm1a7f);
        let hf_s19pjs5d_pv_s3bpjs7f = S::hfxpz2(s19pjs5d_pv_s3bpjs7f);

        S::setpz2(
            xq.add(0x0),
            S::addpz2(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );
        S::setpz2(
            xq.add(s),
            S::addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );
        S::setpz2(
            xq.add(s * 0x2),
            S::addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );
        S::setpz2(
            xq.add(s * 0x3),
            S::addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );
        S::setpz2(
            xq.add(s * 0x4),
            S::addpz2(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );
        S::setpz2(
            xq.add(s * 0x5),
            S::subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );
        S::setpz2(
            xq.add(s * 0x6),
            S::subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );
        S::setpz2(
            xq.add(s * 0x7),
            S::subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );

        S::setpz2(
            xq.add(s * 0x8),
            S::subpz2(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
        );
        S::setpz2(
            xq.add(s * 0x9),
            S::subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
        );
        S::setpz2(
            xq.add(s * 0xa),
            S::subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
        );
        S::setpz2(
            xq.add(s * 0xb),
            S::subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
        );
        S::setpz2(
            xq.add(s * 0xc),
            S::subpz2(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
        );
        S::setpz2(
            xq.add(s * 0xd),
            S::addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
        );
        S::setpz2(
            xq.add(s * 0xe),
            S::addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
        );
        S::setpz2(
            xq.add(s * 0xf),
            S::addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
        );

        q += 2;
    }
}

#[inline(always)]
pub(crate) unsafe fn invend_16_1<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 16);
    debug_assert_eq!(s, 1);

    let z = if eo { y } else { x };

    let z0 = S::getpz(&*z.add(0x0));
    let z1 = S::getpz(&*z.add(0x1));
    let z2 = S::getpz(&*z.add(0x2));
    let z3 = S::getpz(&*z.add(0x3));
    let z4 = S::getpz(&*z.add(0x4));
    let z5 = S::getpz(&*z.add(0x5));
    let z6 = S::getpz(&*z.add(0x6));
    let z7 = S::getpz(&*z.add(0x7));
    let z8 = S::getpz(&*z.add(0x8));
    let z9 = S::getpz(&*z.add(0x9));
    let za = S::getpz(&*z.add(0xa));
    let zb = S::getpz(&*z.add(0xb));
    let zc = S::getpz(&*z.add(0xc));
    let zd = S::getpz(&*z.add(0xd));
    let ze = S::getpz(&*z.add(0xe));
    let zf = S::getpz(&*z.add(0xf));

    let a08 = S::addpz(z0, z8);
    let s08 = S::subpz(z0, z8);
    let a4c = S::addpz(z4, zc);
    let s4c = S::subpz(z4, zc);
    let a2a = S::addpz(z2, za);
    let s2a = S::subpz(z2, za);
    let a6e = S::addpz(z6, ze);
    let s6e = S::subpz(z6, ze);
    let a19 = S::addpz(z1, z9);
    let s19 = S::subpz(z1, z9);
    let a5d = S::addpz(z5, zd);
    let s5d = S::subpz(z5, zd);
    let a3b = S::addpz(z3, zb);
    let s3b = S::subpz(z3, zb);
    let a7f = S::addpz(z7, zf);
    let s7f = S::subpz(z7, zf);

    let js4c = S::jxpz(s4c);
    let js6e = S::jxpz(s6e);
    let js5d = S::jxpz(s5d);
    let js7f = S::jxpz(s7f);

    let a08p1a4c = S::addpz(a08, a4c);
    let s08mjs4c = S::subpz(s08, js4c);
    let a08m1a4c = S::subpz(a08, a4c);
    let s08pjs4c = S::addpz(s08, js4c);
    let a2ap1a6e = S::addpz(a2a, a6e);
    let s2amjs6e = S::subpz(s2a, js6e);
    let a2am1a6e = S::subpz(a2a, a6e);
    let s2apjs6e = S::addpz(s2a, js6e);
    let a19p1a5d = S::addpz(a19, a5d);
    let s19mjs5d = S::subpz(s19, js5d);
    let a19m1a5d = S::subpz(a19, a5d);
    let s19pjs5d = S::addpz(s19, js5d);
    let a3bp1a7f = S::addpz(a3b, a7f);
    let s3bmjs7f = S::subpz(s3b, js7f);
    let a3bm1a7f = S::subpz(a3b, a7f);
    let s3bpjs7f = S::addpz(s3b, js7f);

    let w8_s2amjs6e = S::w8xpz(s2amjs6e);
    let j_a2am1a6e = S::jxpz(a2am1a6e);
    let v8_s2apjs6e = S::v8xpz(s2apjs6e);

    let a08p1a4c_p1_a2ap1a6e = S::addpz(a08p1a4c, a2ap1a6e);
    let s08mjs4c_pw_s2amjs6e = S::addpz(s08mjs4c, w8_s2amjs6e);
    let a08m1a4c_mj_a2am1a6e = S::subpz(a08m1a4c, j_a2am1a6e);
    let s08pjs4c_mv_s2apjs6e = S::subpz(s08pjs4c, v8_s2apjs6e);
    let a08p1a4c_m1_a2ap1a6e = S::subpz(a08p1a4c, a2ap1a6e);
    let s08mjs4c_mw_s2amjs6e = S::subpz(s08mjs4c, w8_s2amjs6e);
    let a08m1a4c_pj_a2am1a6e = S::addpz(a08m1a4c, j_a2am1a6e);
    let s08pjs4c_pv_s2apjs6e = S::addpz(s08pjs4c, v8_s2apjs6e);

    let w8_s3bmjs7f = S::w8xpz(s3bmjs7f);
    let j_a3bm1a7f = S::jxpz(a3bm1a7f);
    let v8_s3bpjs7f = S::v8xpz(s3bpjs7f);

    let a19p1a5d_p1_a3bp1a7f = S::addpz(a19p1a5d, a3bp1a7f);
    let s19mjs5d_pw_s3bmjs7f = S::addpz(s19mjs5d, w8_s3bmjs7f);
    let a19m1a5d_mj_a3bm1a7f = S::subpz(a19m1a5d, j_a3bm1a7f);
    let s19pjs5d_mv_s3bpjs7f = S::subpz(s19pjs5d, v8_s3bpjs7f);
    let a19p1a5d_m1_a3bp1a7f = S::subpz(a19p1a5d, a3bp1a7f);
    let s19mjs5d_mw_s3bmjs7f = S::subpz(s19mjs5d, w8_s3bmjs7f);
    let a19m1a5d_pj_a3bm1a7f = S::addpz(a19m1a5d, j_a3bm1a7f);
    let s19pjs5d_pv_s3bpjs7f = S::addpz(s19pjs5d, v8_s3bpjs7f);

    let h1_s19mjs5d_pw_s3bmjs7f = S::h1xpz(s19mjs5d_pw_s3bmjs7f);
    let w8_a19m1a5d_mj_a3bm1a7f = S::w8xpz(a19m1a5d_mj_a3bm1a7f);
    let h3_s19pjs5d_mv_s3bpjs7f = S::h3xpz(s19pjs5d_mv_s3bpjs7f);
    let j_a19p1a5d_m1_a3bp1a7f = S::jxpz(a19p1a5d_m1_a3bp1a7f);
    let hd_s19mjs5d_mw_s3bmjs7f = S::hdxpz(s19mjs5d_mw_s3bmjs7f);
    let v8_a19m1a5d_pj_a3bm1a7f = S::v8xpz(a19m1a5d_pj_a3bm1a7f);
    let hf_s19pjs5d_pv_s3bpjs7f = S::hfxpz(s19pjs5d_pv_s3bpjs7f);

    S::setpz(
        x.add(0x0),
        S::addpz(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
    );
    S::setpz(
        x.add(0x1),
        S::addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
    );
    S::setpz(
        x.add(0x2),
        S::addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
    );
    S::setpz(
        x.add(0x3),
        S::addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
    );
    S::setpz(
        x.add(0x4),
        S::addpz(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
    );
    S::setpz(
        x.add(0x5),
        S::subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
    );
    S::setpz(
        x.add(0x6),
        S::subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
    );
    S::setpz(
        x.add(0x7),
        S::subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
    );

    S::setpz(
        x.add(0x8),
        S::subpz(a08p1a4c_p1_a2ap1a6e, a19p1a5d_p1_a3bp1a7f),
    );
    S::setpz(
        x.add(0x9),
        S::subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f),
    );
    S::setpz(
        x.add(0xa),
        S::subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f),
    );
    S::setpz(
        x.add(0xb),
        S::subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f),
    );
    S::setpz(
        x.add(0xc),
        S::subpz(a08p1a4c_m1_a2ap1a6e, j_a19p1a5d_m1_a3bp1a7f),
    );
    S::setpz(
        x.add(0xd),
        S::addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f),
    );
    S::setpz(
        x.add(0xe),
        S::addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f),
    );
    S::setpz(
        x.add(0xf),
        S::addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f),
    );
}

include!(concat!(env!("OUT_DIR"), "/dit16.rs"));

/// Initialize twiddles for subsequent forward and inverse Fourier transforms of size `n`.
/// `twiddles` must be of length `2*n`.
pub fn init_twiddles(n: usize, twiddles: &mut [c64]) {
    crate::dif16::init_twiddles(n, twiddles);
}

impl_main_fn!(fwd, &*FWD_FN_ARRAY);
impl_main_fn!(inv, &*INV_FN_ARRAY);

impl_main_fn!(
    #[cfg(target_feature = "fma")]
    fwd_fma,
    fwd_fn_array_fma()
);
impl_main_fn!(
    #[cfg(target_feature = "fma")]
    inv_fma,
    inv_fn_array_fma()
);

impl_main_fn!(
    #[cfg(target_feature = "avx")]
    fwd_avx,
    fwd_fn_array_avx()
);
impl_main_fn!(
    #[cfg(target_feature = "avx")]
    inv_avx,
    inv_fn_array_avx()
);

impl_main_fn!(
    #[cfg(target_feature = "sse3")]
    fwd_sse3,
    fwd_fn_array_sse3()
);
impl_main_fn!(
    #[cfg(target_feature = "sse3")]
    inv_sse3,
    inv_fn_array_sse3()
);

impl_main_fn!(
    #[cfg(target_feature = "sse2")]
    fwd_sse2,
    fwd_fn_array_sse2()
);
impl_main_fn!(
    #[cfg(target_feature = "sse2")]
    inv_sse2,
    inv_fn_array_sse2()
);

impl_main_fn!(
    #[cfg(target_feature = "neon")]
    fwd_neon,
    fwd_fn_array_neon()
);
impl_main_fn!(
    #[cfg(target_feature = "neon")]
    inv_neon,
    inv_fn_array_neon()
);

impl_main_fn!(
    #[cfg(target_feature = "simd128")]
    fwd_simd128,
    fwd_fn_array_simd128()
);
impl_main_fn!(
    #[cfg(target_feature = "simd128")]
    inv_simd128,
    inv_fn_array_simd128()
);

impl_main_fn!(fwd_scalar, fwd_fn_array_scalar());
impl_main_fn!(inv_scalar, inv_fn_array_scalar());

#[cfg(test)]
mod tests {
    use super::*;
    use dyn_stack::ReborrowMut;

    fn test_fft_generic(
        n: usize,
        fwd_fn: fn(&mut [c64], &[c64], DynStack),
        inv_fn: fn(&mut [c64], &[c64], DynStack),
    ) {
        let z = c64::new(0.0, 0.0);

        let mut arr_fwd = vec![z; n];

        for z in &mut arr_fwd {
            z.re = rand::random();
            z.im = rand::random();
        }

        for (i, z) in arr_fwd.iter_mut().enumerate() {
            z.re = i as f64;
            z.im = i as f64;
        }

        let mut arr_inv = arr_fwd.clone();
        let mut arr_fwd_expected = arr_fwd.clone();
        let mut arr_inv_expected = arr_fwd.clone();

        let mut w = vec![z; 2 * n];

        init_twiddles(n, &mut w);
        let mut mem = dyn_stack::uninit_mem_in_global(crate::fft_scratch(n).unwrap());
        let mut stack = DynStack::new(&mut mem);

        fwd_fn(&mut arr_fwd, &w, stack.rb_mut());
        inv_fn(&mut arr_inv, &w, stack);

        #[cfg(not(miri))]
        {
            let mut fft = rustfft::FftPlanner::new();
            let fwd = fft.plan_fft_forward(n);
            let inv = fft.plan_fft_inverse(n);

            fwd.process(&mut arr_fwd_expected);
            inv.process(&mut arr_inv_expected);

            use num_complex::ComplexFloat;
            for (actual, expected) in arr_fwd.iter().zip(&arr_fwd_expected) {
                assert!((*actual - *expected).abs() < 1e-6);
            }
            for (actual, expected) in arr_inv.iter().zip(&arr_inv_expected) {
                assert!((*actual - *expected).abs() < 1e-6);
            }
        }
    }

    fn test_roundtrip_generic(
        n: usize,
        fwd_fn: fn(&mut [c64], &[c64], DynStack),
        inv_fn: fn(&mut [c64], &[c64], DynStack),
    ) {
        let z = c64::new(0.0, 0.0);

        let mut arr_orig = vec![z; n];

        for z in &mut arr_orig {
            z.re = rand::random();
            z.im = rand::random();
        }

        let mut arr_roundtrip = arr_orig.clone();

        let mut w = vec![z; 2 * n];

        init_twiddles(n, &mut w);
        let mut mem = dyn_stack::uninit_mem_in_global(crate::fft_scratch(n).unwrap());
        let mut stack = DynStack::new(&mut mem);

        fwd_fn(&mut arr_roundtrip, &w, stack.rb_mut());
        inv_fn(&mut arr_roundtrip, &w, stack);

        for z in &mut arr_roundtrip {
            *z /= n as f64;
        }

        use num_complex::ComplexFloat;
        for (actual, expected) in arr_roundtrip.iter().zip(&arr_orig) {
            assert!((*actual - *expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_fft() {
        for i in 0..crate::MAX_EXP {
            let n = 1usize << i;
            test_fft_generic(n, fwd_scalar, inv_scalar);

            #[cfg(not(miri))]
            {
                #[cfg(target_feature = "fma")]
                test_fft_generic(n, fwd_fma, inv_fma);
                #[cfg(target_feature = "avx")]
                test_fft_generic(n, fwd_avx, inv_avx);
                #[cfg(target_feature = "sse3")]
                test_fft_generic(n, fwd_sse3, inv_sse3);
                #[cfg(target_feature = "sse2")]
                test_fft_generic(n, fwd_sse2, inv_sse2);

                #[cfg(target_feature = "neon")]
                test_fft_generic(n, fwd_neon, inv_neon);

                #[cfg(target_feature = "simd128")]
                test_fft_generic(n, fwd_simd128, inv_simd128);
            }
        }
    }

    #[test]
    fn test_roundtrip() {
        for i in 0..crate::MAX_EXP {
            let n = 1usize << i;
            test_roundtrip_generic(n, fwd_scalar, inv_scalar);

            #[cfg(not(miri))]
            {
                #[cfg(target_feature = "fma")]
                test_roundtrip_generic(n, fwd_fma, inv_fma);
                #[cfg(target_feature = "avx")]
                test_roundtrip_generic(n, fwd_avx, inv_avx);
                #[cfg(target_feature = "sse3")]
                test_roundtrip_generic(n, fwd_sse3, inv_sse3);
                #[cfg(target_feature = "sse2")]
                test_roundtrip_generic(n, fwd_sse2, inv_sse2);

                #[cfg(target_feature = "neon")]
                test_roundtrip_generic(n, fwd_neon, inv_neon);

                #[cfg(target_feature = "simd128")]
                test_roundtrip_generic(n, fwd_simd128, inv_simd128);
            }
        }
    }
}
