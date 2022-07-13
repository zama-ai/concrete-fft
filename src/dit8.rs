use crate::c64;
use crate::fft_simd::FftSimd16;
use crate::twiddles::{twid, twid_t};

pub(crate) use crate::dit4::fwdend_2_1;
pub(crate) use crate::dit4::fwdend_2_s;
pub(crate) use crate::dit4::fwdend_4_1;
pub(crate) use crate::dit4::fwdend_4_s;
pub(crate) use crate::dit4::invend_2_1;
pub(crate) use crate::dit4::invend_2_s;
pub(crate) use crate::dit4::invend_4_1;
pub(crate) use crate::dit4::invend_4_s;

// forward butterfly
#[inline(always)]
unsafe fn fwdcore_s<S: FftSimd16>(n: usize, s: usize, x: *mut c64, y: *mut c64, w: *const c64) {
    debug_assert_ne!(s, 1);

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

        let w1p = S::duppz5(&*twid_t(8, big_n, 1, w, sp));
        let w2p = S::duppz5(&*twid_t(8, big_n, 2, w, sp));
        let w3p = S::duppz5(&*twid_t(8, big_n, 3, w, sp));
        let w4p = S::duppz5(&*twid_t(8, big_n, 4, w, sp));
        let w5p = S::duppz5(&*twid_t(8, big_n, 5, w, sp));
        let w6p = S::duppz5(&*twid_t(8, big_n, 6, w, sp));
        let w7p = S::duppz5(&*twid_t(8, big_n, 7, w, sp));

        let mut q = 0;
        while q < s {
            let xq_sp = x.add(q + sp);
            let yq_s8p = y.add(q + s8p);

            let y0 = S::getpz4(yq_s8p.add(0));
            let y1 = S::mulpz4(w1p, S::getpz4(yq_s8p.add(s)));
            let y2 = S::mulpz4(w2p, S::getpz4(yq_s8p.add(s * 2)));
            let y3 = S::mulpz4(w3p, S::getpz4(yq_s8p.add(s * 3)));
            let y4 = S::mulpz4(w4p, S::getpz4(yq_s8p.add(s * 4)));
            let y5 = S::mulpz4(w5p, S::getpz4(yq_s8p.add(s * 5)));
            let y6 = S::mulpz4(w6p, S::getpz4(yq_s8p.add(s * 6)));
            let y7 = S::mulpz4(w7p, S::getpz4(yq_s8p.add(s * 7)));
            let a04 = S::addpz4(y0, y4);
            let s04 = S::subpz4(y0, y4);
            let a26 = S::addpz4(y2, y6);
            let js26 = S::jxpz4(S::subpz4(y2, y6));
            let a15 = S::addpz4(y1, y5);
            let s15 = S::subpz4(y1, y5);
            let a37 = S::addpz4(y3, y7);
            let js37 = S::jxpz4(S::subpz4(y3, y7));

            let a04_p1_a26 = S::addpz4(a04, a26);
            let a15_p1_a37 = S::addpz4(a15, a37);
            S::setpz4(xq_sp.add(big_n0), S::addpz4(a04_p1_a26, a15_p1_a37));
            S::setpz4(xq_sp.add(big_n4), S::subpz4(a04_p1_a26, a15_p1_a37));

            let s04_mj_s26 = S::subpz4(s04, js26);
            let w8_s15_mj_s37 = S::w8xpz4(S::subpz4(s15, js37));
            S::setpz4(xq_sp.add(big_n1), S::addpz4(s04_mj_s26, w8_s15_mj_s37));
            S::setpz4(xq_sp.add(big_n5), S::subpz4(s04_mj_s26, w8_s15_mj_s37));

            let a04_m1_a26 = S::subpz4(a04, a26);
            let j_a15_m1_a37 = S::jxpz4(S::subpz4(a15, a37));
            S::setpz4(xq_sp.add(big_n2), S::subpz4(a04_m1_a26, j_a15_m1_a37));
            S::setpz4(xq_sp.add(big_n6), S::addpz4(a04_m1_a26, j_a15_m1_a37));

            let s04_pj_s26 = S::addpz4(s04, js26);
            let v8_s15_pj_s37 = S::v8xpz4(S::addpz4(s15, js37));
            S::setpz4(xq_sp.add(big_n3), S::subpz4(s04_pj_s26, v8_s15_pj_s37));
            S::setpz4(xq_sp.add(big_n7), S::addpz4(s04_pj_s26, v8_s15_pj_s37));

            q += 4;
        }
    }
}

#[inline(always)]
unsafe fn fwdcore_1<S: FftSimd16>(big_n: usize, s: usize, x: *mut c64, y: *mut c64, w: *const c64) {
    debug_assert_eq!(s, 1);
    let big_n0 = 0;
    let big_n1 = big_n / 8;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;
    let big_n4 = big_n1 * 4;
    let big_n5 = big_n1 * 5;
    let big_n6 = big_n1 * 6;
    let big_n7 = big_n1 * 7;

    let mut p = 0;
    while p < big_n1 {
        let x_p = x.add(p);
        let y_8p = y.add(8 * p);

        let w1p = S::getpz2(twid(8, big_n, 1, w, p));
        let w2p = S::getpz2(twid(8, big_n, 2, w, p));
        let w3p = S::getpz2(twid(8, big_n, 3, w, p));
        let w4p = S::getpz2(twid(8, big_n, 4, w, p));
        let w5p = S::getpz2(twid(8, big_n, 5, w, p));
        let w6p = S::getpz2(twid(8, big_n, 6, w, p));
        let w7p = S::getpz2(twid(8, big_n, 7, w, p));
        let ab_0 = S::getpz2(y_8p.add(0));
        let cd_0 = S::getpz2(y_8p.add(2));
        let ef_0 = S::getpz2(y_8p.add(4));
        let gh_0 = S::getpz2(y_8p.add(6));
        let ab_1 = S::getpz2(y_8p.add(8));
        let cd_1 = S::getpz2(y_8p.add(10));
        let ef_1 = S::getpz2(y_8p.add(12));
        let gh_1 = S::getpz2(y_8p.add(14));
        let y0 = S::catlo(ab_0, ab_1);
        let y1 = S::mulpz2(w1p, S::cathi(ab_0, ab_1));
        let y2 = S::mulpz2(w2p, S::catlo(cd_0, cd_1));
        let y3 = S::mulpz2(w3p, S::cathi(cd_0, cd_1));
        let y4 = S::mulpz2(w4p, S::catlo(ef_0, ef_1));
        let y5 = S::mulpz2(w5p, S::cathi(ef_0, ef_1));
        let y6 = S::mulpz2(w6p, S::catlo(gh_0, gh_1));
        let y7 = S::mulpz2(w7p, S::cathi(gh_0, gh_1));

        let a04 = S::addpz2(y0, y4);
        let s04 = S::subpz2(y0, y4);
        let a26 = S::addpz2(y2, y6);
        let js26 = S::jxpz2(S::subpz2(y2, y6));
        let a15 = S::addpz2(y1, y5);
        let s15 = S::subpz2(y1, y5);
        let a37 = S::addpz2(y3, y7);
        let js37 = S::jxpz2(S::subpz2(y3, y7));

        let a04_p1_a26 = S::addpz2(a04, a26);
        let a15_p1_a37 = S::addpz2(a15, a37);
        S::setpz2(x_p.add(big_n0), S::addpz2(a04_p1_a26, a15_p1_a37));
        S::setpz2(x_p.add(big_n4), S::subpz2(a04_p1_a26, a15_p1_a37));

        let s04_mj_s26 = S::subpz2(s04, js26);
        let w8_s15_mj_s37 = S::w8xpz2(S::subpz2(s15, js37));
        S::setpz2(x_p.add(big_n1), S::addpz2(s04_mj_s26, w8_s15_mj_s37));
        S::setpz2(x_p.add(big_n5), S::subpz2(s04_mj_s26, w8_s15_mj_s37));

        let a04_m1_a26 = S::subpz2(a04, a26);
        let j_a15_m1_a37 = S::jxpz2(S::subpz2(a15, a37));
        S::setpz2(x_p.add(big_n2), S::subpz2(a04_m1_a26, j_a15_m1_a37));
        S::setpz2(x_p.add(big_n6), S::addpz2(a04_m1_a26, j_a15_m1_a37));

        let s04_pj_s26 = S::addpz2(s04, js26);
        let v8_s15_pj_s37 = S::v8xpz2(S::addpz2(s15, js37));
        S::setpz2(x_p.add(big_n3), S::subpz2(s04_pj_s26, v8_s15_pj_s37));
        S::setpz2(x_p.add(big_n7), S::addpz2(s04_pj_s26, v8_s15_pj_s37));

        p += 2;
    }
}

// backward butterfly
#[inline(always)]
unsafe fn invcore_s<S: FftSimd16>(n: usize, s: usize, x: *mut c64, y: *mut c64, w: *const c64) {
    debug_assert_ne!(s, 1);

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

        let w1p = S::duppz5(&*twid_t(8, big_n, 1, w, sp));
        let w2p = S::duppz5(&*twid_t(8, big_n, 2, w, sp));
        let w3p = S::duppz5(&*twid_t(8, big_n, 3, w, sp));
        let w4p = S::duppz5(&*twid_t(8, big_n, 4, w, sp));
        let w5p = S::duppz5(&*twid_t(8, big_n, 5, w, sp));
        let w6p = S::duppz5(&*twid_t(8, big_n, 6, w, sp));
        let w7p = S::duppz5(&*twid_t(8, big_n, 7, w, sp));

        let mut q = 0;
        while q < s {
            let xq_sp = x.add(q + sp);
            let yq_s8p = y.add(q + s8p);

            let y0 = S::getpz4(yq_s8p.add(0));
            let y1 = S::mulpz4(w1p, S::getpz4(yq_s8p.add(s)));
            let y2 = S::mulpz4(w2p, S::getpz4(yq_s8p.add(s * 2)));
            let y3 = S::mulpz4(w3p, S::getpz4(yq_s8p.add(s * 3)));
            let y4 = S::mulpz4(w4p, S::getpz4(yq_s8p.add(s * 4)));
            let y5 = S::mulpz4(w5p, S::getpz4(yq_s8p.add(s * 5)));
            let y6 = S::mulpz4(w6p, S::getpz4(yq_s8p.add(s * 6)));
            let y7 = S::mulpz4(w7p, S::getpz4(yq_s8p.add(s * 7)));
            let a04 = S::addpz4(y0, y4);
            let s04 = S::subpz4(y0, y4);
            let a26 = S::addpz4(y2, y6);
            let js26 = S::jxpz4(S::subpz4(y2, y6));
            let a15 = S::addpz4(y1, y5);
            let s15 = S::subpz4(y1, y5);
            let a37 = S::addpz4(y3, y7);
            let js37 = S::jxpz4(S::subpz4(y3, y7));

            let a04_p1_a26 = S::addpz4(a04, a26);
            let a15_p1_a37 = S::addpz4(a15, a37);
            S::setpz4(xq_sp.add(big_n0), S::addpz4(a04_p1_a26, a15_p1_a37));
            S::setpz4(xq_sp.add(big_n4), S::subpz4(a04_p1_a26, a15_p1_a37));

            let s04_pj_s26 = S::addpz4(s04, js26);
            let v8_s15_pj_s37 = S::v8xpz4(S::addpz4(s15, js37));
            S::setpz4(xq_sp.add(big_n1), S::addpz4(s04_pj_s26, v8_s15_pj_s37));
            S::setpz4(xq_sp.add(big_n5), S::subpz4(s04_pj_s26, v8_s15_pj_s37));

            let a04_m1_a26 = S::subpz4(a04, a26);
            let j_a15_m1_a37 = S::jxpz4(S::subpz4(a15, a37));
            S::setpz4(xq_sp.add(big_n2), S::addpz4(a04_m1_a26, j_a15_m1_a37));
            S::setpz4(xq_sp.add(big_n6), S::subpz4(a04_m1_a26, j_a15_m1_a37));

            let s04_mj_s26 = S::subpz4(s04, js26);
            let w8_s15_mj_s37 = S::w8xpz4(S::subpz4(s15, js37));
            S::setpz4(xq_sp.add(big_n3), S::subpz4(s04_mj_s26, w8_s15_mj_s37));
            S::setpz4(xq_sp.add(big_n7), S::addpz4(s04_mj_s26, w8_s15_mj_s37));

            q += 4;
        }
    }
}

#[inline(always)]
unsafe fn invcore_1<S: FftSimd16>(big_n: usize, s: usize, x: *mut c64, y: *mut c64, w: *const c64) {
    debug_assert_eq!(s, 1);
    let big_n0 = 0;
    let big_n1 = big_n / 8;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;
    let big_n4 = big_n1 * 4;
    let big_n5 = big_n1 * 5;
    let big_n6 = big_n1 * 6;
    let big_n7 = big_n1 * 7;

    let mut p = 0;
    while p < big_n1 {
        let x_p = x.add(p);
        let y_8p = y.add(8 * p);

        let w1p = S::getpz2(twid(8, big_n, 1, w, p));
        let w2p = S::getpz2(twid(8, big_n, 2, w, p));
        let w3p = S::getpz2(twid(8, big_n, 3, w, p));
        let w4p = S::getpz2(twid(8, big_n, 4, w, p));
        let w5p = S::getpz2(twid(8, big_n, 5, w, p));
        let w6p = S::getpz2(twid(8, big_n, 6, w, p));
        let w7p = S::getpz2(twid(8, big_n, 7, w, p));
        let ab_0 = S::getpz2(y_8p.add(0));
        let cd_0 = S::getpz2(y_8p.add(2));
        let ef_0 = S::getpz2(y_8p.add(4));
        let gh_0 = S::getpz2(y_8p.add(6));
        let ab_1 = S::getpz2(y_8p.add(8));
        let cd_1 = S::getpz2(y_8p.add(10));
        let ef_1 = S::getpz2(y_8p.add(12));
        let gh_1 = S::getpz2(y_8p.add(14));
        let y0 = S::catlo(ab_0, ab_1);
        let y1 = S::mulpz2(w1p, S::cathi(ab_0, ab_1));
        let y2 = S::mulpz2(w2p, S::catlo(cd_0, cd_1));
        let y3 = S::mulpz2(w3p, S::cathi(cd_0, cd_1));
        let y4 = S::mulpz2(w4p, S::catlo(ef_0, ef_1));
        let y5 = S::mulpz2(w5p, S::cathi(ef_0, ef_1));
        let y6 = S::mulpz2(w6p, S::catlo(gh_0, gh_1));
        let y7 = S::mulpz2(w7p, S::cathi(gh_0, gh_1));

        let a04 = S::addpz2(y0, y4);
        let s04 = S::subpz2(y0, y4);
        let a26 = S::addpz2(y2, y6);
        let js26 = S::jxpz2(S::subpz2(y2, y6));
        let a15 = S::addpz2(y1, y5);
        let s15 = S::subpz2(y1, y5);
        let a37 = S::addpz2(y3, y7);
        let js37 = S::jxpz2(S::subpz2(y3, y7));

        let a04_p1_a26 = S::addpz2(a04, a26);
        let a15_p1_a37 = S::addpz2(a15, a37);
        S::setpz2(x_p.add(big_n0), S::addpz2(a04_p1_a26, a15_p1_a37));
        S::setpz2(x_p.add(big_n4), S::subpz2(a04_p1_a26, a15_p1_a37));

        let s04_pj_s26 = S::addpz2(s04, js26);
        let v8_s15_pj_s37 = S::v8xpz2(S::addpz2(s15, js37));
        S::setpz2(x_p.add(big_n1), S::addpz2(s04_pj_s26, v8_s15_pj_s37));
        S::setpz2(x_p.add(big_n5), S::subpz2(s04_pj_s26, v8_s15_pj_s37));

        let a04_m1_a26 = S::subpz2(a04, a26);
        let j_a15_m1_a37 = S::jxpz2(S::subpz2(a15, a37));
        S::setpz2(x_p.add(big_n2), S::addpz2(a04_m1_a26, j_a15_m1_a37));
        S::setpz2(x_p.add(big_n6), S::subpz2(a04_m1_a26, j_a15_m1_a37));

        let s04_mj_s26 = S::subpz2(s04, js26);
        let w8_s15_mj_s37 = S::w8xpz2(S::subpz2(s15, js37));
        S::setpz2(x_p.add(big_n3), S::subpz2(s04_mj_s26, w8_s15_mj_s37));
        S::setpz2(x_p.add(big_n7), S::addpz2(s04_mj_s26, w8_s15_mj_s37));

        p += 2;
    }
}

#[inline(always)]
pub(crate) unsafe fn fwdend_8_s<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 8);
    debug_assert_ne!(s, 1);

    let z = if eo { y } else { x };

    let mut q = 0;
    while q < s {
        let xq = x.add(q);
        let zq = z.add(q);

        let z0 = S::getpz2(zq.add(0));
        let z1 = S::getpz2(zq.add(s));
        let z2 = S::getpz2(zq.add(s * 2));
        let z3 = S::getpz2(zq.add(s * 3));
        let z4 = S::getpz2(zq.add(s * 4));
        let z5 = S::getpz2(zq.add(s * 5));
        let z6 = S::getpz2(zq.add(s * 6));
        let z7 = S::getpz2(zq.add(s * 7));
        let a04 = S::addpz2(z0, z4);
        let s04 = S::subpz2(z0, z4);
        let a26 = S::addpz2(z2, z6);
        let js26 = S::jxpz2(S::subpz2(z2, z6));
        let a15 = S::addpz2(z1, z5);
        let s15 = S::subpz2(z1, z5);
        let a37 = S::addpz2(z3, z7);
        let js37 = S::jxpz2(S::subpz2(z3, z7));
        let a04_p1_a26 = S::addpz2(a04, a26);
        let s04_mj_s26 = S::subpz2(s04, js26);
        let a04_m1_a26 = S::subpz2(a04, a26);
        let s04_pj_s26 = S::addpz2(s04, js26);
        let a15_p1_a37 = S::addpz2(a15, a37);
        let w8_s15_mj_s37 = S::w8xpz2(S::subpz2(s15, js37));
        let j_a15_m1_a37 = S::jxpz2(S::subpz2(a15, a37));
        let v8_s15_pj_s37 = S::v8xpz2(S::addpz2(s15, js37));
        S::setpz2(xq.add(0), S::addpz2(a04_p1_a26, a15_p1_a37));
        S::setpz2(xq.add(s), S::addpz2(s04_mj_s26, w8_s15_mj_s37));
        S::setpz2(xq.add(s * 2), S::subpz2(a04_m1_a26, j_a15_m1_a37));
        S::setpz2(xq.add(s * 3), S::subpz2(s04_pj_s26, v8_s15_pj_s37));
        S::setpz2(xq.add(s * 4), S::subpz2(a04_p1_a26, a15_p1_a37));
        S::setpz2(xq.add(s * 5), S::subpz2(s04_mj_s26, w8_s15_mj_s37));
        S::setpz2(xq.add(s * 6), S::addpz2(a04_m1_a26, j_a15_m1_a37));
        S::setpz2(xq.add(s * 7), S::addpz2(s04_pj_s26, v8_s15_pj_s37));

        q += 2;
    }
}

#[inline(always)]
pub(crate) unsafe fn fwdend_8_1<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 8);
    debug_assert_eq!(s, 1);

    let z = if eo { y } else { x };

    let z0 = S::getpz(&*z.add(0));
    let z1 = S::getpz(&*z.add(1));
    let z2 = S::getpz(&*z.add(2));
    let z3 = S::getpz(&*z.add(3));
    let z4 = S::getpz(&*z.add(4));
    let z5 = S::getpz(&*z.add(5));
    let z6 = S::getpz(&*z.add(6));
    let z7 = S::getpz(&*z.add(7));
    let a04 = S::addpz(z0, z4);
    let s04 = S::subpz(z0, z4);
    let a26 = S::addpz(z2, z6);
    let js26 = S::jxpz(S::subpz(z2, z6));
    let a15 = S::addpz(z1, z5);
    let s15 = S::subpz(z1, z5);
    let a37 = S::addpz(z3, z7);
    let js37 = S::jxpz(S::subpz(z3, z7));
    let a04_p1_a26 = S::addpz(a04, a26);
    let s04_mj_s26 = S::subpz(s04, js26);
    let a04_m1_a26 = S::subpz(a04, a26);
    let s04_pj_s26 = S::addpz(s04, js26);
    let a15_p1_a37 = S::addpz(a15, a37);
    let w8_s15_mj_s37 = S::w8xpz(S::subpz(s15, js37));
    let j_a15_m1_a37 = S::jxpz(S::subpz(a15, a37));
    let v8_s15_pj_s37 = S::v8xpz(S::addpz(s15, js37));
    S::setpz(x.add(0), S::addpz(a04_p1_a26, a15_p1_a37));
    S::setpz(x.add(1), S::addpz(s04_mj_s26, w8_s15_mj_s37));
    S::setpz(x.add(2), S::subpz(a04_m1_a26, j_a15_m1_a37));
    S::setpz(x.add(3), S::subpz(s04_pj_s26, v8_s15_pj_s37));
    S::setpz(x.add(4), S::subpz(a04_p1_a26, a15_p1_a37));
    S::setpz(x.add(5), S::subpz(s04_mj_s26, w8_s15_mj_s37));
    S::setpz(x.add(6), S::addpz(a04_m1_a26, j_a15_m1_a37));
    S::setpz(x.add(7), S::addpz(s04_pj_s26, v8_s15_pj_s37));
}

#[inline(always)]
pub(crate) unsafe fn invend_8_s<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 8);
    debug_assert_ne!(s, 1);

    let z = if eo { y } else { x };

    let mut q = 0;
    while q < s {
        let xq = x.add(q);
        let zq = z.add(q);

        let z0 = S::getpz2(zq.add(0));
        let z1 = S::getpz2(zq.add(s));
        let z2 = S::getpz2(zq.add(s * 2));
        let z3 = S::getpz2(zq.add(s * 3));
        let z4 = S::getpz2(zq.add(s * 4));
        let z5 = S::getpz2(zq.add(s * 5));
        let z6 = S::getpz2(zq.add(s * 6));
        let z7 = S::getpz2(zq.add(s * 7));
        let a04 = S::addpz2(z0, z4);
        let s04 = S::subpz2(z0, z4);
        let a26 = S::addpz2(z2, z6);
        let js26 = S::jxpz2(S::subpz2(z2, z6));
        let a15 = S::addpz2(z1, z5);
        let s15 = S::subpz2(z1, z5);
        let a37 = S::addpz2(z3, z7);
        let js37 = S::jxpz2(S::subpz2(z3, z7));
        let a04_p1_a26 = S::addpz2(a04, a26);
        let s04_pj_s26 = S::addpz2(s04, js26);
        let a04_m1_a26 = S::subpz2(a04, a26);
        let s04_mj_s26 = S::subpz2(s04, js26);
        let a15_p1_a37 = S::addpz2(a15, a37);
        let v8_s15_pj_s37 = S::v8xpz2(S::addpz2(s15, js37));
        let j_a15_m1_a37 = S::jxpz2(S::subpz2(a15, a37));
        let w8_s15_mj_s37 = S::w8xpz2(S::subpz2(s15, js37));
        S::setpz2(xq.add(0), S::addpz2(a04_p1_a26, a15_p1_a37));
        S::setpz2(xq.add(s), S::addpz2(s04_pj_s26, v8_s15_pj_s37));
        S::setpz2(xq.add(s * 2), S::addpz2(a04_m1_a26, j_a15_m1_a37));
        S::setpz2(xq.add(s * 3), S::subpz2(s04_mj_s26, w8_s15_mj_s37));
        S::setpz2(xq.add(s * 4), S::subpz2(a04_p1_a26, a15_p1_a37));
        S::setpz2(xq.add(s * 5), S::subpz2(s04_pj_s26, v8_s15_pj_s37));
        S::setpz2(xq.add(s * 6), S::subpz2(a04_m1_a26, j_a15_m1_a37));
        S::setpz2(xq.add(s * 7), S::addpz2(s04_mj_s26, w8_s15_mj_s37));

        q += 2;
    }
}

#[inline(always)]
pub(crate) unsafe fn invend_8_1<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 8);
    debug_assert_eq!(s, 1);

    let z = if eo { y } else { x };

    let z0 = S::getpz(&*z.add(0));
    let z1 = S::getpz(&*z.add(1));
    let z2 = S::getpz(&*z.add(2));
    let z3 = S::getpz(&*z.add(3));
    let z4 = S::getpz(&*z.add(4));
    let z5 = S::getpz(&*z.add(5));
    let z6 = S::getpz(&*z.add(6));
    let z7 = S::getpz(&*z.add(7));
    let a04 = S::addpz(z0, z4);
    let s04 = S::subpz(z0, z4);
    let a26 = S::addpz(z2, z6);
    let js26 = S::jxpz(S::subpz(z2, z6));
    let a15 = S::addpz(z1, z5);
    let s15 = S::subpz(z1, z5);
    let a37 = S::addpz(z3, z7);
    let js37 = S::jxpz(S::subpz(z3, z7));
    let a04_p1_a26 = S::addpz(a04, a26);
    let s04_pj_s26 = S::addpz(s04, js26);
    let a04_m1_a26 = S::subpz(a04, a26);
    let s04_mj_s26 = S::subpz(s04, js26);
    let a15_p1_a37 = S::addpz(a15, a37);
    let v8_s15_pj_s37 = S::v8xpz(S::addpz(s15, js37));
    let j_a15_m1_a37 = S::jxpz(S::subpz(a15, a37));
    let w8_s15_mj_s37 = S::w8xpz(S::subpz(s15, js37));
    S::setpz(x.add(0), S::addpz(a04_p1_a26, a15_p1_a37));
    S::setpz(x.add(1), S::addpz(s04_pj_s26, v8_s15_pj_s37));
    S::setpz(x.add(2), S::addpz(a04_m1_a26, j_a15_m1_a37));
    S::setpz(x.add(3), S::subpz(s04_mj_s26, w8_s15_mj_s37));
    S::setpz(x.add(4), S::subpz(a04_p1_a26, a15_p1_a37));
    S::setpz(x.add(5), S::subpz(s04_pj_s26, v8_s15_pj_s37));
    S::setpz(x.add(6), S::subpz(a04_m1_a26, j_a15_m1_a37));
    S::setpz(x.add(7), S::addpz(s04_mj_s26, w8_s15_mj_s37));
}

/// Initialize twiddles for subsequent forward and inverse Fourier transforms of size `n`.
/// `twiddles` must be of length `2*n`.
pub fn init_twiddles(forward: bool, n: usize, twiddles: &mut [c64]) {
    crate::dif8::init_twiddles(forward, n, twiddles);
}

include!(concat!(env!("OUT_DIR"), "/dit8.rs"));
include!("./boilerplate.rs");
