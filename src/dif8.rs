use crate::fft_simd::FftSimd16;
use crate::twiddles::{twid, twid_t};
use crate::{c64, MAX_EXP};
use seq_macro::seq;

pub(crate) use crate::dif4::fwdend_2_1;
pub(crate) use crate::dif4::fwdend_2_s;
pub(crate) use crate::dif4::fwdend_4_1;
pub(crate) use crate::dif4::fwdend_4_s;
pub(crate) use crate::dif4::invend_2_1;
pub(crate) use crate::dif4::invend_2_s;
pub(crate) use crate::dif4::invend_4_1;
pub(crate) use crate::dif4::invend_4_s;

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

        seq! {K in 1..8 {
            let wp~K = S::duppz3(&*twid_t(8, big_n, K, w, sp));
        }}

        let mut q = 0;
        while q < s {
            let xq_sp = x.add(q + sp);
            let yq_s8p = y.add(q + s8p);

            seq! {K in 0..8 {
                let x~K = S::getpz2(xq_sp.add(big_n~K));
            }}

            let a04 = S::addpz2(x0, x4);
            let s04 = S::subpz2(x0, x4);
            let a26 = S::addpz2(x2, x6);
            let js26 = S::jxpz2(S::subpz2(x2, x6));
            let a15 = S::addpz2(x1, x5);
            let s15 = S::subpz2(x1, x5);
            let a37 = S::addpz2(x3, x7);
            let js37 = S::jxpz2(S::subpz2(x3, x7));
            let a04_p1_a26 = S::addpz2(a04, a26);
            let s04_mj_s26 = S::subpz2(s04, js26);
            let a04_m1_a26 = S::subpz2(a04, a26);
            let s04_pj_s26 = S::addpz2(s04, js26);
            let a15_p1_a37 = S::addpz2(a15, a37);
            let w8_s15_mj_s37 = S::w8xpz2(S::subpz2(s15, js37));
            let j_a15_m1_a37 = S::jxpz2(S::subpz2(a15, a37));
            let v8_s15_pj_s37 = S::v8xpz2(S::addpz2(s15, js37));

            S::setpz2(yq_s8p, S::addpz2(a04_p1_a26, a15_p1_a37));
            S::setpz2(
                yq_s8p.add(s),
                S::mulpz2(wp1, S::addpz2(s04_mj_s26, w8_s15_mj_s37)),
            );
            S::setpz2(
                yq_s8p.add(s * 2),
                S::mulpz2(wp2, S::subpz2(a04_m1_a26, j_a15_m1_a37)),
            );
            S::setpz2(
                yq_s8p.add(s * 3),
                S::mulpz2(wp3, S::subpz2(s04_pj_s26, v8_s15_pj_s37)),
            );
            S::setpz2(
                yq_s8p.add(s * 4),
                S::mulpz2(wp4, S::subpz2(a04_p1_a26, a15_p1_a37)),
            );
            S::setpz2(
                yq_s8p.add(s * 5),
                S::mulpz2(wp5, S::subpz2(s04_mj_s26, w8_s15_mj_s37)),
            );
            S::setpz2(
                yq_s8p.add(s * 6),
                S::mulpz2(wp6, S::addpz2(a04_m1_a26, j_a15_m1_a37)),
            );
            S::setpz2(
                yq_s8p.add(s * 7),
                S::mulpz2(wp7, S::addpz2(s04_pj_s26, v8_s15_pj_s37)),
            );

            q += 2;
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

        seq! {K in 0..8 {
            let x~K = S::getpz2(x_p.add(big_n~K));
        }}

        let a04 = S::addpz2(x0, x4);
        let s04 = S::subpz2(x0, x4);
        let a26 = S::addpz2(x2, x6);
        let js26 = S::jxpz2(S::subpz2(x2, x6));
        let a15 = S::addpz2(x1, x5);
        let s15 = S::subpz2(x1, x5);
        let a37 = S::addpz2(x3, x7);
        let js37 = S::jxpz2(S::subpz2(x3, x7));

        let a04_p1_a26 = S::addpz2(a04, a26);
        let s04_mj_s26 = S::subpz2(s04, js26);
        let a04_m1_a26 = S::subpz2(a04, a26);
        let s04_pj_s26 = S::addpz2(s04, js26);
        let a15_p1_a37 = S::addpz2(a15, a37);
        let w8_s15_mj_s37 = S::w8xpz2(S::subpz2(s15, js37));
        let j_a15_m1_a37 = S::jxpz2(S::subpz2(a15, a37));
        let v8_s15_pj_s37 = S::v8xpz2(S::addpz2(s15, js37));

        seq! {K in 1..8 {
            let wp~K = S::getpz2(&*twid(8, big_n, K, w, p));
        }}

        let aa = S::addpz2(a04_p1_a26, a15_p1_a37);
        let bb = S::mulpz2(wp1, S::addpz2(s04_mj_s26, w8_s15_mj_s37));
        let cc = S::mulpz2(wp2, S::subpz2(a04_m1_a26, j_a15_m1_a37));
        let dd = S::mulpz2(wp3, S::subpz2(s04_pj_s26, v8_s15_pj_s37));
        let ee = S::mulpz2(wp4, S::subpz2(a04_p1_a26, a15_p1_a37));
        let ff = S::mulpz2(wp5, S::subpz2(s04_mj_s26, w8_s15_mj_s37));
        let gg = S::mulpz2(wp6, S::addpz2(a04_m1_a26, j_a15_m1_a37));
        let hh = S::mulpz2(wp7, S::addpz2(s04_pj_s26, v8_s15_pj_s37));

        {
            let ab = S::catlo(aa, bb);
            S::setpz2(y_8p.add(0), ab);
            let cd = S::catlo(cc, dd);
            S::setpz2(y_8p.add(2), cd);
            let ef = S::catlo(ee, ff);
            S::setpz2(y_8p.add(4), ef);
            let gh = S::catlo(gg, hh);
            S::setpz2(y_8p.add(6), gh);
        }
        {
            let ab = S::cathi(aa, bb);
            S::setpz2(y_8p.add(8), ab);
            let cd = S::cathi(cc, dd);
            S::setpz2(y_8p.add(10), cd);
            let ef = S::cathi(ee, ff);
            S::setpz2(y_8p.add(12), ef);
            let gh = S::cathi(gg, hh);
            S::setpz2(y_8p.add(14), gh);
        }

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

        seq! {K in 1..8 {
            let wp~K = S::duppz3(&*twid_t(8, big_n, K, w, sp));
        }}

        let mut q = 0;
        while q < s {
            let xq_sp = x.add(q + sp);
            let yq_s8p = y.add(q + s8p);

            seq! {K in 0..8 {
                let x~K = S::getpz2(xq_sp.add(big_n~K));
            }}

            let a04 = S::addpz2(x0, x4);
            let s04 = S::subpz2(x0, x4);
            let a26 = S::addpz2(x2, x6);
            let js26 = S::jxpz2(S::subpz2(x2, x6));
            let a15 = S::addpz2(x1, x5);
            let s15 = S::subpz2(x1, x5);
            let a37 = S::addpz2(x3, x7);
            let js37 = S::jxpz2(S::subpz2(x3, x7));
            let a04_p1_a26 = S::addpz2(a04, a26);
            let s04_pj_s26 = S::addpz2(s04, js26);
            let a04_m1_a26 = S::subpz2(a04, a26);
            let s04_mj_s26 = S::subpz2(s04, js26);
            let a15_p1_a37 = S::addpz2(a15, a37);
            let v8_s15_pj_s37 = S::v8xpz2(S::addpz2(s15, js37));
            let j_a15_m1_a37 = S::jxpz2(S::subpz2(a15, a37));
            let w8_s15_mj_s37 = S::w8xpz2(S::subpz2(s15, js37));

            S::setpz2(yq_s8p.add(0), S::addpz2(a04_p1_a26, a15_p1_a37));
            S::setpz2(
                yq_s8p.add(s),
                S::mulpz2(wp1, S::addpz2(s04_pj_s26, v8_s15_pj_s37)),
            );
            S::setpz2(
                yq_s8p.add(s * 2),
                S::mulpz2(wp2, S::addpz2(a04_m1_a26, j_a15_m1_a37)),
            );
            S::setpz2(
                yq_s8p.add(s * 3),
                S::mulpz2(wp3, S::subpz2(s04_mj_s26, w8_s15_mj_s37)),
            );
            S::setpz2(
                yq_s8p.add(s * 4),
                S::mulpz2(wp4, S::subpz2(a04_p1_a26, a15_p1_a37)),
            );
            S::setpz2(
                yq_s8p.add(s * 5),
                S::mulpz2(wp5, S::subpz2(s04_pj_s26, v8_s15_pj_s37)),
            );
            S::setpz2(
                yq_s8p.add(s * 6),
                S::mulpz2(wp6, S::subpz2(a04_m1_a26, j_a15_m1_a37)),
            );
            S::setpz2(
                yq_s8p.add(s * 7),
                S::mulpz2(wp7, S::addpz2(s04_mj_s26, w8_s15_mj_s37)),
            );

            q += 2;
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

        seq! {K in 0..8 {
            let x~K = S::getpz2(x_p.add(big_n~K));
        }}

        let a04 = S::addpz2(x0, x4);
        let s04 = S::subpz2(x0, x4);
        let a26 = S::addpz2(x2, x6);
        let js26 = S::jxpz2(S::subpz2(x2, x6));
        let a15 = S::addpz2(x1, x5);
        let s15 = S::subpz2(x1, x5);
        let a37 = S::addpz2(x3, x7);
        let js37 = S::jxpz2(S::subpz2(x3, x7));

        let a04_p1_a26 = S::addpz2(a04, a26);
        let s04_pj_s26 = S::addpz2(s04, js26);
        let a04_m1_a26 = S::subpz2(a04, a26);
        let s04_mj_s26 = S::subpz2(s04, js26);
        let a15_p1_a37 = S::addpz2(a15, a37);
        let v8_s15_pj_s37 = S::v8xpz2(S::addpz2(s15, js37));
        let j_a15_m1_a37 = S::jxpz2(S::subpz2(a15, a37));
        let w8_s15_mj_s37 = S::w8xpz2(S::subpz2(s15, js37));

        seq! {K in 1..8 {
            let wp~K = S::getpz2(&*twid(8, big_n, K, w, p));
        }}

        let aa = S::addpz2(a04_p1_a26, a15_p1_a37);
        let bb = S::mulpz2(wp1, S::addpz2(s04_pj_s26, v8_s15_pj_s37));
        let cc = S::mulpz2(wp2, S::addpz2(a04_m1_a26, j_a15_m1_a37));
        let dd = S::mulpz2(wp3, S::subpz2(s04_mj_s26, w8_s15_mj_s37));
        let ee = S::mulpz2(wp4, S::subpz2(a04_p1_a26, a15_p1_a37));
        let ff = S::mulpz2(wp5, S::subpz2(s04_pj_s26, v8_s15_pj_s37));
        let gg = S::mulpz2(wp6, S::subpz2(a04_m1_a26, j_a15_m1_a37));
        let hh = S::mulpz2(wp7, S::addpz2(s04_mj_s26, w8_s15_mj_s37));

        {
            let ab = S::catlo(aa, bb);
            S::setpz2(y_8p.add(0), ab);
            let cd = S::catlo(cc, dd);
            S::setpz2(y_8p.add(2), cd);
            let ef = S::catlo(ee, ff);
            S::setpz2(y_8p.add(4), ef);
            let gh = S::catlo(gg, hh);
            S::setpz2(y_8p.add(6), gh);
        }
        {
            let ab = S::cathi(aa, bb);
            S::setpz2(y_8p.add(8), ab);
            let cd = S::cathi(cc, dd);
            S::setpz2(y_8p.add(10), cd);
            let ef = S::cathi(ee, ff);
            S::setpz2(y_8p.add(12), ef);
            let gh = S::cathi(gg, hh);
            S::setpz2(y_8p.add(14), gh);
        }

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

        seq! {K in 0..8 {
            #[allow(clippy::erasing_op, clippy::identity_op)]
            let x~K = S::getpz2(xq.add(s * K));
        }}

        let a04 = S::addpz2(x0, x4);
        let s04 = S::subpz2(x0, x4);
        let a26 = S::addpz2(x2, x6);
        let js26 = S::jxpz2(S::subpz2(x2, x6));
        let a15 = S::addpz2(x1, x5);
        let s15 = S::subpz2(x1, x5);
        let a37 = S::addpz2(x3, x7);
        let js37 = S::jxpz2(S::subpz2(x3, x7));

        let a04_p1_a26 = S::addpz2(a04, a26);
        let s04_mj_s26 = S::subpz2(s04, js26);
        let a04_m1_a26 = S::subpz2(a04, a26);
        let s04_pj_s26 = S::addpz2(s04, js26);
        let a15_p1_a37 = S::addpz2(a15, a37);
        let w8_s15_mj_s37 = S::w8xpz2(S::subpz2(s15, js37));
        let j_a15_m1_a37 = S::jxpz2(S::subpz2(a15, a37));
        let v8_s15_pj_s37 = S::v8xpz2(S::addpz2(s15, js37));

        S::setpz2(zq.add(0), S::addpz2(a04_p1_a26, a15_p1_a37));
        S::setpz2(zq.add(s), S::addpz2(s04_mj_s26, w8_s15_mj_s37));
        S::setpz2(zq.add(s * 2), S::subpz2(a04_m1_a26, j_a15_m1_a37));
        S::setpz2(zq.add(s * 3), S::subpz2(s04_pj_s26, v8_s15_pj_s37));
        S::setpz2(zq.add(s * 4), S::subpz2(a04_p1_a26, a15_p1_a37));
        S::setpz2(zq.add(s * 5), S::subpz2(s04_mj_s26, w8_s15_mj_s37));
        S::setpz2(zq.add(s * 6), S::addpz2(a04_m1_a26, j_a15_m1_a37));
        S::setpz2(zq.add(s * 7), S::addpz2(s04_pj_s26, v8_s15_pj_s37));

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

    seq! {K in 0..8 {
        #[allow(clippy::erasing_op, clippy::identity_op)]
        let x~K = S::getpz(&*x.add(K));
    }}

    let a04 = S::addpz(x0, x4);
    let s04 = S::subpz(x0, x4);
    let a26 = S::addpz(x2, x6);
    let js26 = S::jxpz(S::subpz(x2, x6));
    let a15 = S::addpz(x1, x5);
    let s15 = S::subpz(x1, x5);
    let a37 = S::addpz(x3, x7);
    let js37 = S::jxpz(S::subpz(x3, x7));
    let a04_p1_a26 = S::addpz(a04, a26);
    let s04_mj_s26 = S::subpz(s04, js26);
    let a04_m1_a26 = S::subpz(a04, a26);
    let s04_pj_s26 = S::addpz(s04, js26);
    let a15_p1_a37 = S::addpz(a15, a37);
    let w8_s15_mj_s37 = S::w8xpz(S::subpz(s15, js37));
    let j_a15_m1_a37 = S::jxpz(S::subpz(a15, a37));
    let v8_s15_pj_s37 = S::v8xpz(S::addpz(s15, js37));

    S::setpz(z.add(0), S::addpz(a04_p1_a26, a15_p1_a37));
    S::setpz(z.add(1), S::addpz(s04_mj_s26, w8_s15_mj_s37));
    S::setpz(z.add(2), S::subpz(a04_m1_a26, j_a15_m1_a37));
    S::setpz(z.add(3), S::subpz(s04_pj_s26, v8_s15_pj_s37));
    S::setpz(z.add(4), S::subpz(a04_p1_a26, a15_p1_a37));
    S::setpz(z.add(5), S::subpz(s04_mj_s26, w8_s15_mj_s37));
    S::setpz(z.add(6), S::addpz(a04_m1_a26, j_a15_m1_a37));
    S::setpz(z.add(7), S::addpz(s04_pj_s26, v8_s15_pj_s37));
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

        seq! {K in 0..8 {
            #[allow(clippy::erasing_op, clippy::identity_op)]
            let x~K = S::getpz2(xq.add(s * K));
        }}

        let a04 = S::addpz2(x0, x4);
        let s04 = S::subpz2(x0, x4);
        let a26 = S::addpz2(x2, x6);
        let js26 = S::jxpz2(S::subpz2(x2, x6));
        let a15 = S::addpz2(x1, x5);
        let s15 = S::subpz2(x1, x5);
        let a37 = S::addpz2(x3, x7);
        let js37 = S::jxpz2(S::subpz2(x3, x7));

        let a04_p1_a26 = S::addpz2(a04, a26);
        let s04_pj_s26 = S::addpz2(s04, js26);
        let a04_m1_a26 = S::subpz2(a04, a26);
        let s04_mj_s26 = S::subpz2(s04, js26);
        let a15_p1_a37 = S::addpz2(a15, a37);
        let v8_s15_pj_s37 = S::v8xpz2(S::addpz2(s15, js37));
        let j_a15_m1_a37 = S::jxpz2(S::subpz2(a15, a37));
        let w8_s15_mj_s37 = S::w8xpz2(S::subpz2(s15, js37));

        S::setpz2(zq.add(0), S::addpz2(a04_p1_a26, a15_p1_a37));
        S::setpz2(zq.add(s), S::addpz2(s04_pj_s26, v8_s15_pj_s37));
        S::setpz2(zq.add(s * 2), S::addpz2(a04_m1_a26, j_a15_m1_a37));
        S::setpz2(zq.add(s * 3), S::subpz2(s04_mj_s26, w8_s15_mj_s37));
        S::setpz2(zq.add(s * 4), S::subpz2(a04_p1_a26, a15_p1_a37));
        S::setpz2(zq.add(s * 5), S::subpz2(s04_pj_s26, v8_s15_pj_s37));
        S::setpz2(zq.add(s * 6), S::subpz2(a04_m1_a26, j_a15_m1_a37));
        S::setpz2(zq.add(s * 7), S::addpz2(s04_mj_s26, w8_s15_mj_s37));

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

    seq! {K in 0..8 {
        #[allow(clippy::erasing_op, clippy::identity_op)]
        let x~K = S::getpz(&*x.add(K));
    }}

    let a04 = S::addpz(x0, x4);
    let s04 = S::subpz(x0, x4);
    let a26 = S::addpz(x2, x6);
    let js26 = S::jxpz(S::subpz(x2, x6));
    let a15 = S::addpz(x1, x5);
    let s15 = S::subpz(x1, x5);
    let a37 = S::addpz(x3, x7);
    let js37 = S::jxpz(S::subpz(x3, x7));

    let a04_p1_a26 = S::addpz(a04, a26);
    let s04_pj_s26 = S::addpz(s04, js26);
    let a04_m1_a26 = S::subpz(a04, a26);
    let s04_mj_s26 = S::subpz(s04, js26);
    let a15_p1_a37 = S::addpz(a15, a37);
    let v8_s15_pj_s37 = S::v8xpz(S::addpz(s15, js37));
    let j_a15_m1_a37 = S::jxpz(S::subpz(a15, a37));
    let w8_s15_mj_s37 = S::w8xpz(S::subpz(s15, js37));

    S::setpz(z.add(0), S::addpz(a04_p1_a26, a15_p1_a37));
    S::setpz(z.add(1), S::addpz(s04_pj_s26, v8_s15_pj_s37));
    S::setpz(z.add(2), S::addpz(a04_m1_a26, j_a15_m1_a37));
    S::setpz(z.add(3), S::subpz(s04_mj_s26, w8_s15_mj_s37));
    S::setpz(z.add(4), S::subpz(a04_p1_a26, a15_p1_a37));
    S::setpz(z.add(5), S::subpz(s04_pj_s26, v8_s15_pj_s37));
    S::setpz(z.add(6), S::subpz(a04_m1_a26, j_a15_m1_a37));
    S::setpz(z.add(7), S::addpz(s04_mj_s26, w8_s15_mj_s37));
}

/// Initialize twiddles for subsequent forward and inverse Fourier transforms of size `n`.
/// `twiddles` must be of length `2*n`.
pub fn init_twiddles(forward: bool, n: usize, twiddles: &mut [c64]) {
    assert!(n.is_power_of_two());
    let i = n.trailing_zeros() as usize;
    assert!(i < MAX_EXP);
    assert_eq!(twiddles.len(), 2 * n);

    unsafe {
        crate::twiddles::init_wt(forward, 8, n, twiddles.as_mut_ptr());
    }
}

include!(concat!(env!("OUT_DIR"), "/dif8.rs"));
