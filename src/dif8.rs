use crate::fft_simd::FftSimd16;
use crate::impl_main_fn;
use crate::twiddles::{twid, twid_t};
use crate::{c64, MAX_EXP};
use dyn_stack::DynStack;
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
            let wp~K = S::duppz5(&*twid_t(8, big_n, K, w, sp));
        }}

        let mut q = 0;
        while q < s {
            let xq_sp = x.add(q + sp);
            let yq_s8p = y.add(q + s8p);

            seq! {K in 0..8 {
                let x~K = S::getpz4(xq_sp.add(big_n~K));
            }}

            let a04 = S::addpz4(x0, x4);
            let s04 = S::subpz4(x0, x4);
            let a26 = S::addpz4(x2, x6);
            let js26 = S::jxpz4(S::subpz4(x2, x6));
            let a15 = S::addpz4(x1, x5);
            let s15 = S::subpz4(x1, x5);
            let a37 = S::addpz4(x3, x7);
            let js37 = S::jxpz4(S::subpz4(x3, x7));
            let a04_p1_a26 = S::addpz4(a04, a26);
            let s04_mj_s26 = S::subpz4(s04, js26);
            let a04_m1_a26 = S::subpz4(a04, a26);
            let s04_pj_s26 = S::addpz4(s04, js26);
            let a15_p1_a37 = S::addpz4(a15, a37);
            let w8_s15_mj_s37 = S::w8xpz4(S::subpz4(s15, js37));
            let j_a15_m1_a37 = S::jxpz4(S::subpz4(a15, a37));
            let v8_s15_pj_s37 = S::v8xpz4(S::addpz4(s15, js37));

            S::setpz4(yq_s8p, S::addpz4(a04_p1_a26, a15_p1_a37));
            S::setpz4(
                yq_s8p.add(s),
                S::mulpz4(wp1, S::addpz4(s04_mj_s26, w8_s15_mj_s37)),
            );
            S::setpz4(
                yq_s8p.add(s * 2),
                S::mulpz4(wp2, S::subpz4(a04_m1_a26, j_a15_m1_a37)),
            );
            S::setpz4(
                yq_s8p.add(s * 3),
                S::mulpz4(wp3, S::subpz4(s04_pj_s26, v8_s15_pj_s37)),
            );
            S::setpz4(
                yq_s8p.add(s * 4),
                S::mulpz4(wp4, S::subpz4(a04_p1_a26, a15_p1_a37)),
            );
            S::setpz4(
                yq_s8p.add(s * 5),
                S::mulpz4(wp5, S::subpz4(s04_mj_s26, w8_s15_mj_s37)),
            );
            S::setpz4(
                yq_s8p.add(s * 6),
                S::mulpz4(wp6, S::addpz4(a04_m1_a26, j_a15_m1_a37)),
            );
            S::setpz4(
                yq_s8p.add(s * 7),
                S::mulpz4(wp7, S::addpz4(s04_pj_s26, v8_s15_pj_s37)),
            );

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
            let wp~K = S::cnjpz4(S::duppz5(&*twid_t(8, big_n, K, w, sp)));
        }}

        let mut q = 0;
        while q < s {
            let xq_sp = x.add(q + sp);
            let yq_s8p = y.add(q + s8p);

            seq! {K in 0..8 {
                let x~K = S::getpz4(xq_sp.add(big_n~K));
            }}

            let a04 = S::addpz4(x0, x4);
            let s04 = S::subpz4(x0, x4);
            let a26 = S::addpz4(x2, x6);
            let js26 = S::jxpz4(S::subpz4(x2, x6));
            let a15 = S::addpz4(x1, x5);
            let s15 = S::subpz4(x1, x5);
            let a37 = S::addpz4(x3, x7);
            let js37 = S::jxpz4(S::subpz4(x3, x7));
            let a04_p1_a26 = S::addpz4(a04, a26);
            let s04_pj_s26 = S::addpz4(s04, js26);
            let a04_m1_a26 = S::subpz4(a04, a26);
            let s04_mj_s26 = S::subpz4(s04, js26);
            let a15_p1_a37 = S::addpz4(a15, a37);
            let v8_s15_pj_s37 = S::v8xpz4(S::addpz4(s15, js37));
            let j_a15_m1_a37 = S::jxpz4(S::subpz4(a15, a37));
            let w8_s15_mj_s37 = S::w8xpz4(S::subpz4(s15, js37));

            S::setpz4(yq_s8p.add(0), S::addpz4(a04_p1_a26, a15_p1_a37));
            S::setpz4(
                yq_s8p.add(s),
                S::mulpz4(wp1, S::addpz4(s04_pj_s26, v8_s15_pj_s37)),
            );
            S::setpz4(
                yq_s8p.add(s * 2),
                S::mulpz4(wp2, S::addpz4(a04_m1_a26, j_a15_m1_a37)),
            );
            S::setpz4(
                yq_s8p.add(s * 3),
                S::mulpz4(wp3, S::subpz4(s04_mj_s26, w8_s15_mj_s37)),
            );
            S::setpz4(
                yq_s8p.add(s * 4),
                S::mulpz4(wp4, S::subpz4(a04_p1_a26, a15_p1_a37)),
            );
            S::setpz4(
                yq_s8p.add(s * 5),
                S::mulpz4(wp5, S::subpz4(s04_pj_s26, v8_s15_pj_s37)),
            );
            S::setpz4(
                yq_s8p.add(s * 6),
                S::mulpz4(wp6, S::subpz4(a04_m1_a26, j_a15_m1_a37)),
            );
            S::setpz4(
                yq_s8p.add(s * 7),
                S::mulpz4(wp7, S::addpz4(s04_mj_s26, w8_s15_mj_s37)),
            );

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
            let wp~K = S::cnjpz2(S::getpz2(&*twid(8, big_n, K, w, p)));
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

include!(concat!(env!("OUT_DIR"), "/dif8.rs"));

/// Initialize twiddles for subsequent forward and inverse Fourier transforms of size `n`.
/// `twiddles` must be of length `2*n`.
pub fn init_twiddles(n: usize, twiddles: &mut [c64]) {
    assert!(n.is_power_of_two());
    let i = n.trailing_zeros() as usize;
    assert!(i < MAX_EXP);
    assert_eq!(twiddles.len(), 2 * n);

    unsafe {
        crate::twiddles::init_wt(8, n, twiddles.as_mut_ptr());
    }
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
        let mut mem = dyn_stack::GlobalMemBuffer::new(crate::fft_scratch(n).unwrap());
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
        let mut mem = dyn_stack::GlobalMemBuffer::new(crate::fft_scratch(n).unwrap());
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
