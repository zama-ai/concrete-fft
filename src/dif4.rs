use crate::fft_simd::FftSimd16;
use crate::twiddles::{twid, twid_t};
use crate::{c64, MAX_EXP};

// forward butterfly
#[inline(always)]
pub(crate) unsafe fn fwdcore_s<S: FftSimd16>(
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    w: *const c64,
) {
    debug_assert_ne!(s, 1);

    let m = n / 4;
    let big_n = n * s;
    let big_n0 = 0;
    let big_n1 = big_n / 4;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;

    for p in 0..m {
        let sp = s * p;
        let s4p = 4 * sp;
        let w1p = S::duppz3(&*twid_t(4, big_n, 1, w, sp));
        let w2p = S::duppz3(&*twid_t(4, big_n, 2, w, sp));
        let w3p = S::duppz3(&*twid_t(4, big_n, 3, w, sp));

        let mut q = 0;
        loop {
            if q >= s {
                break;
            }

            let xq_sp = x.add(q + sp);
            let yq_s4p = y.add(q + s4p);

            let a = S::getpz2(xq_sp.add(big_n0));
            let c = S::getpz2(xq_sp.add(big_n2));
            let apc = S::addpz2(a, c);
            let amc = S::subpz2(a, c);

            let b = S::getpz2(xq_sp.add(big_n1));
            let d = S::getpz2(xq_sp.add(big_n3));
            let bpd = S::addpz2(b, d);
            let jbmd = S::jxpz2(S::subpz2(b, d));

            S::setpz2(yq_s4p.add(0), S::addpz2(apc, bpd));
            S::setpz2(yq_s4p.add(s), S::mulpz2(w1p, S::subpz2(amc, jbmd)));
            S::setpz2(yq_s4p.add(s * 2), S::mulpz2(w2p, S::subpz2(apc, bpd)));
            S::setpz2(yq_s4p.add(s * 3), S::mulpz2(w3p, S::addpz2(amc, jbmd)));

            q += 2;
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn fwdcore_1<S: FftSimd16>(
    big_n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    w: *const c64,
) {
    debug_assert_eq!(s, 1);
    let big_n0 = 0;
    let big_n1 = big_n / 4;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;

    let mut p = 0;
    loop {
        if p >= big_n1 {
            break;
        }

        let x_p = x.add(p);
        let y_4p = y.add(4 * p);

        let a = S::getpz2(x_p.add(big_n0));
        let c = S::getpz2(x_p.add(big_n2));
        let apc = S::addpz2(a, c);
        let amc = S::subpz2(a, c);

        let b = S::getpz2(x_p.add(big_n1));
        let d = S::getpz2(x_p.add(big_n3));
        let bpd = S::addpz2(b, d);
        let jbmd = S::jxpz2(S::subpz2(b, d));

        let w1p = S::getpz2(twid(4, big_n, 1, w, p));
        let w2p = S::getpz2(twid(4, big_n, 2, w, p));
        let w3p = S::getpz2(twid(4, big_n, 3, w, p));

        let aa = S::addpz2(apc, bpd);
        let bb = S::mulpz2(w1p, S::subpz2(amc, jbmd));
        let cc = S::mulpz2(w2p, S::subpz2(apc, bpd));
        let dd = S::mulpz2(w3p, S::addpz2(amc, jbmd));

        {
            let ab = S::catlo(aa, bb);
            S::setpz2(y_4p.add(0), ab);
            let cd = S::catlo(cc, dd);
            S::setpz2(y_4p.add(2), cd);
        }
        {
            let ab = S::cathi(aa, bb);
            S::setpz2(y_4p.add(4), ab);
            let cd = S::cathi(cc, dd);
            S::setpz2(y_4p.add(6), cd);
        }

        p += 2;
    }
}

#[inline(always)]
pub(crate) unsafe fn fwdend_4_s<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 4);
    debug_assert_ne!(s, 1);
    let z = if eo { y } else { x };

    let mut q = 0;
    loop {
        if q >= s {
            break;
        }

        let xq = x.add(q);
        let zq = z.add(q);

        let a = S::getpz2(xq.add(0));
        let b = S::getpz2(xq.add(s));
        let c = S::getpz2(xq.add(s * 2));
        let d = S::getpz2(xq.add(s * 3));

        let apc = S::addpz2(a, c);
        let amc = S::subpz2(a, c);
        let bpd = S::addpz2(b, d);
        let jbmd = S::jxpz2(S::subpz2(b, d));

        S::setpz2(zq.add(0), S::addpz2(apc, bpd));
        S::setpz2(zq.add(s), S::subpz2(amc, jbmd));
        S::setpz2(zq.add(s * 2), S::subpz2(apc, bpd));
        S::setpz2(zq.add(s * 3), S::addpz2(amc, jbmd));

        q += 2;
    }
}

#[inline(always)]
pub(crate) unsafe fn fwdend_4_1<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 4);
    debug_assert_eq!(s, 1);
    let z = if eo { y } else { x };

    let a = S::getpz(&*x.add(0));
    let b = S::getpz(&*x.add(1));
    let c = S::getpz(&*x.add(2));
    let d = S::getpz(&*x.add(3));

    let apc = S::addpz(a, c);
    let amc = S::subpz(a, c);
    let bpd = S::addpz(b, d);
    let jbmd = S::jxpz(S::subpz(b, d));

    S::setpz(z.add(0), S::addpz(apc, bpd));
    S::setpz(z.add(1), S::subpz(amc, jbmd));
    S::setpz(z.add(2), S::subpz(apc, bpd));
    S::setpz(z.add(3), S::addpz(amc, jbmd));
}

#[inline(always)]
pub(crate) unsafe fn fwdend_2_s<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 2);
    debug_assert_ne!(s, 1);
    let z = if eo { y } else { x };

    let mut q = 0;
    loop {
        if q >= s {
            break;
        }

        let xq = x.add(q);
        let zq = z.add(q);

        let a = S::getpz2(xq.add(0));
        let b = S::getpz2(xq.add(s));

        S::setpz2(zq.add(0), S::addpz2(a, b));
        S::setpz2(zq.add(s), S::subpz2(a, b));

        q += 2;
    }
}

#[inline(always)]
pub(crate) unsafe fn fwdend_2_1<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 2);
    debug_assert_eq!(s, 1);
    let z = if eo { y } else { x };

    let a = S::getpz(&*x.add(0));
    let b = S::getpz(&*x.add(1));

    S::setpz(z.add(0), S::addpz(a, b));
    S::setpz(z.add(1), S::subpz(a, b));
}

// backward butterfly
#[inline(always)]
pub(crate) unsafe fn invcore_s<S: FftSimd16>(
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    w: *const c64,
) {
    debug_assert_ne!(s, 1);

    let m = n / 4;
    let big_n = n * s;
    let big_n0 = 0;
    let big_n1 = big_n / 4;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;

    for p in 0..m {
        let sp = s * p;
        let s4p = 4 * sp;
        let w1p = S::duppz3(&*twid_t(4, big_n, 1, w, sp));
        let w2p = S::duppz3(&*twid_t(4, big_n, 2, w, sp));
        let w3p = S::duppz3(&*twid_t(4, big_n, 3, w, sp));

        let mut q = 0;
        loop {
            if q >= s {
                break;
            }

            let xq_sp = x.add(q + sp);
            let yq_s4p = y.add(q + s4p);

            let a = S::getpz2(xq_sp.add(big_n0));
            let c = S::getpz2(xq_sp.add(big_n2));
            let apc = S::addpz2(a, c);
            let amc = S::subpz2(a, c);

            let b = S::getpz2(xq_sp.add(big_n1));
            let d = S::getpz2(xq_sp.add(big_n3));
            let bpd = S::addpz2(b, d);
            let jbmd = S::jxpz2(S::subpz2(b, d));

            S::setpz2(yq_s4p.add(0), S::addpz2(apc, bpd));
            S::setpz2(yq_s4p.add(s), S::mulpz2(w1p, S::addpz2(amc, jbmd)));
            S::setpz2(yq_s4p.add(s * 2), S::mulpz2(w2p, S::subpz2(apc, bpd)));
            S::setpz2(yq_s4p.add(s * 3), S::mulpz2(w3p, S::subpz2(amc, jbmd)));

            q += 2;
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn invcore_1<S: FftSimd16>(
    big_n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    w: *const c64,
) {
    debug_assert_eq!(s, 1);
    let big_n0 = 0;
    let big_n1 = big_n / 4;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;

    let mut p = 0;
    loop {
        if p >= big_n1 {
            break;
        }

        let x_p = x.add(p);
        let y_4p = y.add(4 * p);

        let a = S::getpz2(x_p.add(big_n0));
        let c = S::getpz2(x_p.add(big_n2));
        let apc = S::addpz2(a, c);
        let amc = S::subpz2(a, c);

        let b = S::getpz2(x_p.add(big_n1));
        let d = S::getpz2(x_p.add(big_n3));
        let bpd = S::addpz2(b, d);
        let jbmd = S::jxpz2(S::subpz2(b, d));

        let w1p = S::getpz2(twid(4, big_n, 1, w, p));
        let w2p = S::getpz2(twid(4, big_n, 2, w, p));
        let w3p = S::getpz2(twid(4, big_n, 3, w, p));

        let aa = S::addpz2(apc, bpd);
        let bb = S::mulpz2(w1p, S::addpz2(amc, jbmd));
        let cc = S::mulpz2(w2p, S::subpz2(apc, bpd));
        let dd = S::mulpz2(w3p, S::subpz2(amc, jbmd));

        {
            let ab = S::catlo(aa, bb);
            S::setpz2(y_4p.add(0), ab);
            let cd = S::catlo(cc, dd);
            S::setpz2(y_4p.add(2), cd);
        }
        {
            let ab = S::cathi(aa, bb);
            S::setpz2(y_4p.add(4), ab);
            let cd = S::cathi(cc, dd);
            S::setpz2(y_4p.add(6), cd);
        }

        p += 2;
    }
}

#[inline(always)]
pub(crate) unsafe fn invend_4_s<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 4);
    debug_assert_ne!(s, 1);
    let z = if eo { y } else { x };

    let mut q = 0;
    loop {
        if q >= s {
            break;
        }

        let xq = x.add(q);
        let zq = z.add(q);

        let a = S::getpz2(xq.add(0));
        let b = S::getpz2(xq.add(s));
        let c = S::getpz2(xq.add(s * 2));
        let d = S::getpz2(xq.add(s * 3));

        let apc = S::addpz2(a, c);
        let amc = S::subpz2(a, c);
        let bpd = S::addpz2(b, d);
        let jbmd = S::jxpz2(S::subpz2(b, d));

        S::setpz2(zq.add(0), S::addpz2(apc, bpd));
        S::setpz2(zq.add(s), S::addpz2(amc, jbmd));
        S::setpz2(zq.add(s * 2), S::subpz2(apc, bpd));
        S::setpz2(zq.add(s * 3), S::subpz2(amc, jbmd));

        q += 2;
    }
}

#[inline(always)]
pub(crate) unsafe fn invend_4_1<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    debug_assert_eq!(n, 4);
    debug_assert_eq!(s, 1);
    let z = if eo { y } else { x };

    let a = S::getpz(&*x.add(0));
    let b = S::getpz(&*x.add(1));
    let c = S::getpz(&*x.add(2));
    let d = S::getpz(&*x.add(3));

    let apc = S::addpz(a, c);
    let amc = S::subpz(a, c);
    let bpd = S::addpz(b, d);
    let jbmd = S::jxpz(S::subpz(b, d));

    S::setpz(z.add(0), S::addpz(apc, bpd));
    S::setpz(z.add(1), S::addpz(amc, jbmd));
    S::setpz(z.add(2), S::subpz(apc, bpd));
    S::setpz(z.add(3), S::subpz(amc, jbmd));
}

#[inline(always)]
pub(crate) unsafe fn invend_2_s<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    fwdend_2_s::<S>(n, s, eo, x, y);
}

#[inline(always)]
pub(crate) unsafe fn invend_2_1<S: FftSimd16>(
    n: usize,
    s: usize,
    eo: bool,
    x: *mut c64,
    y: *mut c64,
) {
    fwdend_2_1::<S>(n, s, eo, x, y);
}

/// Initialize twiddles for subsequent forward and inverse Fourier transforms of size `n`.
/// `twiddles` must be of length `2*n`.
pub fn init_twiddles(forward: bool, n: usize, twiddles: &mut [c64]) {
    assert!(n.is_power_of_two());
    let i = n.trailing_zeros() as usize;
    assert!(i < MAX_EXP);
    assert_eq!(twiddles.len(), 2 * n);

    unsafe {
        crate::twiddles::init_wt(forward, 4, n, twiddles.as_mut_ptr());
    }
}

include!(concat!(env!("OUT_DIR"), "/dif4.rs"));
