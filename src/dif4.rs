use crate::fft_simd::FftSimd16;
use crate::twiddles::{twid, twid_t};
use crate::{c64, MAX_EXP};
use dyn_stack::DynStack;

// forward butterfly
#[inline(always)]
unsafe fn fwdcore_s<S: FftSimd16>(n: usize, s: usize, x: *mut c64, y: *mut c64, w: *const c64) {
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
        let w1p = S::duppz5(&*twid_t(4, big_n, 1, w, sp));
        let w2p = S::duppz5(&*twid_t(4, big_n, 2, w, sp));
        let w3p = S::duppz5(&*twid_t(4, big_n, 3, w, sp));

        let mut q = 0;
        loop {
            if q >= s {
                break;
            }

            let xq_sp = x.add(q + sp);
            let yq_s4p = y.add(q + s4p);

            let a = S::getpz4(xq_sp.add(big_n0));
            let c = S::getpz4(xq_sp.add(big_n2));
            let apc = S::addpz4(a, c);
            let amc = S::subpz4(a, c);

            let b = S::getpz4(xq_sp.add(big_n1));
            let d = S::getpz4(xq_sp.add(big_n3));
            let bpd = S::addpz4(b, d);
            let jbmd = S::jxpz4(S::subpz4(b, d));

            S::setpz4(yq_s4p.add(0), S::addpz4(apc, bpd));
            S::setpz4(yq_s4p.add(s), S::mulpz4(w1p, S::subpz4(amc, jbmd)));
            S::setpz4(yq_s4p.add(s * 2), S::mulpz4(w2p, S::subpz4(apc, bpd)));
            S::setpz4(yq_s4p.add(s * 3), S::mulpz4(w3p, S::addpz4(amc, jbmd)));

            q += 4;
        }
    }
}

#[inline(always)]
unsafe fn fwdcore_1<S: FftSimd16>(big_n: usize, s: usize, x: *mut c64, y: *mut c64, w: *const c64) {
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
unsafe fn fwdend_4_s<S: FftSimd16>(n: usize, s: usize, eo: bool, x: *mut c64, y: *mut c64) {
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
unsafe fn fwdend_4_1<S: FftSimd16>(n: usize, s: usize, eo: bool, x: *mut c64, y: *mut c64) {
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
unsafe fn fwdend_2_s<S: FftSimd16>(n: usize, s: usize, eo: bool, x: *mut c64, y: *mut c64) {
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
unsafe fn fwdend_2_1<S: FftSimd16>(n: usize, s: usize, eo: bool, x: *mut c64, y: *mut c64) {
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
unsafe fn invcore_s<S: FftSimd16>(n: usize, s: usize, x: *mut c64, y: *mut c64, w: *const c64) {
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
        let w1p = S::duppz5(&(&*twid_t(4, big_n, 1, w, sp)).conj());
        let w2p = S::duppz5(&(&*twid_t(4, big_n, 2, w, sp)).conj());
        let w3p = S::duppz5(&(&*twid_t(4, big_n, 3, w, sp)).conj());

        let mut q = 0;
        loop {
            if q >= s {
                break;
            }

            let xq_sp = x.add(q + sp);
            let yq_s4p = y.add(q + s4p);

            let a = S::getpz4(xq_sp.add(big_n0));
            let c = S::getpz4(xq_sp.add(big_n2));
            let apc = S::addpz4(a, c);
            let amc = S::subpz4(a, c);

            let b = S::getpz4(xq_sp.add(big_n1));
            let d = S::getpz4(xq_sp.add(big_n3));
            let bpd = S::addpz4(b, d);
            let jbmd = S::jxpz4(S::subpz4(b, d));

            S::setpz4(yq_s4p.add(0), S::addpz4(apc, bpd));
            S::setpz4(yq_s4p.add(s), S::mulpz4(w1p, S::addpz4(amc, jbmd)));
            S::setpz4(yq_s4p.add(s * 2), S::mulpz4(w2p, S::subpz4(apc, bpd)));
            S::setpz4(yq_s4p.add(s * 3), S::mulpz4(w3p, S::subpz4(amc, jbmd)));

            q += 4;
        }
    }
}

#[inline(always)]
unsafe fn invcore_1<S: FftSimd16>(big_n: usize, s: usize, x: *mut c64, y: *mut c64, w: *const c64) {
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

        let w1p = S::cnjpz2(S::getpz2(twid(4, big_n, 1, w, p)));
        let w2p = S::cnjpz2(S::getpz2(twid(4, big_n, 2, w, p)));
        let w3p = S::cnjpz2(S::getpz2(twid(4, big_n, 3, w, p)));

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
unsafe fn invend_4_s<S: FftSimd16>(n: usize, s: usize, eo: bool, x: *mut c64, y: *mut c64) {
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
unsafe fn invend_4_1<S: FftSimd16>(n: usize, s: usize, eo: bool, x: *mut c64, y: *mut c64) {
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
unsafe fn invend_2_s<S: FftSimd16>(n: usize, s: usize, eo: bool, x: *mut c64, y: *mut c64) {
    fwdend_2_s::<S>(n, s, eo, x, y);
}

#[inline(always)]
unsafe fn invend_2_1<S: FftSimd16>(n: usize, s: usize, eo: bool, x: *mut c64, y: *mut c64) {
    fwdend_2_1::<S>(n, s, eo, x, y);
}

include!(concat!(env!("OUT_DIR"), "/dif4.rs"));

macro_rules! impl_main_fn {
    ($(#[$attr: meta])? $name: ident, $array_expr: expr) => {
        pub fn $name(data: &mut [c64], twiddles: &[c64], stack: DynStack) {
            let n = data.len();
            let i = n.trailing_zeros() as usize;

            assert!(n.is_power_of_two());
            assert!(i < MAX_EXP);
            assert_eq!(twiddles.len(), 2 * n);

            let (mut scratch, _) = stack.make_aligned_uninit::<c64>(n, 64);
            let scratch = scratch.as_mut_ptr();
            let data = data.as_mut_ptr();
            let w = twiddles.as_ptr();

            unsafe {
                ($array_expr)[i](data, scratch as *mut c64, w);
            }
        }
    };
}

lazy_static::lazy_static! {}

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

        let mut arr_inv = arr_fwd.clone();
        let mut arr_fwd_expected = arr_fwd.clone();
        let mut arr_inv_expected = arr_fwd.clone();

        unsafe {
            let mut w = vec![z; 2 * n];

            crate::twiddles::init_wt(4, n, w.as_mut_ptr());
            let mut mem = dyn_stack::uninit_mem_in_global(crate::fft_scratch(n).unwrap());
            let mut stack = DynStack::new(&mut mem);

            fwd_fn(&mut arr_fwd, &w, stack.rb_mut());
            inv_fn(&mut arr_inv, &w, stack);
        }

        #[cfg(not(miri))]
        {
            let mut fft = rustfft::FftPlanner::new();
            let fwd = fft.plan_fft_forward(n);
            let inv = fft.plan_fft_inverse(n);

            fwd.process(&mut arr_fwd_expected);
            inv.process(&mut arr_inv_expected);

            use num_complex::ComplexFloat;
            for (actual, expected) in arr_fwd.iter().zip(&arr_fwd_expected) {
                assert!((*actual - *expected).abs() < 1e-9);
            }
            for (actual, expected) in arr_inv.iter().zip(&arr_inv_expected) {
                assert!((*actual - *expected).abs() < 1e-9);
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

        crate::init_twiddles(n, &mut w);
        let mut mem = dyn_stack::uninit_mem_in_global(crate::fft_scratch(n).unwrap());
        let mut stack = DynStack::new(&mut mem);

        fwd_fn(&mut arr_roundtrip, &w, stack.rb_mut());
        inv_fn(&mut arr_roundtrip, &w, stack);

        for z in &mut arr_roundtrip {
            *z /= n as f64;
        }

        use num_complex::ComplexFloat;
        for (actual, expected) in arr_roundtrip.iter().zip(&arr_orig) {
            assert!((*actual - *expected).abs() < 1e-9);
        }
    }

    #[test]
    fn test_fft() {
        for i in 0..crate::MAX_EXP {
            let n = 1usize << i;
            test_fft_generic(n, fwd_scalar, inv_scalar);

            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                #[cfg(target_feature = "fma")]
                test_fft_generic(n, fwd_fma, inv_fma);
                #[cfg(target_feature = "avx")]
                test_fft_generic(n, fwd_avx, inv_avx);
                #[cfg(target_feature = "sse3")]
                test_fft_generic(n, fwd_sse3, inv_sse3);
                #[cfg(target_feature = "sse2")]
                test_fft_generic(n, fwd_sse2, inv_sse2);
            }
        }
    }

    #[test]
    fn test_roundtrip() {
        for i in 0..crate::MAX_EXP {
            let n = 1usize << i;
            test_roundtrip_generic(n, fwd_scalar, inv_scalar);

            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                #[cfg(target_feature = "fma")]
                test_roundtrip_generic(n, fwd_fma, inv_fma);
                #[cfg(target_feature = "avx")]
                test_roundtrip_generic(n, fwd_avx, inv_avx);
                #[cfg(target_feature = "sse3")]
                test_roundtrip_generic(n, fwd_sse3, inv_sse3);
                #[cfg(target_feature = "sse2")]
                test_roundtrip_generic(n, fwd_sse2, inv_sse2);
            }
        }
    }
}
