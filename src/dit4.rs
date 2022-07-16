use crate::c64;
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

    let m = n / 4;
    let big_n = n * s;
    let big_n0 = 0;
    let big_n1 = big_n / 4;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;

    for p in 0..m {
        let sp = s * p;
        let s4p = 4 * sp;
        let w1p = I::splat(twid_t(4, big_n, 1, w, sp));
        let w2p = I::splat(twid_t(4, big_n, 2, w, sp));
        let w3p = I::splat(twid_t(4, big_n, 3, w, sp));

        let mut q = 0;
        while q < s {
            let xq_sp = x.add(q + sp);
            let yq_s4p = y.add(q + s4p);

            let a = I::load(yq_s4p.add(0));
            let b = I::mul(w1p, I::load(yq_s4p.add(s)));
            let c = I::mul(w2p, I::load(yq_s4p.add(s * 2)));
            let d = I::mul(w3p, I::load(yq_s4p.add(s * 3)));

            let apc = I::add(a, c);
            let amc = I::sub(a, c);

            let bpd = I::add(b, d);
            let jbmd = I::xpj(fwd, I::sub(b, d));

            I::store(xq_sp.add(big_n0), I::add(apc, bpd));
            I::store(xq_sp.add(big_n1), I::sub(amc, jbmd));
            I::store(xq_sp.add(big_n2), I::sub(apc, bpd));
            I::store(xq_sp.add(big_n3), I::add(amc, jbmd));

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
    let big_n1 = big_n / 4;
    let big_n2 = big_n1 * 2;
    let big_n3 = big_n1 * 3;

    let mut p = 0;
    while p < big_n1 {
        let x_p = x.add(p);
        let y_4p = y.add(4 * p);

        let w1p = I::load(twid(4, big_n, 1, w, p));
        let w2p = I::load(twid(4, big_n, 2, w, p));
        let w3p = I::load(twid(4, big_n, 3, w, p));

        let ab0 = I::load(y_4p.add(0));
        let cd0 = I::load(y_4p.add(2));
        let ab1 = I::load(y_4p.add(4));
        let cd1 = I::load(y_4p.add(6));

        let a = I::catlo(ab0, ab1);
        let b = I::mul(w1p, I::cathi(ab0, ab1));
        let c = I::mul(w2p, I::catlo(cd0, cd1));
        let d = I::mul(w3p, I::cathi(cd0, cd1));

        let apc = I::add(a, c);
        let amc = I::sub(a, c);
        let bpd = I::add(b, d);
        let jbmd = I::xpj(fwd, I::sub(b, d));

        I::store(x_p.add(big_n0), I::add(apc, bpd));
        I::store(x_p.add(big_n1), I::sub(amc, jbmd));
        I::store(x_p.add(big_n2), I::sub(apc, bpd));
        I::store(x_p.add(big_n3), I::add(amc, jbmd));

        p += 2;
    }
}

#[inline(always)]
pub unsafe fn end_4<I: FftSimd64>(
    fwd: bool,
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    eo: bool,
) {
    debug_assert_eq!(n, 4);
    debug_assert_eq!(s % I::COMPLEX_PER_REG, 0);
    let z = if eo { y } else { x };

    let mut q = 0;
    while q < s {
        let xq = x.add(q);
        let zq = z.add(q);

        let a = I::load(zq.add(0));
        let b = I::load(zq.add(s));
        let c = I::load(zq.add(s * 2));
        let d = I::load(zq.add(s * 3));

        let apc = I::add(a, c);
        let amc = I::sub(a, c);
        let bpd = I::add(b, d);
        let jbmd = I::xpj(fwd, I::sub(b, d));

        I::store(xq.add(0), I::add(apc, bpd));
        I::store(xq.add(s), I::sub(amc, jbmd));
        I::store(xq.add(s * 2), I::sub(apc, bpd));
        I::store(xq.add(s * 3), I::add(amc, jbmd));

        q += I::COMPLEX_PER_REG;
    }
}

#[inline(always)]
pub unsafe fn end_2<I: FftSimd64>(
    _fwd: bool,
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    eo: bool,
) {
    debug_assert_eq!(n, 2);
    debug_assert_eq!(s % I::COMPLEX_PER_REG, 0);
    let z = if eo { y } else { x };

    let mut q = 0;
    while q < s {
        let xq = x.add(q);
        let zq = z.add(q);

        let a = I::load(zq.add(0));
        let b = I::load(zq.add(s));

        I::store(xq.add(0), I::add(a, b));
        I::store(xq.add(s), I::sub(a, b));

        q += I::COMPLEX_PER_REG;
    }
}

macro_rules! dit4_impl {
    (
        $(
            $(#[$attr: meta])*
            pub static $fft: ident = Fft {
                core_1: $core1______: expr,
                native: $xn: ty,
                x1: $x1: ty,
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
                unsafe fn fft_00<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {}
                unsafe fn fft_01<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$x1>(FWD, 1 << 1, 1 << 0, x, y, false);
                }
                unsafe fn fft_02<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$x1>(FWD, 1 << 2, 1 << 0, x, y, false);
                }
                unsafe fn fft_03<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 2, y, x, true);
                    $core1______(FWD, 1 << 3, 1 << 0, x, y, w);
                }
                unsafe fn fft_04<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 2, 1 << 2, y, x, true);
                    $core1______(FWD, 1 << 4, 1 << 0, x, y, w);
                }
                unsafe fn fft_05<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 4, x, y, false);
                    core_::<$xn>(FWD, 1 << 3, 1 << 2, y, x, w);
                    $core1______(FWD, 1 << 5, 1 << 0, x, y, w);
                }
                unsafe fn fft_06<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 2, 1 << 4, x, y, false);
                    core_::<$xn>(FWD, 1 << 4, 1 << 2, y, x, w);
                    $core1______(FWD, 1 << 6, 1 << 0, x, y, w);
                }
                unsafe fn fft_07<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 6, y, x, true);
                    core_::<$xn>(FWD, 1 << 3, 1 << 4, x, y, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 2, y, x, w);
                    $core1______(FWD, 1 << 7, 1 << 0, x, y, w);
                }
                unsafe fn fft_08<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 2, 1 << 6, y, x, true);
                    core_::<$xn>(FWD, 1 << 4, 1 << 4, x, y, w);
                    core_::<$xn>(FWD, 1 << 6, 1 << 2, y, x, w);
                    $core1______(FWD, 1 << 8, 1 << 0, x, y, w);
                }
                unsafe fn fft_09<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 8, x, y, false);
                    core_::<$xn>(FWD, 1 << 3, 1 << 6, y, x, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 4, x, y, w);
                    core_::<$xn>(FWD, 1 << 7, 1 << 2, y, x, w);
                    $core1______(FWD, 1 << 9, 1 << 0, x, y, w);
                }
                unsafe fn fft_10<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 02, 1 << 8, x, y, false);
                    core_::<$xn>(FWD, 1 << 04, 1 << 6, y, x, w);
                    core_::<$xn>(FWD, 1 << 06, 1 << 4, x, y, w);
                    core_::<$xn>(FWD, 1 << 08, 1 << 2, y, x, w);
                    $core1______(FWD, 1 << 10, 1 << 0, x, y, w);
                }
                unsafe fn fft_11<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 01, 1 << 10, y, x, true);
                    core_::<$xn>(FWD, 1 << 03, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 05, 1 << 06, y, x, w);
                    core_::<$xn>(FWD, 1 << 07, 1 << 04, x, y, w);
                    core_::<$xn>(FWD, 1 << 09, 1 << 02, y, x, w);
                    $core1______(FWD, 1 << 11, 1 << 00, x, y, w);
                }
                unsafe fn fft_12<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 02, 1 << 10, y, x, true);
                    core_::<$xn>(FWD, 1 << 04, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 06, 1 << 06, y, x, w);
                    core_::<$xn>(FWD, 1 << 08, 1 << 04, x, y, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 02, y, x, w);
                    $core1______(FWD, 1 << 12, 1 << 00, x, y, w);
                }
                unsafe fn fft_13<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 01, 1 << 12, x, y, false);
                    core_::<$xn>(FWD, 1 << 03, 1 << 10, y, x, w);
                    core_::<$xn>(FWD, 1 << 05, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 07, 1 << 06, y, x, w);
                    core_::<$xn>(FWD, 1 << 09, 1 << 04, x, y, w);
                    core_::<$xn>(FWD, 1 << 11, 1 << 02, y, x, w);
                    $core1______(FWD, 1 << 13, 1 << 00, x, y, w);
                }
                unsafe fn fft_14<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 02, 1 << 12, x, y, false);
                    core_::<$xn>(FWD, 1 << 04, 1 << 10, y, x, w);
                    core_::<$xn>(FWD, 1 << 06, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 08, 1 << 06, y, x, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 04, x, y, w);
                    core_::<$xn>(FWD, 1 << 12, 1 << 02, y, x, w);
                    $core1______(FWD, 1 << 14, 1 << 00, x, y, w);
                }
                unsafe fn fft_15<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 01, 1 << 14, y, x, true);
                    core_::<$xn>(FWD, 1 << 03, 1 << 12, x, y, w);
                    core_::<$xn>(FWD, 1 << 05, 1 << 10, y, x, w);
                    core_::<$xn>(FWD, 1 << 07, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 09, 1 << 06, y, x, w);
                    core_::<$xn>(FWD, 1 << 11, 1 << 04, x, y, w);
                    core_::<$xn>(FWD, 1 << 13, 1 << 02, y, x, w);
                    $core1______(FWD, 1 << 15, 1 << 00, x, y, w);
                }
                unsafe fn fft_16<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 02, 1 << 14, y, x, true);
                    core_::<$xn>(FWD, 1 << 04, 1 << 12, x, y, w);
                    core_::<$xn>(FWD, 1 << 06, 1 << 10, y, x, w);
                    core_::<$xn>(FWD, 1 << 08, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 06, y, x, w);
                    core_::<$xn>(FWD, 1 << 12, 1 << 04, x, y, w);
                    core_::<$xn>(FWD, 1 << 14, 1 << 02, y, x, w);
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

dit4_impl! {
    pub static DIT4_SCALAR = Fft {
        core_1: core_::<Scalar>,
        native: Scalar,
        x1: Scalar,
    };

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub static DIT4_AVX = Fft {
        core_1: core_x2::<AvxX2>,
        native: AvxX2,
        x1: AvxX1,
    };

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub static DIT4_FMA = Fft {
        core_1: core_x2::<AvxX2>,
        native: FmaX2,
        x1: FmaX1,
    };
}

pub(crate) fn runtime_fft() -> crate::FftImpl {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if is_x86_feature_detected!("fma") {
        return DIT4_FMA;
    } else if is_x86_feature_detected!("avx") {
        return DIT4_AVX;
    }

    DIT4_SCALAR
}
