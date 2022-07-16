use crate::c64;
use crate::fft_simd::{twid, twid_t, FftSimd64, FftSimd64X2, Scalar};

#[inline(always)]
unsafe fn core_<I: FftSimd64>(
    _fwd: bool,
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    w: *const c64,
) {
    debug_assert_eq!(s % I::COMPLEX_PER_REG, 0);

    let m = n / 2;
    let big_n = n * s;
    let big_n0 = 0;
    let big_n1 = big_n / 2;

    for p in 0..m {
        let sp = s * p;
        let s2p = 2 * sp;
        let w1p = I::splat(twid_t(2, big_n, 1, w, sp));

        let mut q = 0;
        while q < s {
            let xq_sp = x.add(q + sp);
            let yq_s2p = y.add(q + s2p);

            let a = I::load(yq_s2p.add(s * 0));
            let b = I::mul(w1p, I::load(yq_s2p.add(s * 1)));

            I::store(xq_sp.add(big_n0), I::add(a, b));
            I::store(xq_sp.add(big_n1), I::sub(a, b));

            q += I::COMPLEX_PER_REG;
        }
    }
}

#[inline(always)]
unsafe fn core_x2<I: FftSimd64X2>(
    _fwd: bool,
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    w: *const c64,
) {
    debug_assert_eq!(s, 1);

    let big_n = n;
    let big_n0 = 0;
    let big_n1 = big_n / 2;

    let mut p = 0;
    while p < big_n1 {
        let x_p = x.add(p);
        let y_2p = y.add(2 * p);

        let w1p = I::load(twid(2, big_n, 1, w, p));

        let ab0 = I::load(y_2p.add(0));
        let ab1 = I::load(y_2p.add(2));

        let a = I::catlo(ab0, ab1);
        let b = I::mul(w1p, I::cathi(ab0, ab1));

        I::store(x_p.add(big_n0), I::add(a, b));
        I::store(x_p.add(big_n1), I::sub(a, b));

        p += 2;
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

macro_rules! dit2_impl {
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
                    end_2::<$xn>(FWD, 1 << 1, 1 << 1, y, x, true);
                    $core1______(FWD, 1 << 2, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_03<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 2, x, y, false);
                    core_::<$xn>(FWD, 1 << 2, 1 << 1, y, x, w);
                    $core1______(FWD, 1 << 3, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_04<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 3, y, x, true);
                    core_::<$xn>(FWD, 1 << 2, 1 << 2, x, y, w);
                    core_::<$xn>(FWD, 1 << 3, 1 << 1, y, x, w);
                    $core1______(FWD, 1 << 4, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_05<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 4, x, y, false);
                    core_::<$xn>(FWD, 1 << 2, 1 << 3, y, x, w);
                    core_::<$xn>(FWD, 1 << 3, 1 << 2, x, y, w);
                    core_::<$xn>(FWD, 1 << 4, 1 << 1, y, x, w);
                    $core1______(FWD, 1 << 5, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_06<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 5, y, x, true);
                    core_::<$xn>(FWD, 1 << 2, 1 << 4, x, y, w);
                    core_::<$xn>(FWD, 1 << 3, 1 << 3, y, x, w);
                    core_::<$xn>(FWD, 1 << 4, 1 << 2, x, y, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 1, y, x, w);
                    $core1______(FWD, 1 << 6, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_07<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 6, x, y, false);
                    core_::<$xn>(FWD, 1 << 2, 1 << 5, y, x, w);
                    core_::<$xn>(FWD, 1 << 3, 1 << 4, x, y, w);
                    core_::<$xn>(FWD, 1 << 4, 1 << 3, y, x, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 2, x, y, w);
                    core_::<$xn>(FWD, 1 << 6, 1 << 1, y, x, w);
                    $core1______(FWD, 1 << 7, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_08<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 7, y, x, true);
                    core_::<$xn>(FWD, 1 << 2, 1 << 6, x, y, w);
                    core_::<$xn>(FWD, 1 << 3, 1 << 5, y, x, w);
                    core_::<$xn>(FWD, 1 << 4, 1 << 4, x, y, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 3, y, x, w);
                    core_::<$xn>(FWD, 1 << 6, 1 << 2, x, y, w);
                    core_::<$xn>(FWD, 1 << 7, 1 << 1, y, x, w);
                    $core1______(FWD, 1 << 8, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_09<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 8, x, y, false);
                    core_::<$xn>(FWD, 1 << 2, 1 << 7, y, x, w);
                    core_::<$xn>(FWD, 1 << 3, 1 << 6, x, y, w);
                    core_::<$xn>(FWD, 1 << 4, 1 << 5, y, x, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 4, x, y, w);
                    core_::<$xn>(FWD, 1 << 6, 1 << 3, y, x, w);
                    core_::<$xn>(FWD, 1 << 7, 1 << 2, x, y, w);
                    core_::<$xn>(FWD, 1 << 8, 1 << 1, y, x, w);
                    $core1______(FWD, 1 << 9, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_10<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 9, y, x, true);
                    core_::<$xn>(FWD, 1 << 2, 1 << 8, x, y, w);
                    core_::<$xn>(FWD, 1 << 3, 1 << 7, y, x, w);
                    core_::<$xn>(FWD, 1 << 4, 1 << 6, x, y, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 5, y, x, w);
                    core_::<$xn>(FWD, 1 << 6, 1 << 4, x, y, w);
                    core_::<$xn>(FWD, 1 << 7, 1 << 3, y, x, w);
                    core_::<$xn>(FWD, 1 << 8, 1 << 2, x, y, w);
                    core_::<$xn>(FWD, 1 << 9, 1 << 1, y, x, w);
                    $core1______(FWD, 1 << 10, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_11<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 10, x, y, false);
                    core_::<$xn>(FWD, 1 << 2, 1 << 09, y, x, w);
                    core_::<$xn>(FWD, 1 << 3, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 4, 1 << 07, y, x, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 6, 1 << 05, y, x, w);
                    core_::<$xn>(FWD, 1 << 7, 1 << 04, x, y, w);
                    core_::<$xn>(FWD, 1 << 8, 1 << 03, y, x, w);
                    core_::<$xn>(FWD, 1 << 9, 1 << 02, x, y, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 01, y, x, w);
                    $core1______(FWD, 1 << 11, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_12<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 11, y, x, true);
                    core_::<$xn>(FWD, 1 << 2, 1 << 10, x, y, w);
                    core_::<$xn>(FWD, 1 << 3, 1 << 09, y, x, w);
                    core_::<$xn>(FWD, 1 << 4, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 07, y, x, w);
                    core_::<$xn>(FWD, 1 << 6, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 7, 1 << 05, y, x, w);
                    core_::<$xn>(FWD, 1 << 8, 1 << 04, x, y, w);
                    core_::<$xn>(FWD, 1 << 9, 1 << 03, y, x, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 02, x, y, w);
                    core_::<$xn>(FWD, 1 << 11, 1 << 01, y, x, w);
                    $core1______(FWD, 1 << 12, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_13<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 12, x, y, false);
                    core_::<$xn>(FWD, 1 << 2, 1 << 11, y, x, w);
                    core_::<$xn>(FWD, 1 << 3, 1 << 10, x, y, w);
                    core_::<$xn>(FWD, 1 << 4, 1 << 09, y, x, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 6, 1 << 07, y, x, w);
                    core_::<$xn>(FWD, 1 << 7, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 8, 1 << 05, y, x, w);
                    core_::<$xn>(FWD, 1 << 9, 1 << 04, x, y, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 03, y, x, w);
                    core_::<$xn>(FWD, 1 << 11, 1 << 02, x, y, w);
                    core_::<$xn>(FWD, 1 << 12, 1 << 01, y, x, w);
                    $core1______(FWD, 1 << 13, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_14<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 13, y, x, true);
                    core_::<$xn>(FWD, 1 << 2, 1 << 12, x, y, w);
                    core_::<$xn>(FWD, 1 << 3, 1 << 11, y, x, w);
                    core_::<$xn>(FWD, 1 << 4, 1 << 10, x, y, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 09, y, x, w);
                    core_::<$xn>(FWD, 1 << 6, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 7, 1 << 07, y, x, w);
                    core_::<$xn>(FWD, 1 << 8, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 9, 1 << 05, y, x, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 04, x, y, w);
                    core_::<$xn>(FWD, 1 << 11, 1 << 03, y, x, w);
                    core_::<$xn>(FWD, 1 << 12, 1 << 02, x, y, w);
                    core_::<$xn>(FWD, 1 << 13, 1 << 01, y, x, w);
                    $core1______(FWD, 1 << 14, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_15<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 14, x, y, false);
                    core_::<$xn>(FWD, 1 << 2, 1 << 13, y, x, w);
                    core_::<$xn>(FWD, 1 << 3, 1 << 12, x, y, w);
                    core_::<$xn>(FWD, 1 << 4, 1 << 11, y, x, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 10, x, y, w);
                    core_::<$xn>(FWD, 1 << 6, 1 << 09, y, x, w);
                    core_::<$xn>(FWD, 1 << 7, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 8, 1 << 07, y, x, w);
                    core_::<$xn>(FWD, 1 << 9, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 05, y, x, w);
                    core_::<$xn>(FWD, 1 << 11, 1 << 04, x, y, w);
                    core_::<$xn>(FWD, 1 << 12, 1 << 03, y, x, w);
                    core_::<$xn>(FWD, 1 << 13, 1 << 02, x, y, w);
                    core_::<$xn>(FWD, 1 << 14, 1 << 01, y, x, w);
                    $core1______(FWD, 1 << 15, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_16<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 15, y, x, true);
                    core_::<$xn>(FWD, 1 << 2, 1 << 14, x, y, w);
                    core_::<$xn>(FWD, 1 << 3, 1 << 13, y, x, w);
                    core_::<$xn>(FWD, 1 << 4, 1 << 12, x, y, w);
                    core_::<$xn>(FWD, 1 << 5, 1 << 11, y, x, w);
                    core_::<$xn>(FWD, 1 << 6, 1 << 10, x, y, w);
                    core_::<$xn>(FWD, 1 << 7, 1 << 09, y, x, w);
                    core_::<$xn>(FWD, 1 << 8, 1 << 08, x, y, w);
                    core_::<$xn>(FWD, 1 << 9, 1 << 07, y, x, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 11, 1 << 05, y, x, w);
                    core_::<$xn>(FWD, 1 << 12, 1 << 04, x, y, w);
                    core_::<$xn>(FWD, 1 << 13, 1 << 03, y, x, w);
                    core_::<$xn>(FWD, 1 << 14, 1 << 02, x, y, w);
                    core_::<$xn>(FWD, 1 << 15, 1 << 01, y, x, w);
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

dit2_impl! {
    pub static DIT2_SCALAR = Fft {
        core_1: core_::<Scalar>,
        native: Scalar,
        x1: Scalar,
    };

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub static DIT2_AVX = Fft {
        core_1: core_x2::<AvxX2>,
        native: AvxX2,
        x1: AvxX1,
        target: "avx",
    };

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub static DIT2_FMA = Fft {
        core_1: core_x2::<FmaX2>,
        native: FmaX2,
        x1: FmaX1,
        target: "fma",
    };
}

pub(crate) fn runtime_fft() -> crate::FftImpl {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if is_x86_feature_detected!("fma") {
        return DIT2_FMA;
    } else if is_x86_feature_detected!("avx") {
        return DIT2_AVX;
    }

    DIT2_SCALAR
}
