use crate::c64;
use crate::dit4::{end_2, end_4};
use crate::fft_simd::{twid, twid_t, FftSimd64, FftSimd64Ext, FftSimd64X2, FftSimd64X4, Scalar};

#[inline(always)]
#[rustfmt::skip]
unsafe fn core_<I: FftSimd64>(
    fwd: bool,
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    w: *const c64,
) {
    debug_assert_eq!(s % I::COMPLEX_PER_REG, 0);

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
        let w1p = I::splat(twid_t(8, big_n, 1, w, sp));
        let w2p = I::splat(twid_t(8, big_n, 2, w, sp));
        let w3p = I::splat(twid_t(8, big_n, 3, w, sp));
        let w4p = I::splat(twid_t(8, big_n, 4, w, sp));
        let w5p = I::splat(twid_t(8, big_n, 5, w, sp));
        let w6p = I::splat(twid_t(8, big_n, 6, w, sp));
        let w7p = I::splat(twid_t(8, big_n, 7, w, sp));

        let mut q = 0;
        while q < s {
            let xq_sp = x.add(q + sp);
            let yq_s8p = y.add(q + s8p);

            let y0 = I::load(yq_s8p.add(0));
            let y1 = I::mul(w1p, I::load(yq_s8p.add(s)));
            let y2 = I::mul(w2p, I::load(yq_s8p.add(s * 2)));
            let y3 = I::mul(w3p, I::load(yq_s8p.add(s * 3)));
            let y4 = I::mul(w4p, I::load(yq_s8p.add(s * 4)));
            let y5 = I::mul(w5p, I::load(yq_s8p.add(s * 5)));
            let y6 = I::mul(w6p, I::load(yq_s8p.add(s * 6)));
            let y7 = I::mul(w7p, I::load(yq_s8p.add(s * 7)));
            let a04 = I::add(y0, y4);
            let s04 = I::sub(y0, y4);
            let a26 = I::add(y2, y6);
            let js26 = I::xpj(fwd, I::sub(y2, y6));
            let a15 = I::add(y1, y5);
            let s15 = I::sub(y1, y5);
            let a37 = I::add(y3, y7);
            let js37 = I::xpj(fwd, I::sub(y3, y7));

            let a04_p1_a26 = I::add(a04, a26);
            let a15_p1_a37 = I::add(a15, a37);
            I::store(xq_sp.add(big_n0), I::add(a04_p1_a26, a15_p1_a37));
            I::store(xq_sp.add(big_n4), I::sub(a04_p1_a26, a15_p1_a37));

            let s04_mj_s26 = I::sub(s04, js26);
            let w8_s15_mj_s37 = I::xw8(fwd, I::sub(s15, js37));
            I::store(xq_sp.add(big_n1), I::add(s04_mj_s26, w8_s15_mj_s37));
            I::store(xq_sp.add(big_n5), I::sub(s04_mj_s26, w8_s15_mj_s37));

            let a04_m1_a26 = I::sub(a04, a26);
            let j_a15_m1_a37 = I::xpj(fwd, I::sub(a15, a37));
            I::store(xq_sp.add(big_n2), I::sub(a04_m1_a26, j_a15_m1_a37));
            I::store(xq_sp.add(big_n6), I::add(a04_m1_a26, j_a15_m1_a37));

            let s04_pj_s26 = I::add(s04, js26);
            let v8_s15_pj_s37 = I::xv8(fwd, I::add(s15, js37));
            I::store(xq_sp.add(big_n3), I::sub(s04_pj_s26, v8_s15_pj_s37));
            I::store(xq_sp.add(big_n7), I::add(s04_pj_s26, v8_s15_pj_s37));

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
    debug_assert_eq!(I::COMPLEX_PER_REG, 2);

    let big_n = n;
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

        let w1p = I::load(twid(8, big_n, 1, w, p));
        let w2p = I::load(twid(8, big_n, 2, w, p));
        let w3p = I::load(twid(8, big_n, 3, w, p));
        let w4p = I::load(twid(8, big_n, 4, w, p));
        let w5p = I::load(twid(8, big_n, 5, w, p));
        let w6p = I::load(twid(8, big_n, 6, w, p));
        let w7p = I::load(twid(8, big_n, 7, w, p));
        let ab_0 = I::load(y_8p.add(0));
        let cd_0 = I::load(y_8p.add(2));
        let ef_0 = I::load(y_8p.add(4));
        let gh_0 = I::load(y_8p.add(6));
        let ab_1 = I::load(y_8p.add(8));
        let cd_1 = I::load(y_8p.add(10));
        let ef_1 = I::load(y_8p.add(12));
        let gh_1 = I::load(y_8p.add(14));
        let y0 = I::catlo(ab_0, ab_1);
        let y1 = I::mul(w1p, I::cathi(ab_0, ab_1));
        let y2 = I::mul(w2p, I::catlo(cd_0, cd_1));
        let y3 = I::mul(w3p, I::cathi(cd_0, cd_1));
        let y4 = I::mul(w4p, I::catlo(ef_0, ef_1));
        let y5 = I::mul(w5p, I::cathi(ef_0, ef_1));
        let y6 = I::mul(w6p, I::catlo(gh_0, gh_1));
        let y7 = I::mul(w7p, I::cathi(gh_0, gh_1));

        let a04 = I::add(y0, y4);
        let s04 = I::sub(y0, y4);
        let a26 = I::add(y2, y6);
        let js26 = I::xpj(fwd, I::sub(y2, y6));
        let a15 = I::add(y1, y5);
        let s15 = I::sub(y1, y5);
        let a37 = I::add(y3, y7);
        let js37 = I::xpj(fwd, I::sub(y3, y7));

        let a04_p1_a26 = I::add(a04, a26);
        let a15_p1_a37 = I::add(a15, a37);
        I::store(x_p.add(big_n0), I::add(a04_p1_a26, a15_p1_a37));
        I::store(x_p.add(big_n4), I::sub(a04_p1_a26, a15_p1_a37));

        let s04_mj_s26 = I::sub(s04, js26);
        let w8_s15_mj_s37 = I::xw8(fwd, I::sub(s15, js37));
        I::store(x_p.add(big_n1), I::add(s04_mj_s26, w8_s15_mj_s37));
        I::store(x_p.add(big_n5), I::sub(s04_mj_s26, w8_s15_mj_s37));

        let a04_m1_a26 = I::sub(a04, a26);
        let j_a15_m1_a37 = I::xpj(fwd, I::sub(a15, a37));
        I::store(x_p.add(big_n2), I::sub(a04_m1_a26, j_a15_m1_a37));
        I::store(x_p.add(big_n6), I::add(a04_m1_a26, j_a15_m1_a37));

        let s04_pj_s26 = I::add(s04, js26);
        let v8_s15_pj_s37 = I::xv8(fwd, I::add(s15, js37));
        I::store(x_p.add(big_n3), I::sub(s04_pj_s26, v8_s15_pj_s37));
        I::store(x_p.add(big_n7), I::add(s04_pj_s26, v8_s15_pj_s37));

        p += 2;
    }
}

#[inline(always)]
unsafe fn core_x4<I2: FftSimd64X2, I4: FftSimd64X4>(
    fwd: bool,
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    w: *const c64,
) {
    debug_assert_eq!(s, 1);
    if n == 16 {
        return core_x2::<I2>(fwd, n, s, x, y, w);
    }

    let big_n = n;
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

        let w1p = I4::load(twid(8, big_n, 1, w, p));
        let w2p = I4::load(twid(8, big_n, 2, w, p));
        let w3p = I4::load(twid(8, big_n, 3, w, p));
        let w4p = I4::load(twid(8, big_n, 4, w, p));
        let w5p = I4::load(twid(8, big_n, 5, w, p));
        let w6p = I4::load(twid(8, big_n, 6, w, p));
        let w7p = I4::load(twid(8, big_n, 7, w, p));

        let abcd_0 = I4::load(y_8p.add(0));
        let efgh_0 = I4::load(y_8p.add(4));
        let abcd_1 = I4::load(y_8p.add(8));
        let efgh_1 = I4::load(y_8p.add(12));
        let abcd_2 = I4::load(y_8p.add(16));
        let efgh_2 = I4::load(y_8p.add(20));
        let abcd_3 = I4::load(y_8p.add(24));
        let efgh_3 = I4::load(y_8p.add(28));

        let (a, b, c, d) = I4::transpose(abcd_0, abcd_1, abcd_2, abcd_3);
        let (e, f, g, h) = I4::transpose(efgh_0, efgh_1, efgh_2, efgh_3);

        let y0 = a;
        let y1 = I4::mul(w1p, b);
        let y2 = I4::mul(w2p, c);
        let y3 = I4::mul(w3p, d);
        let y4 = I4::mul(w4p, e);
        let y5 = I4::mul(w5p, f);
        let y6 = I4::mul(w6p, g);
        let y7 = I4::mul(w7p, h);

        let a04 = I4::add(y0, y4);
        let s04 = I4::sub(y0, y4);
        let a26 = I4::add(y2, y6);
        let js26 = I4::xpj(fwd, I4::sub(y2, y6));
        let a15 = I4::add(y1, y5);
        let s15 = I4::sub(y1, y5);
        let a37 = I4::add(y3, y7);
        let js37 = I4::xpj(fwd, I4::sub(y3, y7));

        let a04_p1_a26 = I4::add(a04, a26);
        let a15_p1_a37 = I4::add(a15, a37);
        I4::store(x_p.add(big_n0), I4::add(a04_p1_a26, a15_p1_a37));
        I4::store(x_p.add(big_n4), I4::sub(a04_p1_a26, a15_p1_a37));

        let s04_mj_s26 = I4::sub(s04, js26);
        let w8_s15_mj_s37 = I4::xw8(fwd, I4::sub(s15, js37));
        I4::store(x_p.add(big_n1), I4::add(s04_mj_s26, w8_s15_mj_s37));
        I4::store(x_p.add(big_n5), I4::sub(s04_mj_s26, w8_s15_mj_s37));

        let a04_m1_a26 = I4::sub(a04, a26);
        let j_a15_m1_a37 = I4::xpj(fwd, I4::sub(a15, a37));
        I4::store(x_p.add(big_n2), I4::sub(a04_m1_a26, j_a15_m1_a37));
        I4::store(x_p.add(big_n6), I4::add(a04_m1_a26, j_a15_m1_a37));

        let s04_pj_s26 = I4::add(s04, js26);
        let v8_s15_pj_s37 = I4::xv8(fwd, I4::add(s15, js37));
        I4::store(x_p.add(big_n3), I4::sub(s04_pj_s26, v8_s15_pj_s37));
        I4::store(x_p.add(big_n7), I4::add(s04_pj_s26, v8_s15_pj_s37));

        p += 4;
    }
}

#[inline(always)]
pub(crate) unsafe fn end_8<I: FftSimd64>(
    fwd: bool,
    n: usize,
    s: usize,
    x: *mut c64,
    y: *mut c64,
    eo: bool,
) {
    debug_assert_eq!(n, 8);
    debug_assert_eq!(s % I::COMPLEX_PER_REG, 0);

    let z = if eo { y } else { x };

    let mut q = 0;
    while q < s {
        let xq = x.add(q);
        let zq = z.add(q);

        let z0 = I::load(zq.add(0));
        let z1 = I::load(zq.add(s));
        let z2 = I::load(zq.add(s * 2));
        let z3 = I::load(zq.add(s * 3));
        let z4 = I::load(zq.add(s * 4));
        let z5 = I::load(zq.add(s * 5));
        let z6 = I::load(zq.add(s * 6));
        let z7 = I::load(zq.add(s * 7));
        let a04 = I::add(z0, z4);
        let s04 = I::sub(z0, z4);
        let a26 = I::add(z2, z6);
        let js26 = I::xpj(fwd, I::sub(z2, z6));
        let a15 = I::add(z1, z5);
        let s15 = I::sub(z1, z5);
        let a37 = I::add(z3, z7);
        let js37 = I::xpj(fwd, I::sub(z3, z7));
        let a04_p1_a26 = I::add(a04, a26);
        let s04_mj_s26 = I::sub(s04, js26);
        let a04_m1_a26 = I::sub(a04, a26);
        let s04_pj_s26 = I::add(s04, js26);
        let a15_p1_a37 = I::add(a15, a37);
        let w8_s15_mj_s37 = I::xw8(fwd, I::sub(s15, js37));
        let j_a15_m1_a37 = I::xpj(fwd, I::sub(a15, a37));
        let v8_s15_pj_s37 = I::xv8(fwd, I::add(s15, js37));
        I::store(xq.add(0), I::add(a04_p1_a26, a15_p1_a37));
        I::store(xq.add(s), I::add(s04_mj_s26, w8_s15_mj_s37));
        I::store(xq.add(s * 2), I::sub(a04_m1_a26, j_a15_m1_a37));
        I::store(xq.add(s * 3), I::sub(s04_pj_s26, v8_s15_pj_s37));
        I::store(xq.add(s * 4), I::sub(a04_p1_a26, a15_p1_a37));
        I::store(xq.add(s * 5), I::sub(s04_mj_s26, w8_s15_mj_s37));
        I::store(xq.add(s * 6), I::add(a04_m1_a26, j_a15_m1_a37));
        I::store(xq.add(s * 7), I::add(s04_pj_s26, v8_s15_pj_s37));

        q += I::COMPLEX_PER_REG;
    }
}

macro_rules! dit8_impl {
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
                    end_2::<$xn>(FWD, 1 << 1, 1 << 3, y, x, true);
                    $core1______(FWD, 1 << 4, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_05<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 2, 1 << 3, y, x, true);
                    $core1______(FWD, 1 << 5, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_06<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_8::<$xn>(FWD, 1 << 3, 1 << 3, y, x, true);
                    $core1______(FWD, 1 << 6, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_07<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 1, 1 << 6, x, y, false);
                    core_::<$xn>(FWD, 1 << 4, 1 << 3, y, x, w);
                    $core1______(FWD, 1 << 7, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_08<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 2, 1 << 6, x, y, false);
                    core_::<$xn>(FWD, 1 << 5, 1 << 3, y, x, w);
                    $core1______(FWD, 1 << 8, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_09<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_8::<$xn>(FWD, 1 << 3, 1 << 6, x, y, false);
                    core_::<$xn>(FWD, 1 << 6, 1 << 3, y, x, w);
                    $core1______(FWD, 1 << 9, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_10<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 01, 1 << 9, y, x, true);
                    core_::<$xn>(FWD, 1 << 04, 1 << 6, x, y, w);
                    core_::<$xn>(FWD, 1 << 07, 1 << 3, y, x, w);
                    $core1______(FWD, 1 << 10, 1 << 0, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_11<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 02, 1 << 09, y, x, true);
                    core_::<$xn>(FWD, 1 << 05, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 08, 1 << 03, y, x, w);
                    $core1______(FWD, 1 << 11, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_12<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_8::<$xn>(FWD, 1 << 03, 1 << 09, y, x, true);
                    core_::<$xn>(FWD, 1 << 06, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 09, 1 << 03, y, x, w);
                    $core1______(FWD, 1 << 12, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_13<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 01, 1 << 12, x, y, false);
                    core_::<$xn>(FWD, 1 << 04, 1 << 09, y, x, w);
                    core_::<$xn>(FWD, 1 << 07, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 03, y, x, w);
                    $core1______(FWD, 1 << 13, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_14<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_4::<$xn>(FWD, 1 << 02, 1 << 12, x, y, false);
                    core_::<$xn>(FWD, 1 << 05, 1 << 09, y, x, w);
                    core_::<$xn>(FWD, 1 << 08, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 11, 1 << 03, y, x, w);
                    $core1______(FWD, 1 << 14, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_15<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_8::<$xn>(FWD, 1 << 03, 1 << 12, x, y, false);
                    core_::<$xn>(FWD, 1 << 06, 1 << 09, y, x, w);
                    core_::<$xn>(FWD, 1 << 09, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 12, 1 << 03, y, x, w);
                    $core1______(FWD, 1 << 15, 1 << 00, x, y, w);
                }
                $(#[target_feature(enable = $target)])?
                unsafe fn fft_16<const FWD: bool>(x: *mut c64, y: *mut c64, w: *const c64) {
                    end_2::<$xn>(FWD, 1 << 01, 1 << 15, y, x, true);
                    core_::<$xn>(FWD, 1 << 04, 1 << 12, x, y, w);
                    core_::<$xn>(FWD, 1 << 07, 1 << 09, y, x, w);
                    core_::<$xn>(FWD, 1 << 10, 1 << 06, x, y, w);
                    core_::<$xn>(FWD, 1 << 13, 1 << 03, y, x, w);
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

dit8_impl! {
    pub static DIT8_SCALAR = Fft {
        core_1: core_::<Scalar>,
        native: Scalar,
        x1: Scalar,
    };

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub static DIT8_AVX = Fft {
        core_1: core_x2::<AvxX2>,
        native: AvxX2,
        x1: AvxX1,
        target: "avx",
    };

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub static DIT8_FMA = Fft {
        core_1: core_x2::<FmaX2>,
        native: FmaX2,
        x1: FmaX1,
        target: "fma",
    };

    #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
    pub static DIT8_AVX512 = Fft {
        core_1: core_x4::<Avx512X2, Avx512X4>,
        native: Avx512X4,
        x1: Avx512X1,
        target: "avx512f",
    };
}

pub(crate) fn runtime_fft() -> crate::FftImpl {
    #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
    if is_x86_feature_detected!("avx512f") {
        return DIT8_AVX512;
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if is_x86_feature_detected!("fma") {
        return DIT8_FMA;
    } else if is_x86_feature_detected!("avx") {
        return DIT8_AVX;
    }

    DIT8_SCALAR
}
