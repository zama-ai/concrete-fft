use crate::c64;
use crate::fft_simd::{FftSimd64, FftSimd64Ext, FftSimd64X2};

#[inline(always)]
unsafe fn fwd_butterfly<I: FftSimd64>(
    z0: I::Reg,
    z1: I::Reg,
    z2: I::Reg,
    z3: I::Reg,
) -> (I::Reg, I::Reg, I::Reg, I::Reg) {
    let z0p2 = I::add(z0, z2);
    let z0m2 = I::sub(z0, z2);
    let z1p3 = I::add(z1, z3);
    let jz1m3 = I::xpj(true, I::sub(z1, z3));

    (
        I::add(z0p2, z1p3),
        I::add(z0m2, jz1m3),
        I::sub(z0p2, z1p3),
        I::sub(z0m2, jz1m3),
    )
}

mod x2 {
    use super::*;

    #[inline(always)]
    pub unsafe fn fwd_breadth<I: FftSimd64X2>(n: usize, z: *mut c64, mut w: *const c64) {
        let mut chunk_size = n;
        let mut chunk_count = 1;

        while chunk_size >= 16 {
            // split into 4 mini chunks

            let m = chunk_size / 4;
            debug_assert_eq!(m % 4, 0);
            let mut p = 0;
            while p < m {
                let w01 = I::load(w.add(3 * p + 2 * 0));
                let w11 = I::load(w.add(3 * p + 2 * 1));
                let w02 = I::load(w.add(3 * p + 2 * 2));
                let w12 = I::load(w.add(3 * p + 2 * 3));
                let w03 = I::load(w.add(3 * p + 2 * 4));
                let w13 = I::load(w.add(3 * p + 2 * 5));

                for i in 0..chunk_count {
                    let z_chunk = z.add(chunk_size * i);

                    let z0 = z_chunk.add(m * 0);
                    let z1 = z_chunk.add(m * 1);
                    let z2 = z_chunk.add(m * 2);
                    let z3 = z_chunk.add(m * 3);

                    let z00 = I::load(z0.add(p));
                    let z10 = I::load(z0.add(p + 2));
                    let z01 = I::load(z1.add(p));
                    let z11 = I::load(z1.add(p + 2));
                    let z02 = I::load(z2.add(p));
                    let z12 = I::load(z2.add(p + 2));
                    let z03 = I::load(z3.add(p));
                    let z13 = I::load(z3.add(p + 2));

                    let (z00, z01, z02, z03) = fwd_butterfly::<I>(z00, z01, z02, z03);
                    let (z10, z11, z12, z13) = fwd_butterfly::<I>(z10, z11, z12, z13);

                    I::store(z0.add(p), z00);
                    I::store(z0.add(p + 2), z10);
                    I::store(z1.add(p), I::mul(w02, z02));
                    I::store(z1.add(p + 2), I::mul(w12, z12));
                    I::store(z2.add(p), I::mul(w01, z01));
                    I::store(z2.add(p + 2), I::mul(w11, z11));
                    I::store(z3.add(p), I::mul(w03, z03));
                    I::store(z3.add(p + 2), I::mul(w13, z13));
                }
                p += 4;
            }

            w = w.add(3 * m);
            chunk_size /= 4;
            chunk_count *= 4;
        }

        if chunk_size == 8 {
            // process chunks by 1
            for i in 0..chunk_count {
                let z = z.add(8 * i);

                let w01 = I::load(w.add(2 * 0));
                let w02 = I::load(w.add(2 * 1));
                let w03 = I::load(w.add(2 * 2));

                let z00 = I::load(z.add(2 * 0));
                let z01 = I::load(z.add(2 * 1));
                let z02 = I::load(z.add(2 * 2));
                let z03 = I::load(z.add(2 * 3));

                let (z00, z01, z02, z03) = fwd_butterfly::<I>(z00, z01, z02, z03);

                I::store(z.add(2 * 0), z00);
                I::store(z.add(2 * 1), I::mul(w02, z02));
                I::store(z.add(2 * 2), I::mul(w01, z01));
                I::store(z.add(2 * 3), I::mul(w03, z03));
            }

            chunk_size /= 4;
            chunk_count *= 4;
        }

        if chunk_size == 4 {
            // process chunks by 2
            debug_assert_eq!(chunk_count % 2, 0);
            let mut i = 0;
            while i < chunk_count {
                let z = z.add(4 * i);

                let z00z01 = I::load(z.add(0));
                let z02z03 = I::load(z.add(2));
                let z10z11 = I::load(z.add(4));
                let z12z13 = I::load(z.add(6));

                let z_0 = I::catlo(z00z01, z10z11);
                let z_1 = I::cathi(z00z01, z10z11);
                let z_2 = I::catlo(z02z03, z12z13);
                let z_3 = I::cathi(z02z03, z12z13);

                let (z_0, z_1, z_2, z_3) = fwd_butterfly::<I>(z_0, z_1, z_2, z_3);

                let z00z02 = I::catlo(z_0, z_2);
                let z10z12 = I::cathi(z_0, z_2);
                let z01z03 = I::catlo(z_1, z_3);
                let z11z13 = I::cathi(z_1, z_3);

                I::store(z.add(0), z00z02);
                I::store(z.add(2), z01z03);
                I::store(z.add(4), z10z12);
                I::store(z.add(6), z11z13);

                i += 2;
            }
        }

        if chunk_size == 2 {
            // process chunks by 4
            debug_assert_eq!(chunk_count % 4, 0);
            let mut i = 0;
            while i < chunk_count {
                let z = z.add(2 * i);
                let z00z01 = I::load(z.add(0));
                let z10z11 = I::load(z.add(2));
                let z20z21 = I::load(z.add(4));
                let z30z31 = I::load(z.add(6));

                let z_0 = I::catlo(z00z01, z10z11);
                let z_1 = I::cathi(z00z01, z10z11);
                let (z_0, z_1) = (I::add(z_0, z_1), I::sub(z_0, z_1));
                let z00z01 = I::catlo(z_0, z_1);
                let z10z11 = I::cathi(z_0, z_1);

                let z_0 = I::catlo(z20z21, z30z31);
                let z_1 = I::cathi(z20z21, z30z31);
                let (z_0, z_1) = (I::add(z_0, z_1), I::sub(z_0, z_1));
                let z20z21 = I::catlo(z_0, z_1);
                let z30z31 = I::cathi(z_0, z_1);

                I::store(z.add(0), z00z01);
                I::store(z.add(2), z10z11);
                I::store(z.add(4), z20z21);
                I::store(z.add(6), z30z31);

                i += 4;
            }
        }
    }

    #[inline(never)]
    pub unsafe fn fwd_depth<I: FftSimd64X2>(n: usize, z: *mut c64, w: *const c64) {
        if n <= 512 {
            fwd_breadth::<I>(n, z, w);
        } else {
            let m = n / 4;
            let z0 = z.add(m * 0);
            let z1 = z.add(m * 1);
            let z2 = z.add(m * 2);
            let z3 = z.add(m * 3);

            debug_assert_eq!(m % 4, 0);
            let mut p = 0;
            while p < m {
                let w01 = I::load(w.add(3 * p + 2 * 0));
                let w11 = I::load(w.add(3 * p + 2 * 1));
                let w02 = I::load(w.add(3 * p + 2 * 2));
                let w12 = I::load(w.add(3 * p + 2 * 3));
                let w03 = I::load(w.add(3 * p + 2 * 4));
                let w13 = I::load(w.add(3 * p + 2 * 5));

                let z00 = I::load(z0.add(p));
                let z10 = I::load(z0.add(p + 2));
                let z01 = I::load(z1.add(p));
                let z11 = I::load(z1.add(p + 2));
                let z02 = I::load(z2.add(p));
                let z12 = I::load(z2.add(p + 2));
                let z03 = I::load(z3.add(p));
                let z13 = I::load(z3.add(p + 2));

                let (z00, z01, z02, z03) = fwd_butterfly::<I>(z00, z01, z02, z03);
                let (z10, z11, z12, z13) = fwd_butterfly::<I>(z10, z11, z12, z13);

                I::store(z0.add(p), z00);
                I::store(z0.add(p + 2), z10);
                I::store(z1.add(p), I::mul(w02, z02));
                I::store(z1.add(p + 2), I::mul(w12, z12));
                I::store(z2.add(p), I::mul(w01, z01));
                I::store(z2.add(p + 2), I::mul(w11, z11));
                I::store(z3.add(p), I::mul(w03, z03));
                I::store(z3.add(p + 2), I::mul(w13, z13));

                p += 4;
            }

            fwd_depth::<I>(m, z0, w.add(3 * m));
            fwd_depth::<I>(m, z1, w.add(3 * m));
            fwd_depth::<I>(m, z2, w.add(3 * m));
            fwd_depth::<I>(m, z3, w.add(3 * m));
        }
    }
}

#[target_feature(enable = "fma")]
pub unsafe fn fwd_fma(n: usize, z: *mut c64, w: *const c64) {
    x2::fwd_depth::<crate::x86::FmaX2>(n, z, w);
}
