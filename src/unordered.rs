//! Unordered FFT module.
//!
//! This module computes the forward or inverse FFT in a similar fashion to the ordered module,
//! with two crucial differences.
//! Given an FFT plan, the forward transform takes its inputs in standard order, and outputs the
//! forward FFT terms in an unspecified order. And the backward transform takes its inputs in the
//! aforementioned order, and outputs the inverse FFT in the standard order.

use crate::{
    c64,
    dif2::{split_2, split_mut_2},
    dif4::split_mut_4,
    dif8::split_mut_8,
    fft_simd::{init_wt, sincospi64, FftSimd, FftSimdExt, Pod},
    ordered::FftAlgo,
};
use aligned_vec::{avec, ABox, CACHELINE_ALIGN};
#[cfg(feature = "std")]
use core::time::Duration;
#[cfg(feature = "std")]
use dyn_stack::{GlobalPodBuffer, ReborrowMut};
use dyn_stack::{PodStack, SizeOverflow, StackReq};

#[inline(always)]
fn fwd_butterfly_x2<c64xN: Pod>(
    simd: impl FftSimd<c64xN>,
    z0: c64xN,
    z1: c64xN,
    w1: c64xN,
) -> (c64xN, c64xN) {
    (simd.add(z0, z1), simd.mul(w1, simd.sub(z0, z1)))
}

#[inline(always)]
fn inv_butterfly_x2<c64xN: Pod>(
    simd: impl FftSimd<c64xN>,
    z0: c64xN,
    z1: c64xN,
    w1: c64xN,
) -> (c64xN, c64xN) {
    let z1 = simd.mul(w1, z1);
    (simd.add(z0, z1), simd.sub(z0, z1))
}

#[inline(always)]
fn fwd_butterfly_x4<c64xN: Pod>(
    simd: impl FftSimd<c64xN>,
    z0: c64xN,
    z1: c64xN,
    z2: c64xN,
    z3: c64xN,
    w1: c64xN,
    w2: c64xN,
    w3: c64xN,
) -> (c64xN, c64xN, c64xN, c64xN) {
    let z0p2 = simd.add(z0, z2);
    let z0m2 = simd.sub(z0, z2);
    let z1p3 = simd.add(z1, z3);
    let jz1m3 = simd.mul_j(true, simd.sub(z1, z3));

    (
        simd.add(z0p2, z1p3),
        simd.mul(w1, simd.sub(z0m2, jz1m3)),
        simd.mul(w2, simd.sub(z0p2, z1p3)),
        simd.mul(w3, simd.add(z0m2, jz1m3)),
    )
}

#[inline(always)]
fn inv_butterfly_x4<c64xN: Pod>(
    simd: impl FftSimd<c64xN>,
    z0: c64xN,
    z1: c64xN,
    z2: c64xN,
    z3: c64xN,
    w1: c64xN,
    w2: c64xN,
    w3: c64xN,
) -> (c64xN, c64xN, c64xN, c64xN) {
    let z0 = z0;
    let z1 = simd.mul(w1, z1);
    let z2 = simd.mul(w2, z2);
    let z3 = simd.mul(w3, z3);

    let z0p2 = simd.add(z0, z2);
    let z0m2 = simd.sub(z0, z2);
    let z1p3 = simd.add(z1, z3);
    let jz1m3 = simd.mul_j(false, simd.sub(z1, z3));

    (
        simd.add(z0p2, z1p3),
        simd.sub(z0m2, jz1m3),
        simd.sub(z0p2, z1p3),
        simd.add(z0m2, jz1m3),
    )
}

#[inline(always)]
fn fwd_butterfly_x8<c64xN: Pod>(
    simd: impl FftSimd<c64xN>,
    z0: c64xN,
    z1: c64xN,
    z2: c64xN,
    z3: c64xN,
    z4: c64xN,
    z5: c64xN,
    z6: c64xN,
    z7: c64xN,
    w1: c64xN,
    w2: c64xN,
    w3: c64xN,
    w4: c64xN,
    w5: c64xN,
    w6: c64xN,
    w7: c64xN,
) -> (c64xN, c64xN, c64xN, c64xN, c64xN, c64xN, c64xN, c64xN) {
    let z0p4 = simd.add(z0, z4);
    let z0m4 = simd.sub(z0, z4);
    let z2p6 = simd.add(z2, z6);
    let jz2m6 = simd.mul_j(true, simd.sub(z2, z6));

    let z1p5 = simd.add(z1, z5);
    let z1m5 = simd.sub(z1, z5);
    let z3p7 = simd.add(z3, z7);
    let jz3m7 = simd.mul_j(true, simd.sub(z3, z7));

    // z0 + z2 + z4 + z6
    let t0 = simd.add(z0p4, z2p6);
    // z1 + z3 + z5 + z7
    let t1 = simd.add(z1p5, z3p7);
    // z0 + w4z2 + z4 + w4z6
    let t2 = simd.sub(z0p4, z2p6);
    // w2z1 + w6z3 + w2z5 + w6z7
    let t3 = simd.mul_j(true, simd.sub(z1p5, z3p7));
    // z0 + w2z2 + z4 + w6z6
    let t4 = simd.sub(z0m4, jz2m6);
    // w1z1 + w3z3 + w5z5 + w7z7
    let t5 = simd.mul_exp_neg_pi_over_8(true, simd.sub(z1m5, jz3m7));
    // z0 + w2z2 + w4z4 + w6z6
    let t6 = simd.add(z0m4, jz2m6);
    // w7z1 + w1z3 + w3z5 + w5z7
    let t7 = simd.mul_exp_pi_over_8(true, simd.add(z1m5, jz3m7));

    (
        simd.add(t0, t1),
        simd.mul(w1, simd.add(t4, t5)),
        simd.mul(w2, simd.sub(t2, t3)),
        simd.mul(w3, simd.sub(t6, t7)),
        simd.mul(w4, simd.sub(t0, t1)),
        simd.mul(w5, simd.sub(t4, t5)),
        simd.mul(w6, simd.add(t2, t3)),
        simd.mul(w7, simd.add(t6, t7)),
    )
}

#[inline(always)]
fn inv_butterfly_x8<c64xN: Pod>(
    simd: impl FftSimd<c64xN>,
    z0: c64xN,
    z1: c64xN,
    z2: c64xN,
    z3: c64xN,
    z4: c64xN,
    z5: c64xN,
    z6: c64xN,
    z7: c64xN,
    w1: c64xN,
    w2: c64xN,
    w3: c64xN,
    w4: c64xN,
    w5: c64xN,
    w6: c64xN,
    w7: c64xN,
) -> (c64xN, c64xN, c64xN, c64xN, c64xN, c64xN, c64xN, c64xN) {
    let z0 = z0;
    let z1 = simd.mul(w1, z1);
    let z2 = simd.mul(w2, z2);
    let z3 = simd.mul(w3, z3);
    let z4 = simd.mul(w4, z4);
    let z5 = simd.mul(w5, z5);
    let z6 = simd.mul(w6, z6);
    let z7 = simd.mul(w7, z7);

    let z0p4 = simd.add(z0, z4);
    let z0m4 = simd.sub(z0, z4);
    let z2p6 = simd.add(z2, z6);
    let jz2m6 = simd.mul_j(false, simd.sub(z2, z6));

    let z1p5 = simd.add(z1, z5);
    let z1m5 = simd.sub(z1, z5);
    let z3p7 = simd.add(z3, z7);
    let jz3m7 = simd.mul_j(false, simd.sub(z3, z7));

    // z0 + z2 + z4 + z6
    let t0 = simd.add(z0p4, z2p6);
    // z1 + z3 + z5 + z7
    let t1 = simd.add(z1p5, z3p7);
    // z0 + w4z2 + z4 + w4z6
    let t2 = simd.sub(z0p4, z2p6);
    // w2z1 + w6z3 + w2z5 + w6z7
    let t3 = simd.mul_j(false, simd.sub(z1p5, z3p7));
    // z0 + w2z2 + z4 + w6z6
    let t4 = simd.sub(z0m4, jz2m6);
    // w1z1 + w3z3 + w5z5 + w7z7
    let t5 = simd.mul_exp_neg_pi_over_8(false, simd.sub(z1m5, jz3m7));
    // z0 + w2z2 + w4z4 + w6z6
    let t6 = simd.add(z0m4, jz2m6);
    // w7z1 + w1z3 + w3z5 + w5z7
    let t7 = simd.mul_exp_pi_over_8(false, simd.add(z1m5, jz3m7));

    (
        simd.add(t0, t1),
        simd.add(t4, t5),
        simd.sub(t2, t3),
        simd.sub(t6, t7),
        simd.sub(t0, t1),
        simd.sub(t4, t5),
        simd.add(t2, t3),
        simd.add(t6, t7),
    )
}

#[inline(always)]
fn fwd_process_x2<c64xN: Pod>(simd: impl FftSimd<c64xN>, z: &mut [c64], w: &[c64]) {
    let z: &mut [c64xN] = bytemuck::cast_slice_mut(z);
    let w: &[[c64xN; 1]] = bytemuck::cast_slice(w);
    let (z0, z1) = split_mut_2(z);

    for (z0, z1, &[w1]) in izip!(z0, z1, w) {
        (*z0, *z1) = fwd_butterfly_x2(simd, *z0, *z1, w1);
    }
}

#[inline(always)]
fn inv_process_x2<c64xN: Pod>(simd: impl FftSimd<c64xN>, z: &mut [c64], w: &[c64]) {
    let z: &mut [c64xN] = bytemuck::cast_slice_mut(z);
    let w: &[[c64xN; 1]] = bytemuck::cast_slice(w);
    let (z0, z1) = split_mut_2(z);

    for (z0, z1, &[w1]) in izip!(z0, z1, w) {
        (*z0, *z1) = inv_butterfly_x2(simd, *z0, *z1, w1);
    }
}

#[inline(always)]
fn fwd_process_x4<c64xN: Pod>(simd: impl FftSimd<c64xN>, z: &mut [c64], w: &[c64]) {
    let z: &mut [c64xN] = bytemuck::cast_slice_mut(z);
    let w: &[[c64xN; 3]] = bytemuck::cast_slice(w);
    let (z0, z1, z2, z3) = split_mut_4(z);

    for (z0, z1, z2, z3, &[w1, w2, w3]) in izip!(z0, z1, z2, z3, w) {
        (*z0, *z2, *z1, *z3) = fwd_butterfly_x4(simd, *z0, *z1, *z2, *z3, w1, w2, w3);
    }
}

#[inline(always)]
fn inv_process_x4<c64xN: Pod>(simd: impl FftSimd<c64xN>, z: &mut [c64], w: &[c64]) {
    let z: &mut [c64xN] = bytemuck::cast_slice_mut(z);
    let w: &[[c64xN; 3]] = bytemuck::cast_slice(w);
    let (z0, z1, z2, z3) = split_mut_4(z);

    for (z0, z1, z2, z3, &[w1, w2, w3]) in izip!(z0, z1, z2, z3, w) {
        (*z0, *z1, *z2, *z3) = inv_butterfly_x4(simd, *z0, *z2, *z1, *z3, w1, w2, w3);
    }
}

#[inline(always)]
fn fwd_process_x8<c64xN: Pod>(simd: impl FftSimd<c64xN>, z: &mut [c64], w: &[c64]) {
    let z: &mut [c64xN] = bytemuck::cast_slice_mut(z);
    let w: &[[c64xN; 7]] = bytemuck::cast_slice(w);
    let (z0, z1, z2, z3, z4, z5, z6, z7) = split_mut_8(z);

    for (z0, z1, z2, z3, z4, z5, z6, z7, &[w1, w2, w3, w4, w5, w6, w7]) in
        izip!(z0, z1, z2, z3, z4, z5, z6, z7, w)
    {
        (*z0, *z4, *z2, *z6, *z1, *z5, *z3, *z7) = fwd_butterfly_x8(
            simd, *z0, *z1, *z2, *z3, *z4, *z5, *z6, *z7, w1, w2, w3, w4, w5, w6, w7,
        );
    }
}

#[inline(always)]
fn inv_process_x8<c64xN: Pod>(simd: impl FftSimd<c64xN>, z: &mut [c64], w: &[c64]) {
    let z: &mut [c64xN] = bytemuck::cast_slice_mut(z);
    let w: &[[c64xN; 7]] = bytemuck::cast_slice(w);
    let (z0, z1, z2, z3, z4, z5, z6, z7) = split_mut_8(z);

    for (z0, z1, z2, z3, z4, z5, z6, z7, &[w1, w2, w3, w4, w5, w6, w7]) in
        izip!(z0, z1, z2, z3, z4, z5, z6, z7, w)
    {
        (*z0, *z1, *z2, *z3, *z4, *z5, *z6, *z7) = inv_butterfly_x8(
            simd, *z0, *z4, *z2, *z6, *z1, *z5, *z3, *z7, w1, w2, w3, w4, w5, w6, w7,
        );
    }
}

macro_rules! dispatcher {
    ($name: ident, $impl: ident) => {
        fn $name() -> fn(&mut [c64], &[c64]) {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                #[cfg(feature = "nightly")]
                if pulp::x86::V4::try_new().is_some() {
                    return |z, w| {
                        let simd = pulp::x86::V4::try_new().unwrap();
                        simd.vectorize(
                            #[inline(always)]
                            || $impl(simd, z, w),
                        );
                    };
                }

                if pulp::x86::V3::try_new().is_some() {
                    return |z, w| {
                        let simd = pulp::x86::V3::try_new().unwrap();
                        simd.vectorize(
                            #[inline(always)]
                            || $impl(simd, z, w),
                        );
                    };
                }
            }

            |z, w| $impl(crate::fft_simd::Scalar, z, w)
        }
    };
}

dispatcher!(get_fwd_process_x2, fwd_process_x2);
dispatcher!(get_fwd_process_x4, fwd_process_x4);
dispatcher!(get_fwd_process_x8, fwd_process_x8);

dispatcher!(get_inv_process_x2, inv_process_x2);
dispatcher!(get_inv_process_x4, inv_process_x4);
dispatcher!(get_inv_process_x8, inv_process_x8);

fn get_complex_per_reg() -> usize {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        #[cfg(feature = "nightly")]
        if let Some(simd) = pulp::x86::V4::try_new() {
            return simd.lane_count();
        }
        if let Some(simd) = pulp::x86::V3::try_new() {
            return simd.lane_count();
        }
    }
    crate::fft_simd::Scalar.lane_count()
}

fn init_twiddles(
    n: usize,
    complex_per_reg: usize,
    base_n: usize,
    base_r: usize,
    w: &mut [c64],
    w_inv: &mut [c64],
) {
    let theta = 2.0 / n as f64;
    if n <= base_n {
        init_wt(base_r, n, w, w_inv);
    } else {
        let r = if n == 2 * base_n {
            2
        } else if n == 4 * base_n {
            4
        } else {
            8
        };

        let m = n / r;
        let (w, w_next) = w.split_at_mut((r - 1) * m);
        let (w_inv_next, w_inv) = w_inv.split_at_mut(w_inv.len() - (r - 1) * m);

        let mut p = 0;
        while p < m {
            for i in 0..complex_per_reg {
                for k in 1..r {
                    let (sk, ck) = sincospi64(theta * (k * (p + i)) as f64);
                    let idx = (r - 1) * p + (k - 1) * complex_per_reg + i;
                    w[idx] = c64 { re: ck, im: -sk };
                    w_inv[idx] = c64 { re: ck, im: sk };
                }
            }

            p += complex_per_reg;
        }

        init_twiddles(n / r, complex_per_reg, base_n, base_r, w_next, w_inv_next);
    }
}

#[inline(never)]
fn fwd_depth(
    z: &mut [c64],
    w: &[c64],
    base_fn: fn(&mut [c64], &mut [c64], &[c64], &[c64]),
    base_n: usize,
    base_scratch: &mut [c64],
    fwd_process_x2: fn(&mut [c64], &[c64]),
    fwd_process_x4: fn(&mut [c64], &[c64]),
    fwd_process_x8: fn(&mut [c64], &[c64]),
) {
    let n = z.len();
    if n == base_n {
        let (w_init, w) = split_2(w);
        base_fn(z, base_scratch, w_init, w);
    } else {
        let r = if n == 2 * base_n {
            2
        } else if n == 4 * base_n {
            4
        } else {
            8
        };

        let m = n / r;
        let (w_head, w_tail) = w.split_at((r - 1) * m);

        if n == 2 * base_n {
            fwd_process_x2(z, w_head);
        } else if n == 4 * base_n {
            fwd_process_x4(z, w_head);
        } else {
            fwd_process_x8(z, w_head);
        }

        for z in z.chunks_exact_mut(m) {
            fwd_depth(
                z,
                w_tail,
                base_fn,
                base_n,
                base_scratch,
                fwd_process_x2,
                fwd_process_x4,
                fwd_process_x8,
            );
        }
    }
}

#[inline(never)]
fn inv_depth(
    z: &mut [c64],
    w: &[c64],
    base_fn: fn(&mut [c64], &mut [c64], &[c64], &[c64]),
    base_n: usize,
    base_scratch: &mut [c64],
    inv_process_x2: fn(&mut [c64], &[c64]),
    inv_process_x4: fn(&mut [c64], &[c64]),
    inv_process_x8: fn(&mut [c64], &[c64]),
) {
    let n = z.len();

    if n == base_n {
        let (w_init, w) = split_2(w);
        base_fn(z, base_scratch, w_init, w);
    } else {
        let r = if n == 2 * base_n {
            2
        } else if n == 4 * base_n {
            4
        } else {
            8
        };

        let m = n / r;
        let (w_head, w_tail) = w.split_at(w.len() - (r - 1) * m);
        for z in z.chunks_exact_mut(m) {
            inv_depth(
                z,
                w_head,
                base_fn,
                base_n,
                base_scratch,
                inv_process_x2,
                inv_process_x4,
                inv_process_x8,
            );
        }

        if r == 2 {
            inv_process_x2(z, w_tail);
        } else if r == 4 {
            inv_process_x4(z, w_tail);
        } else {
            inv_process_x8(z, w_tail);
        }
    }
}

/// Unordered FFT plan.
///
/// This type holds a forward and inverse FFT plan and twiddling factors for a specific size.
/// The size must be a power of two.
#[derive(Clone)]
pub struct Plan {
    twiddles: ABox<[c64]>,
    twiddles_inv: ABox<[c64]>,
    fwd_process_x2: fn(&mut [c64], &[c64]),
    fwd_process_x4: fn(&mut [c64], &[c64]),
    fwd_process_x8: fn(&mut [c64], &[c64]),
    inv_process_x2: fn(&mut [c64], &[c64]),
    inv_process_x4: fn(&mut [c64], &[c64]),
    inv_process_x8: fn(&mut [c64], &[c64]),
    base_n: usize,
    base_fn_fwd: fn(&mut [c64], &mut [c64], &[c64], &[c64]),
    base_fn_inv: fn(&mut [c64], &mut [c64], &[c64], &[c64]),
    base_algo: FftAlgo,
    n: usize,
}

impl core::fmt::Debug for Plan {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Plan")
            .field("base_algo", &self.base_algo)
            .field("base_size", &self.base_n)
            .field("fft_size", &self.fft_size())
            .finish()
    }
}

/// Method for selecting the unordered FFT plan.
#[derive(Clone, Copy, Debug)]
pub enum Method {
    /// Select the FFT plan by manually providing the underlying algorithm.
    /// The unordered FFT works by using an internal ordered FFT plan, whose size and algorithm can
    /// be specified by the user.
    UserProvided { base_algo: FftAlgo, base_n: usize },
    /// Select the FFT plan by measuring the running time of all the possible plans and selecting
    /// the fastest one. The provided duration specifies how long the benchmark of each plan should
    /// last.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    Measure(Duration),
}

#[cfg(feature = "std")]
fn measure_fastest_scratch(n: usize) -> StackReq {
    if n <= 512 {
        crate::ordered::measure_fastest_scratch(n)
    } else {
        let base_n = 4096;
        crate::ordered::measure_fastest_scratch(base_n)
            .and(StackReq::new_aligned::<c64>(n + base_n, CACHELINE_ALIGN)) // twiddles
            .and(StackReq::new_aligned::<c64>(n, CACHELINE_ALIGN)) // buf
            .and(StackReq::new_aligned::<c64>(base_n, CACHELINE_ALIGN)) // scratch
    }
}

#[cfg(feature = "std")]
fn measure_fastest(
    mut min_bench_duration_per_algo: Duration,
    n: usize,
    mut stack: PodStack,
) -> (FftAlgo, usize, Duration) {
    const MIN_DURATION: Duration = Duration::from_millis(1);
    min_bench_duration_per_algo = min_bench_duration_per_algo.max(MIN_DURATION);

    if n <= 256 {
        let (algo, duration) =
            crate::ordered::measure_fastest(min_bench_duration_per_algo, n, stack);
        (algo, n, duration)
    } else {
        // bench

        let bases = [512, 1024];
        let mut algos: [Option<FftAlgo>; 4] = [None; 4];
        let mut avg_durations: [Option<Duration>; 4] = [None; 4];
        let fwd_process_x2 = get_fwd_process_x2();
        let fwd_process_x4 = get_fwd_process_x4();
        let fwd_process_x8 = get_fwd_process_x8();

        let mut n_algos = 0;
        for (i, base_n) in bases.into_iter().enumerate() {
            if n < base_n {
                break;
            }

            n_algos += 1;

            // we'll measure the corresponding plan
            let (base_algo, duration) = crate::ordered::measure_fastest(
                min_bench_duration_per_algo,
                base_n,
                stack.rb_mut(),
            );

            algos[i] = Some(base_algo);

            if n == base_n {
                avg_durations[i] = Some(duration);
                continue;
            }

            // get the forward base algo
            let base_fn = crate::ordered::get_fn_ptr(base_algo, base_n)[0];

            let f = |_| c64 { re: 0.0, im: 0.0 };
            let align = CACHELINE_ALIGN;
            let (w, stack) = stack
                .rb_mut()
                .make_aligned_with::<c64, _>(n + base_n, align, f);
            let (mut scratch, stack) = stack.make_aligned_with::<c64, _>(base_n, align, f);
            let (mut z, _) = stack.make_aligned_with::<c64, _>(n, align, f);

            let n_runs = min_bench_duration_per_algo.as_secs_f64()
                / (duration.as_secs_f64() * (n / base_n) as f64);

            let n_runs = n_runs.ceil() as u32;

            use std::time::Instant;
            let now = Instant::now();
            for _ in 0..n_runs {
                fwd_depth(
                    &mut z,
                    &w,
                    base_fn,
                    base_n,
                    &mut scratch,
                    fwd_process_x2,
                    fwd_process_x4,
                    fwd_process_x8,
                );
            }
            let duration = now.elapsed();
            avg_durations[i] = Some(duration / n_runs);
        }

        let best_time = avg_durations[..n_algos].iter().min().unwrap().unwrap();
        let best_index = avg_durations[..n_algos]
            .iter()
            .position(|elem| elem.unwrap() == best_time)
            .unwrap();

        (algos[best_index].unwrap(), bases[best_index], best_time)
    }
}

impl Plan {
    /// Returns a new FFT plan for the given vector size, selected by the provided method.
    ///
    /// # Panics
    ///
    /// - Panics if `n` is not a power of two.
    /// - If the method is user-provided, panics if `n` is not equal to the base ordered FFT size,
    /// and the base FFT size is less than `32`.
    ///
    /// # Example
    #[cfg_attr(feature = "std", doc = " ```")]
    #[cfg_attr(not(feature = "std"), doc = " ```ignore")]
    /// use concrete_fft::unordered::{Method, Plan};
    /// use core::time::Duration;
    ///
    /// let plan = Plan::new(4, Method::Measure(Duration::from_millis(10)));
    /// ```
    pub fn new(n: usize, method: Method) -> Self {
        assert!(n.is_power_of_two());

        let (base_algo, base_n) = match method {
            Method::UserProvided { base_algo, base_n } => {
                assert!(base_n.is_power_of_two());
                assert!(base_n <= n);
                if base_n != n {
                    assert!(base_n >= 32);
                }
                assert!(base_n.trailing_zeros() <= 10);
                (base_algo, base_n)
            }

            #[cfg(feature = "std")]
            Method::Measure(duration) => {
                let (algo, base_n, _) = measure_fastest(
                    duration,
                    n,
                    PodStack::new(&mut GlobalPodBuffer::new(measure_fastest_scratch(n))),
                );
                (algo, base_n)
            }
        };

        let [base_fn_fwd, base_fn_inv] = crate::ordered::get_fn_ptr(base_algo, base_n);

        let nan = c64 {
            re: f64::NAN,
            im: f64::NAN,
        };
        let mut twiddles = avec![nan; n + base_n].into_boxed_slice();
        let mut twiddles_inv = avec![nan; n + base_n].into_boxed_slice();

        use crate::ordered::FftAlgo::*;
        let base_r = match base_algo {
            Dif2 | Dit2 => 2,
            Dif4 | Dit4 => 4,
            Dif8 | Dit8 => 8,
            Dif16 | Dit16 => 16,
        };

        init_twiddles(
            n,
            get_complex_per_reg(),
            base_n,
            base_r,
            &mut twiddles,
            &mut twiddles_inv,
        );

        Self {
            twiddles,
            twiddles_inv,
            fwd_process_x2: get_fwd_process_x2(),
            fwd_process_x4: get_fwd_process_x4(),
            fwd_process_x8: get_fwd_process_x8(),
            inv_process_x2: get_inv_process_x2(),
            inv_process_x4: get_inv_process_x4(),
            inv_process_x8: get_inv_process_x8(),
            base_n,
            base_fn_fwd,
            base_fn_inv,
            n,
            base_algo,
        }
    }

    /// Returns the vector size of the FFT.
    ///
    /// # Example
    #[cfg_attr(feature = "std", doc = " ```")]
    #[cfg_attr(not(feature = "std"), doc = " ```ignore")]
    /// use concrete_fft::unordered::{Method, Plan};
    /// use core::time::Duration;
    ///
    /// let plan = Plan::new(4, Method::Measure(Duration::from_millis(10)));
    /// assert_eq!(plan.fft_size(), 4);
    /// ```
    pub fn fft_size(&self) -> usize {
        self.n
    }

    /// Returns the algorithm and size of the internal ordered FFT plan.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_fft::{
    ///     ordered::FftAlgo,
    ///     unordered::{Method, Plan},
    /// };
    ///
    /// let plan = Plan::new(
    ///     4,
    ///     Method::UserProvided {
    ///         base_algo: FftAlgo::Dif2,
    ///         base_n: 4,
    ///     },
    /// );
    /// assert_eq!(plan.algo(), (FftAlgo::Dif2, 4));
    /// ```
    pub fn algo(&self) -> (FftAlgo, usize) {
        (self.base_algo, self.base_n)
    }

    /// Returns the size and alignment of the scratch memory needed to perform an FFT.
    ///
    /// # Example
    #[cfg_attr(feature = "std", doc = " ```")]
    #[cfg_attr(not(feature = "std"), doc = " ```ignore")]
    /// use concrete_fft::unordered::{Method, Plan};
    /// use core::time::Duration;
    ///
    /// let plan = Plan::new(4, Method::Measure(Duration::from_millis(10)));
    /// let scratch = plan.fft_scratch().unwrap();
    /// ```
    pub fn fft_scratch(&self) -> Result<StackReq, SizeOverflow> {
        StackReq::try_new_aligned::<c64>(self.algo().1, CACHELINE_ALIGN)
    }

    /// Performs a forward FFT in place, using the provided stack as scratch space.
    ///
    /// # Note
    ///
    /// The values in `buf` must be in standard order prior to calling this function.
    /// When this function returns, the values in `buf` will contain the terms of the forward
    /// transform in permuted order.
    ///
    /// # Example
    #[cfg_attr(feature = "std", doc = " ```")]
    #[cfg_attr(not(feature = "std"), doc = " ```ignore")]
    /// use concrete_fft::c64;
    /// use concrete_fft::unordered::{Method, Plan};
    /// use dyn_stack::{PodStack, GlobalPodBuffer};
    /// use core::time::Duration;
    ///
    /// let plan = Plan::new(4, Method::Measure(Duration::from_millis(10)));
    ///
    /// let mut memory = GlobalPodBuffer::new(plan.fft_scratch().unwrap());
    /// let stack = PodStack::new(&mut memory);
    ///
    /// let mut buf = [c64::default(); 4];
    /// plan.fwd(&mut buf, stack);
    /// ```
    pub fn fwd(&self, buf: &mut [c64], stack: PodStack) {
        assert_eq!(self.fft_size(), buf.len());
        let (mut scratch, _) = stack.make_aligned_raw::<c64>(self.algo().1, CACHELINE_ALIGN);
        fwd_depth(
            buf,
            &self.twiddles,
            self.base_fn_fwd,
            self.base_n,
            &mut scratch,
            self.fwd_process_x2,
            self.fwd_process_x4,
            self.fwd_process_x8,
        );
    }

    /// Performs an inverse FFT in place, using the provided stack as scratch space.
    ///
    /// # Note
    ///
    /// The values in `buf` must be in permuted order prior to calling this function.
    /// When this function returns, the values in `buf` will contain the terms of the forward
    /// transform in standard order.
    ///
    /// # Example
    #[cfg_attr(feature = "std", doc = " ```")]
    #[cfg_attr(not(feature = "std"), doc = " ```ignore")]
    /// use concrete_fft::c64;
    /// use concrete_fft::unordered::{Method, Plan};
    /// use dyn_stack::{PodStack, GlobalPodBuffer, ReborrowMut};
    /// use core::time::Duration;
    ///
    /// let plan = Plan::new(4, Method::Measure(Duration::from_millis(10)));
    ///
    /// let mut memory = GlobalPodBuffer::new(plan.fft_scratch().unwrap());
    /// let mut stack = PodStack::new(&mut memory);
    ///
    /// let mut buf = [c64::default(); 4];
    /// plan.fwd(&mut buf, stack.rb_mut());
    /// plan.inv(&mut buf, stack);
    /// ```
    pub fn inv(&self, buf: &mut [c64], stack: PodStack) {
        assert_eq!(self.fft_size(), buf.len());
        let (mut scratch, _) = stack.make_aligned_raw::<c64>(self.algo().1, CACHELINE_ALIGN);
        inv_depth(
            buf,
            &self.twiddles_inv,
            self.base_fn_inv,
            self.base_n,
            &mut scratch,
            self.inv_process_x2,
            self.inv_process_x4,
            self.inv_process_x8,
        );
    }

    /// Serialize a buffer containing data in the Fourier domain that is stored in the
    /// plan-specific permuted order, and store the result with the serializer in the standard
    /// order.
    ///
    /// # Panics
    ///
    /// - Panics if the length of `buf` is not equal to the FFT size.
    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    pub fn serialize_fourier_buffer<S: serde::Serializer>(
        &self,
        serializer: S,
        buf: &[c64],
    ) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;

        let n = self.n;
        let base_n = self.base_n;
        assert_eq!(n, buf.len());

        let mut seq = serializer.serialize_seq(Some(n))?;

        let nbits = n.trailing_zeros();
        let base_nbits = base_n.trailing_zeros();

        for i in 0..n {
            seq.serialize_element(&buf[bit_rev_twice(nbits, base_nbits, i)])?;
        }

        seq.end()
    }

    /// Deserialize data in the Fourier domain that is produced by the deserializer in the standard
    /// order into a buffer so that it will contain the data in the plan-specific permuted order
    ///
    /// # Panics
    ///
    /// - Panics if the length of `buf` is not equal to the FFT size.
    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    pub fn deserialize_fourier_buffer<'de, D: serde::Deserializer<'de>>(
        &self,
        deserializer: D,
        buf: &mut [c64],
    ) -> Result<(), D::Error> {
        use serde::de::{SeqAccess, Visitor};

        let n = self.n;
        let base_n = self.base_n;
        assert_eq!(n, buf.len());

        struct SeqVisitor<'a> {
            buf: &'a mut [c64],
            base_n: usize,
        }

        impl<'de, 'a> Visitor<'de> for SeqVisitor<'a> {
            type Value = ();

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(
                    formatter,
                    "a sequence of {} 64-bit complex numbers",
                    self.buf.len()
                )
            }

            fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
            where
                S: SeqAccess<'de>,
            {
                let n = self.buf.len();
                let nbits = n.trailing_zeros();
                let base_nbits = self.base_n.trailing_zeros();

                let mut i = 0;

                while let Some(value) = seq.next_element::<c64>()? {
                    if i < n {
                        self.buf[bit_rev_twice(nbits, base_nbits, i)] = value;
                    }

                    i += 1;
                }

                if i != n {
                    Err(serde::de::Error::invalid_length(i, &self))
                } else {
                    Ok(())
                }
            }
        }

        deserializer.deserialize_seq(SeqVisitor { buf, base_n })
    }
}

#[cfg(any(test, feature = "serde"))]
#[inline]
fn bit_rev(nbits: u32, i: usize) -> usize {
    i.reverse_bits() >> (usize::BITS - nbits)
}

#[cfg(any(test, feature = "serde"))]
#[inline]
fn bit_rev_twice(nbits: u32, base_nbits: u32, i: usize) -> usize {
    let i_rev = bit_rev(nbits, i);
    let bottom_mask = (1 << base_nbits) - 1;
    let bottom_bits = bit_rev(base_nbits, i_rev);
    (i_rev & !bottom_mask) | bottom_bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use dyn_stack::{GlobalPodBuffer, ReborrowMut};
    use num_complex::ComplexFloat;
    use rand::random;

    extern crate alloc;

    #[test]
    fn test_fwd() {
        for n in [128, 256, 512, 1024] {
            let mut z = vec![c64::default(); n];

            for z in &mut z {
                z.re = random();
                z.im = random();
            }

            let mut z_target = z.clone();
            let mut planner = rustfft::FftPlanner::new();
            let fwd = planner.plan_fft_forward(n);
            fwd.process(&mut z_target);

            let plan = Plan::new(
                n,
                Method::UserProvided {
                    base_algo: FftAlgo::Dif4,
                    base_n: 32,
                },
            );
            let base_n = plan.algo().1;
            let mut mem = GlobalPodBuffer::new(plan.fft_scratch().unwrap());
            let stack = PodStack::new(&mut mem);
            plan.fwd(&mut z, stack);

            for (i, z_target) in z_target.iter().enumerate() {
                let idx = bit_rev_twice(n.trailing_zeros(), base_n.trailing_zeros(), i);
                assert!((z[idx] - z_target).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_roundtrip() {
        for n in [32, 64, 256, 512, 1024] {
            let mut z = vec![c64::default(); n];

            for z in &mut z {
                z.re = random();
                z.im = random();
            }

            let orig = z.clone();

            let plan = Plan::new(
                n,
                Method::UserProvided {
                    base_algo: FftAlgo::Dif4,
                    base_n: 32,
                },
            );
            let mut mem = GlobalPodBuffer::new(plan.fft_scratch().unwrap());
            let mut stack = PodStack::new(&mut mem);
            plan.fwd(&mut z, stack.rb_mut());
            plan.inv(&mut z, stack);

            for z in &mut z {
                *z /= n as f64;
            }

            for (z_actual, z_expected) in z.iter().zip(&orig) {
                assert!((z_actual - z_expected).abs() < 1e-12);
            }
        }
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests_serde {
    use super::*;
    use alloc::{vec, vec::Vec};
    use dyn_stack::{GlobalPodBuffer, ReborrowMut};
    use num_complex::ComplexFloat;
    use rand::random;

    extern crate alloc;

    #[test]
    fn test_serde() {
        for n in [64, 128, 256, 512, 1024] {
            let mut z = vec![c64::default(); n];

            for z in &mut z {
                z.re = random();
                z.im = random();
            }

            let orig = z.clone();

            let plan1 = Plan::new(
                n,
                Method::UserProvided {
                    base_algo: FftAlgo::Dif4,
                    base_n: 32,
                },
            );
            let plan2 = Plan::new(
                n,
                Method::UserProvided {
                    base_algo: FftAlgo::Dif4,
                    base_n: 64,
                },
            );

            let mut mem = GlobalPodBuffer::new(
                plan1
                    .fft_scratch()
                    .unwrap()
                    .or(plan2.fft_scratch().unwrap()),
            );
            let mut stack = PodStack::new(&mut mem);

            plan1.fwd(&mut z, stack.rb_mut());

            let mut buf = Vec::<u8>::new();
            let mut serializer = bincode::Serializer::new(&mut buf, bincode::options());
            plan1.serialize_fourier_buffer(&mut serializer, &z).unwrap();

            let mut deserializer = bincode::de::Deserializer::from_slice(&buf, bincode::options());
            plan2
                .deserialize_fourier_buffer(&mut deserializer, &mut z)
                .unwrap();

            plan2.inv(&mut z, stack);

            for z in &mut z {
                *z /= n as f64;
            }

            for (z_actual, z_expected) in z.iter().zip(&orig) {
                assert!((z_actual - z_expected).abs() < 1e-12);
            }
        }
    }
}
