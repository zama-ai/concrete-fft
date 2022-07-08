//! Ordered FFT module.
//!
//! This FFT is currently based on hte Stockham algorithm, and was ported from the
//! [OTFFT](http://wwwa.pikara.ne.jp/okojisan/otfft-en/) C++ library by Takuya OKAHISA.
//!
//! This module computes the forward or inverse FFT in standard ordering.
//! This means that given a buffer of complex numbers $[x_0, \dots, x_{n-1}]$,
//! the forward FFT $[X_0, \dots, X_{n-1}]$ is given by
//! $$X_p = \sum_{q = 0}^{n-1} \exp\left(-\frac{i 2\pi pq}{n}\right),$$
//! and the inverse FFT $[Y_0, \dots, Y_{n-1}]$ is given by
//! $$Y_p = \sum_{q = 0}^{n-1} \exp\left(\frac{i 2\pi pq}{n}\right).$$

use crate::*;
use aligned_vec::avec;
use aligned_vec::ABox;
use aligned_vec::CACHELINE_ALIGN;

#[cfg(feature = "std")]
use core::time::Duration;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
#[cfg(feature = "std")]
use dyn_stack::{GlobalMemBuffer, ReborrowMut};

/// Internal FFT algorithm.
///
/// The FFT can use a decimation-in-frequency (DIF) or decimation-in-time (DIT) approach.
/// And the FFT radix can be any of 2, 4, 8, 16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FftAlgo {
    /// Decimation in frequency with radix 2
    Dif2,
    /// Decimation in time with radix 2
    Dit2,
    /// Decimation in frequency with radix 4
    Dif4,
    /// Decimation in time with radix 4
    Dit4,
    /// Decimation in frequency with radix 8
    Dif8,
    /// Decimation in time with radix 8
    Dit8,
    /// Decimation in frequency with radix 16
    Dif16,
    /// Decimation in time with radix 16
    Dit16,
}

/// Method for selecting the ordered FFT plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Method {
    /// Select the FFT plan by manually providing the underlying algorithm.
    UserProvided(FftAlgo),
    /// Select the FFT plan by measuring the running time of all the possible plans and selecting
    /// the fastest one. The provided duration specifies how long the benchmark of each plan should
    /// last.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    Measure(Duration),
}

#[cfg(feature = "std")]
fn measure_n_runs(
    n_runs: u128,
    algo: FftAlgo,
    buf: &mut [c64],
    twiddles: &[c64],
    stack: DynStack,
) -> Duration {
    let n = buf.len();
    let (mut scratch, _) = stack.make_aligned_uninit::<c64>(n, CACHELINE_ALIGN);
    let scratch = scratch.as_mut_ptr() as *mut c64;
    let [fwd, _] = get_fn_ptr(algo, n);

    use std::time::Instant;
    let now = Instant::now();

    for _ in 0..n_runs {
        unsafe {
            fwd(buf.as_mut_ptr(), scratch, twiddles.as_ptr());
        }
    }

    now.elapsed()
}

#[cfg(feature = "std")]
fn duration_div_f64(duration: Duration, n: f64) -> Duration {
    Duration::from_secs_f64(duration.as_secs_f64() / n as f64)
}

#[cfg(feature = "std")]
pub(crate) fn measure_fastest_scratch(n: usize) -> StackReq {
    let align = CACHELINE_ALIGN;
    StackReq::new_aligned::<c64>(2 * n, align) // twiddles
        .and(StackReq::new_aligned::<c64>(n, align)) // buffer
        .and(StackReq::new_aligned::<c64>(n, align))
}

#[cfg(feature = "std")]
pub(crate) fn measure_fastest(
    min_bench_duration_per_algo: Duration,
    n: usize,
    stack: DynStack,
) -> (FftAlgo, Duration) {
    const N_ALGOS: usize = 8;
    const MIN_DURATION: Duration = Duration::from_millis(1);

    assert!(n.is_power_of_two());

    let align = CACHELINE_ALIGN;

    let f = |_| c64::default();

    let (twiddles, stack) = stack.make_aligned_with::<c64, _>(2 * n, align, f);
    let (mut buf, mut stack) = stack.make_aligned_with::<c64, _>(n, align, f);

    {
        // initialize scratch to load it in the cpu cache
        drop(stack.rb_mut().make_aligned_with::<c64, _>(n, align, f));
    }

    let mut avg_durations = [Duration::ZERO; N_ALGOS];

    let discriminant_to_algo = |i: usize| -> FftAlgo {
        match i {
            0 => FftAlgo::Dif2,
            1 => FftAlgo::Dit2,
            2 => FftAlgo::Dif4,
            3 => FftAlgo::Dit4,
            4 => FftAlgo::Dif8,
            5 => FftAlgo::Dit8,
            6 => FftAlgo::Dif16,
            7 => FftAlgo::Dit16,
            _ => unreachable!(),
        }
    };

    for (i, avg) in (0..N_ALGOS).zip(&mut avg_durations) {
        let algo = discriminant_to_algo(i);

        let (init_n_runs, approx_duration) = {
            let mut n_runs: u128 = 1;

            loop {
                let duration = measure_n_runs(n_runs, algo, &mut buf, &twiddles, stack.rb_mut());

                if duration < MIN_DURATION {
                    n_runs *= 2;
                } else {
                    break (n_runs, duration_div_f64(duration, n_runs as f64));
                }
            }
        };

        let n_runs = (min_bench_duration_per_algo.as_secs_f64() / approx_duration.as_secs_f64())
            .ceil() as u128;
        *avg = if n_runs <= init_n_runs {
            approx_duration
        } else {
            let duration = measure_n_runs(n_runs, algo, &mut buf, &twiddles, stack.rb_mut());
            duration_div_f64(duration, n_runs as f64)
        };
    }

    let best_time = avg_durations.iter().min().unwrap();
    let best_index = avg_durations
        .iter()
        .position(|elem| elem == best_time)
        .unwrap();
    (discriminant_to_algo(best_index), *best_time)
}

/// Ordered FFT plan.
///
/// This type holds a forward and inverse FFT plan and twiddling factors for a specific size.
/// The size must be a power of two, and can be as large as `2^16` (inclusive).
#[derive(Clone)]
pub struct Plan {
    fwd: unsafe fn(*mut c64, *mut c64, *const c64),
    inv: unsafe fn(*mut c64, *mut c64, *const c64),
    twiddles: ABox<[c64]>,
    twiddles_inv: ABox<[c64]>,
    algo: FftAlgo,
}

impl core::fmt::Debug for Plan {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Plan")
            .field("algo", &self.algo)
            .field("fft_size", &self.fft_size())
            .finish()
    }
}

pub(crate) fn get_fn_ptr(
    algo: FftAlgo,
    n: usize,
) -> [unsafe fn(*mut c64, *mut c64, *const c64); 2] {
    use FftAlgo::*;

    let fft = match algo {
        Dif2 => dif2::runtime_fft(),
        Dit2 => dit2::runtime_fft(),
        Dif4 => dif4::runtime_fft(),
        Dit4 => dit4::runtime_fft(),
        Dif8 => dif8::runtime_fft(),
        Dit8 => dit8::runtime_fft(),
        Dif16 => dif16::runtime_fft(),
        Dit16 => dit16::runtime_fft(),
    };

    let idx = n.trailing_zeros() as usize;

    [fft.fwd[idx], fft.inv[idx]]
}

impl Plan {
    /// Returns a new FFT plan for the given vector size, selected by the provided method.
    ///
    /// # Panics
    ///
    /// - Panics if `n` is not a power of two.
    /// - Panics if `n` is greater than `2^16`.
    ///
    /// # Example
    ///
    #[cfg_attr(feature = "std", doc = " ```")]
    #[cfg_attr(not(feature = "std"), doc = " ```ignore")]
    /// use concrete_fft::ordered::{Method, Plan};
    /// use core::time::Duration;
    ///
    /// let plan = Plan::new(4, Method::Measure(Duration::from_millis(10)));
    /// ```
    pub fn new(n: usize, method: Method) -> Self {
        assert!(n.is_power_of_two());
        assert!(n.trailing_zeros() < 17);

        let algo = match method {
            Method::UserProvided(algo) => algo,
            #[cfg(feature = "std")]
            Method::Measure(duration) => {
                measure_fastest(
                    duration,
                    n,
                    DynStack::new(&mut GlobalMemBuffer::new(measure_fastest_scratch(n))),
                )
                .0
            }
        };

        let [fwd, inv] = get_fn_ptr(algo, n);

        let mut twiddles = avec![c64::default(); 2 * n].into_boxed_slice();
        let mut twiddles_inv = avec![c64::default(); 2 * n].into_boxed_slice();
        use FftAlgo::*;
        let r = match algo {
            Dif2 | Dit2 => 2,
            Dif4 | Dit4 => 4,
            Dif8 | Dit8 => 8,
            Dif16 | Dit16 => 16,
        };
        fft_simd::init_wt(r, n, &mut twiddles, &mut twiddles_inv);
        Self {
            fwd,
            inv,
            twiddles,
            algo,
            twiddles_inv,
        }
    }

    /// Returns the vector size of the FFT.
    ///
    /// # Example
    ///
    #[cfg_attr(feature = "std", doc = " ```")]
    #[cfg_attr(not(feature = "std"), doc = " ```ignore")]
    /// use concrete_fft::ordered::{Method, Plan};
    /// use core::time::Duration;
    ///
    /// let plan = Plan::new(4, Method::Measure(Duration::from_millis(10)));
    /// assert_eq!(plan.fft_size(), 4);
    /// ```
    pub fn fft_size(&self) -> usize {
        self.twiddles.len() / 2
    }

    /// Returns the algorithm that's internally used by the FFT.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_fft::ordered::{FftAlgo, Method, Plan};
    ///
    /// let plan = Plan::new(4, Method::UserProvided(FftAlgo::Dif2));
    /// assert_eq!(plan.algo(), FftAlgo::Dif2);
    /// ```
    pub fn algo(&self) -> FftAlgo {
        self.algo
    }

    /// Returns the size and alignment of the scratch memory needed to perform an FFT.
    ///
    /// # Example
    ///
    #[cfg_attr(feature = "std", doc = " ```")]
    #[cfg_attr(not(feature = "std"), doc = " ```ignore")]
    /// use concrete_fft::ordered::{Method, Plan};
    /// use core::time::Duration;
    ///
    /// let plan = Plan::new(4, Method::Measure(Duration::from_millis(10)));
    /// let scratch = plan.fft_scratch().unwrap();
    /// ```
    pub fn fft_scratch(&self) -> Result<StackReq, SizeOverflow> {
        StackReq::try_new_aligned::<c64>(self.fft_size(), CACHELINE_ALIGN)
    }

    /// Performs a forward FFT in place, using the provided stack as scratch space.
    ///
    /// # Example
    ///
    #[cfg_attr(feature = "std", doc = " ```")]
    #[cfg_attr(not(feature = "std"), doc = " ```ignore")]
    /// use concrete_fft::c64;
    /// use concrete_fft::ordered::{Method, Plan};
    /// use dyn_stack::{DynStack, GlobalMemBuffer};
    /// use core::time::Duration;
    ///
    /// let plan = Plan::new(4, Method::Measure(Duration::from_millis(10)));
    ///
    /// let mut memory = GlobalMemBuffer::new(plan.fft_scratch().unwrap());
    /// let stack = DynStack::new(&mut memory);
    ///
    /// let mut buf = [c64::default(); 4];
    /// plan.fwd(&mut buf, stack);
    /// ```
    pub fn fwd(&self, buf: &mut [c64], stack: DynStack) {
        let n = self.fft_size();
        assert_eq!(n, buf.len());
        let (mut scratch, _) = stack.make_aligned_uninit::<c64>(n, CACHELINE_ALIGN);
        let buf = buf.as_mut_ptr();
        let scratch = scratch.as_mut_ptr();
        unsafe { (self.fwd)(buf, scratch as *mut c64, self.twiddles.as_ptr()) }
    }

    /// Performs an inverse FFT in place, using the provided stack as scratch space.
    ///
    /// # Example
    ///
    #[cfg_attr(feature = "std", doc = " ```")]
    #[cfg_attr(not(feature = "std"), doc = " ```ignore")]
    /// use concrete_fft::c64;
    /// use concrete_fft::ordered::{Method, Plan};
    /// use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
    /// use core::time::Duration;
    ///
    /// let plan = Plan::new(4, Method::Measure(Duration::from_millis(10)));
    ///
    /// let mut memory = GlobalMemBuffer::new(plan.fft_scratch().unwrap());
    /// let mut stack = DynStack::new(&mut memory);
    ///
    /// let mut buf = [c64::default(); 4];
    /// plan.fwd(&mut buf, stack.rb_mut());
    /// plan.inv(&mut buf, stack);
    /// ```
    pub fn inv(&self, buf: &mut [c64], stack: DynStack) {
        let n = self.fft_size();
        assert_eq!(n, buf.len());
        let (mut scratch, _) = stack.make_aligned_uninit::<c64>(n, CACHELINE_ALIGN);
        let buf = buf.as_mut_ptr();
        let scratch = scratch.as_mut_ptr();
        unsafe { (self.inv)(buf, scratch as *mut c64, self.twiddles_inv.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dif16::*;
    use crate::dif2::*;
    use crate::dif4::*;
    use crate::dif8::*;
    use crate::dit16::*;
    use crate::dit2::*;
    use crate::dit4::*;
    use crate::dit8::*;
    use crate::fft_simd::init_wt;
    use crate::x86_feature_detected;
    use num_complex::ComplexFloat;
    use rand::random;
    use rustfft::FftPlanner;

    extern crate alloc;
    use alloc::vec;

    #[test]
    fn test_fft() {
        unsafe {
            for (can_run, r, arr) in [
                (true, 2, &DIT2_SCALAR),
                (true, 4, &DIT4_SCALAR),
                (true, 8, &DIT8_SCALAR),
                (true, 16, &DIT16_SCALAR),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("avx"), 2, &DIT2_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("avx"), 4, &DIT4_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("avx"), 8, &DIT8_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("avx"), 16, &DIT16_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("fma"), 2, &DIT2_FMA),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("fma"), 4, &DIT4_FMA),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("fma"), 8, &DIT8_FMA),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("fma"), 16, &DIT16_FMA),
                #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
                (x86_feature_detected!("avx512f"), 4, &DIT4_AVX512),
                #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
                (x86_feature_detected!("avx512f"), 8, &DIT8_AVX512),
                #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
                (x86_feature_detected!("avx512f"), 16, &DIT16_AVX512),
                (true, 2, &DIF2_SCALAR),
                (true, 4, &DIF4_SCALAR),
                (true, 8, &DIF8_SCALAR),
                (true, 16, &DIF16_SCALAR),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("avx"), 2, &DIF2_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("avx"), 4, &DIF4_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("avx"), 8, &DIF8_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("avx"), 16, &DIF16_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("fma"), 2, &DIF2_FMA),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("fma"), 4, &DIF4_FMA),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("fma"), 8, &DIF8_FMA),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (x86_feature_detected!("fma"), 16, &DIF16_FMA),
                #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
                (x86_feature_detected!("avx512f"), 4, &DIF4_AVX512),
                #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
                (x86_feature_detected!("avx512f"), 8, &DIF8_AVX512),
                #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
                (x86_feature_detected!("avx512f"), 16, &DIF16_AVX512),
            ] {
                if can_run {
                    for exp in 0..17 {
                        let n: usize = 1 << exp;
                        let fwd = arr.fwd[n.trailing_zeros() as usize];
                        let inv = arr.inv[n.trailing_zeros() as usize];

                        let mut scratch = vec![c64::default(); n];
                        let mut twiddles = vec![c64::default(); 2 * n];
                        let mut twiddles_inv = vec![c64::default(); 2 * n];

                        init_wt(r, n, &mut twiddles, &mut twiddles_inv);

                        let mut x = vec![c64::default(); n];
                        for z in &mut x {
                            *z = c64::new(random(), random());
                        }

                        let orig = x.clone();

                        fwd(x.as_mut_ptr(), scratch.as_mut_ptr(), twiddles.as_ptr());

                        // compare with rustfft
                        {
                            let mut planner = FftPlanner::new();
                            let plan = planner.plan_fft_forward(n);
                            let mut y = orig.clone();
                            plan.process(&mut y);

                            for (z_expected, z_actual) in y.iter().zip(&x) {
                                assert!((*z_expected - *z_actual).abs() < 1e-12);
                            }
                        }

                        inv(x.as_mut_ptr(), scratch.as_mut_ptr(), twiddles_inv.as_ptr());

                        for z in &mut x {
                            *z /= n as f64;
                        }

                        for (z_expected, z_actual) in orig.iter().zip(&x) {
                            assert!((*z_expected - *z_actual).abs() < 1e-14);
                        }
                    }
                }
            }
        }
    }
}
