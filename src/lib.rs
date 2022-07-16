#![allow(clippy::erasing_op, clippy::identity_op)]
#![cfg_attr(feature = "nightly", feature(stdsimd, avx512_target_feature))]

use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut, SizeOverflow, StackReq};
use num_complex::Complex32;
use num_complex::Complex64;
use std::time::{Duration, Instant};

mod fft_simd;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod x86;

mod dif16;
mod dif2;
mod dif4;
mod dif8;

mod dit16;
mod dit2;
mod dit4;
mod dit8;

#[allow(non_camel_case_types)]
pub type c64 = Complex64;
#[allow(non_camel_case_types)]
pub type c32 = Complex32;

type FnArray = [unsafe fn(*mut c64, *mut c64, *const c64); 17];

#[derive(Copy, Clone)]
struct FftImpl {
    fwd: FnArray,
    inv: FnArray,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FftAlgo {
    Dif2,
    Dit2,
    Dif4,
    Dit4,
    Dif8,
    Dit8,
    Dif16,
    Dit16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Method {
    UserProvided(FftAlgo),
    Measure(Duration),
}

struct AlignedBox {
    ptr: *const c64,
    len: usize,
}

impl AlignedBox {
    fn new_zeroed(len: usize) -> Self {
        let buf = GlobalMemBuffer::new(StackReq::new_aligned::<c64>(len, 64));
        let (ptr, _, _) = buf.into_raw_parts();
        let ptr = ptr as *mut c64;
        for i in 0..len {
            unsafe { ptr.add(i).write(c64::default()) };
        }
        Self { ptr, len }
    }
}

impl Drop for AlignedBox {
    fn drop(&mut self) {
        unsafe {
            std::mem::drop(GlobalMemBuffer::from_raw_parts(self.ptr as _, self.len, 64));
        }
    }
}

impl Clone for AlignedBox {
    fn clone(&self) -> Self {
        let len = self.len;
        let src = self.ptr;

        let buf = GlobalMemBuffer::new(StackReq::new_aligned::<c64>(len, 64));
        let (ptr, len, _) = buf.into_raw_parts();
        let dst = ptr as *mut c64;
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
        Self { ptr: dst, len }
    }
}

impl std::ops::Deref for AlignedBox {
    type Target = [c64];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::ops::DerefMut for AlignedBox {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut c64, self.len) }
    }
}

fn measure_n_runs(
    n_runs: u128,
    algo: FftAlgo,
    buf: &mut [c64],
    twiddles: &[c64],
    stack: DynStack,
) -> Duration {
    let n = buf.len();
    let (mut scratch, _) = stack.make_aligned_uninit::<c64>(n, 64);
    let scratch = scratch.as_mut_ptr() as *mut c64;
    let [fwd, _] = get_fn_ptr(algo, n);

    let now = Instant::now();

    for _ in 0..n_runs {
        unsafe {
            fwd(buf.as_mut_ptr(), scratch, twiddles.as_ptr());
        }
    }

    now.elapsed()
}

fn duration_div_f64(duration: Duration, n: f64) -> Duration {
    Duration::from_secs_f64(duration.as_secs_f64() / n as f64)
}

fn measure_fastest(min_bench_duration_per_algo: Duration, n: usize) -> FftAlgo {
    const N_ALGOS: usize = 8;
    const MIN_DURATION: Duration = Duration::from_millis(1);

    assert!(n.is_power_of_two());

    let align = 64;
    let stack_req = StackReq::new_aligned::<c64>(2 * n, align) // twiddles
        .and(StackReq::new_aligned::<c64>(n, align)) // buffer
        .and(StackReq::new_aligned::<c64>(n, align)); // scratch

    let mut mem = GlobalMemBuffer::new(stack_req);
    let stack = DynStack::new(&mut mem);

    let f = |_| c64::default();

    let (twiddles, stack) = stack.make_aligned_with::<c64, _>(2 * n, align, f);
    let (mut buf, mut stack) = stack.make_aligned_with::<c64, _>(n, align, f);

    {
        // initialize scratch to load it in the cpu cache
        stack.rb_mut().make_aligned_with::<c64, _>(n, align, f);
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
    discriminant_to_algo(best_index)
}

#[derive(Clone)]
pub struct Plan {
    fwd: unsafe fn(*mut c64, *mut c64, *const c64),
    inv: unsafe fn(*mut c64, *mut c64, *const c64),
    twiddles: AlignedBox,
    twiddles_inv: AlignedBox,
    algo: FftAlgo,
}

impl std::fmt::Debug for Plan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Plan")
            .field("algo", &self.algo)
            .field("length", &self.len())
            .finish()
    }
}

fn get_fn_ptr(algo: FftAlgo, n: usize) -> [unsafe fn(*mut c64, *mut c64, *const c64); 2] {
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
    pub fn new(n: usize, method: Method) -> Self {
        assert!(n.is_power_of_two());
        assert!((n.trailing_zeros() as usize) < 17);
        let algo = match method {
            Method::UserProvided(algo) => algo,
            Method::Measure(duration) => measure_fastest(duration, n),
        };

        let [fwd, inv] = get_fn_ptr(algo, n);

        let mut twiddles = AlignedBox::new_zeroed(2 * n);
        let mut twiddles_inv = AlignedBox::new_zeroed(2 * n);
        use FftAlgo::*;
        let r = match algo {
            Dif2 | Dit2 => 2,
            Dif4 | Dit4 => 4,
            Dif8 | Dit8 => 8,
            Dif16 | Dit16 => 16,
        };
        fft_simd::init_wt(r, n, &mut *twiddles, &mut *twiddles_inv);
        Self {
            fwd,
            inv,
            twiddles,
            algo,
            twiddles_inv,
        }
    }

    pub fn len(&self) -> usize {
        self.twiddles.len() / 2
    }

    pub fn algo(&self) -> FftAlgo {
        self.algo
    }

    pub fn fft_scratch(&self) -> Result<StackReq, SizeOverflow> {
        StackReq::try_new_aligned::<c64>(self.len(), 64)
    }

    pub fn fwd(&self, buf: &mut [c64], stack: DynStack) {
        let n = self.len();
        assert_eq!(n, buf.len());
        let (mut scratch, _) = stack.make_aligned_uninit::<c64>(n, 64);
        let buf = buf.as_mut_ptr();
        let scratch = scratch.as_mut_ptr();
        unsafe { (self.fwd)(buf, scratch as *mut c64, self.twiddles.ptr) }
    }

    pub fn inv(&self, buf: &mut [c64], stack: DynStack) {
        let n = self.len();
        assert_eq!(n, buf.len());
        let (mut scratch, _) = stack.make_aligned_uninit::<c64>(n, 64);
        let buf = buf.as_mut_ptr();
        let scratch = scratch.as_mut_ptr();
        unsafe { (self.inv)(buf, scratch as *mut c64, self.twiddles_inv.ptr) }
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
    use num_complex::ComplexFloat;
    use rand::random;
    use rustfft::FftPlanner;

    #[test]
    fn test_fft() {
        unsafe {
            for (can_run, r, arr) in [
                (true, 2, &DIT2_SCALAR),
                (true, 4, &DIT4_SCALAR),
                (true, 8, &DIT8_SCALAR),
                (true, 16, &DIT16_SCALAR),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("avx"), 2, &DIT2_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("avx"), 4, &DIT4_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("avx"), 8, &DIT8_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("avx"), 16, &DIT16_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("fma"), 2, &DIT2_FMA),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("fma"), 4, &DIT4_FMA),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("fma"), 8, &DIT8_FMA),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("fma"), 16, &DIT16_FMA),
                #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
                (is_x86_feature_detected!("avx512f"), 4, &DIT4_AVX512),
                #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
                (is_x86_feature_detected!("avx512f"), 8, &DIT8_AVX512),
                #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
                (is_x86_feature_detected!("avx512f"), 16, &DIT16_AVX512),
                (true, 2, &DIF2_SCALAR),
                (true, 4, &DIF4_SCALAR),
                (true, 8, &DIF8_SCALAR),
                (true, 16, &DIF16_SCALAR),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("avx"), 2, &DIF2_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("avx"), 4, &DIF4_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("avx"), 8, &DIF8_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("avx"), 16, &DIF16_AVX),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("fma"), 2, &DIF2_FMA),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("fma"), 4, &DIF4_FMA),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("fma"), 8, &DIF8_FMA),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                (is_x86_feature_detected!("fma"), 16, &DIF16_FMA),
                #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
                (is_x86_feature_detected!("avx512f"), 4, &DIF4_AVX512),
                #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
                (is_x86_feature_detected!("avx512f"), 8, &DIF8_AVX512),
                #[cfg(all(feature = "nightly", any(target_arch = "x86_64", target_arch = "x86")))]
                (is_x86_feature_detected!("avx512f"), 16, &DIF16_AVX512),
            ] {
                if can_run {
                    for exp in 1..17 {
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
