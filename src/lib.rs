//! This crate computes the forward and inverse Fourier transform of a given array of some size
//! that is a small-ish power of `2`.
//!
//! # Example
//!
//! ```
//! use binfft::{c64, dif4::{init_twiddles, fwd, inv}, fft_scratch};
//! use dyn_stack::{GlobalMemBuffer, DynStack, ReborrowMut};
//! use num_complex::ComplexFloat;
//!
//! const N: usize = 4;
//!
//! let mut mem = GlobalMemBuffer::new(fft_scratch(N).unwrap());
//! let mut stack = DynStack::new(&mut mem);
//!
//! let mut twiddles = [c64::new(0.0, 0.0); 2 * N];
//!
//! let data = [
//!     c64::new(1.0, 0.0),
//!     c64::new(2.0, 0.0),
//!     c64::new(3.0, 0.0),
//!     c64::new(4.0, 0.0),
//! ];
//!
//! let mut transformed_fwd = data;
//! init_twiddles(true, N, &mut twiddles);
//! fwd(&mut transformed_fwd, &twiddles, stack.rb_mut());
//!
//! let mut transformed_inv = transformed_fwd;
//! init_twiddles(false, N, &mut twiddles);
//! inv(&mut transformed_inv, &twiddles, stack.rb_mut());
//!
//! for (expected, actual) in transformed_inv.iter().map(|z| z / N as f64).zip(data) {
//!     let diff = (expected - actual).abs();
//!     assert!(diff < 1e-9);
//! }
//! ```

use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut, SizeOverflow, StackReq};
use num_complex::Complex;

mod fft_simd;
mod twiddles;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod x86;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "wasm32")]
mod wasm32;

mod dif16;
mod dif4;
mod dif8;

mod boilerplate;
mod dit16;
mod dit4;
mod dit8;

/// Complex type containing two `f64` values.
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;

/// Maximum exponent of the power of two size that we can process.
pub const MAX_EXP: usize = 17;

/// Scratch memory requirements for calling fft functions.
fn fft_scratch(n: usize) -> Result<StackReq, SizeOverflow> {
    StackReq::try_new_aligned::<c64>(n, 64)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FftAlgo {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Inverse,
}

use std::time::{Duration, Instant};

fn init_twiddles(algo: FftAlgo, direction: Direction, n: usize, twiddles: &mut [c64]) {
    match algo {
        FftAlgo::Dif4 => dif4::init_twiddles(direction == Direction::Forward, n, twiddles),
        FftAlgo::Dit4 => dit4::init_twiddles(direction == Direction::Forward, n, twiddles),
        FftAlgo::Dif8 => dif8::init_twiddles(direction == Direction::Forward, n, twiddles),
        FftAlgo::Dit8 => dit8::init_twiddles(direction == Direction::Forward, n, twiddles),
        FftAlgo::Dif16 => dif16::init_twiddles(direction == Direction::Forward, n, twiddles),
        FftAlgo::Dit16 => dit16::init_twiddles(direction == Direction::Forward, n, twiddles),
    }
}

fn fft(algo: FftAlgo, direction: Direction, data: &mut [c64], twiddles: &[c64], stack: DynStack) {
    match direction {
        Direction::Forward => fwd(algo, data, twiddles, stack),
        Direction::Inverse => inv(algo, data, twiddles, stack),
    }
}

fn fwd(algo: FftAlgo, data: &mut [c64], twiddles: &[c64], stack: DynStack) {
    match algo {
        FftAlgo::Dif4 => dif4::fwd(data, twiddles, stack),
        FftAlgo::Dit4 => dit4::fwd(data, twiddles, stack),
        FftAlgo::Dif8 => dif8::fwd(data, twiddles, stack),
        FftAlgo::Dit8 => dit8::fwd(data, twiddles, stack),
        FftAlgo::Dif16 => dif16::fwd(data, twiddles, stack),
        FftAlgo::Dit16 => dit16::fwd(data, twiddles, stack),
    }
}

fn inv(algo: FftAlgo, data: &mut [c64], twiddles: &[c64], stack: DynStack) {
    match algo {
        FftAlgo::Dif4 => dif4::inv(data, twiddles, stack),
        FftAlgo::Dit4 => dit4::inv(data, twiddles, stack),
        FftAlgo::Dif8 => dif8::inv(data, twiddles, stack),
        FftAlgo::Dit8 => dit8::inv(data, twiddles, stack),
        FftAlgo::Dif16 => dif16::inv(data, twiddles, stack),
        FftAlgo::Dit16 => dit16::inv(data, twiddles, stack),
    }
}

fn measure_n_runs(
    n_runs: u128,
    algo: FftAlgo,
    direction: Direction,
    buf: &mut [c64],
    twiddles: &[c64],
    mut stack: DynStack,
) -> Duration {
    let now = Instant::now();

    for _ in 0..n_runs {
        fft(algo, direction, buf, twiddles, stack.rb_mut());
    }

    now.elapsed()
}

fn duration_div_f64(duration: Duration, n: f64) -> Duration {
    Duration::from_secs_f64(duration.as_secs_f64() / n as f64)
}

fn measure_fastest(
    min_bench_duration_per_algo: Duration,
    n: usize,
    direction: Direction,
) -> FftAlgo {
    const N_ALGOS: usize = 6;
    const MIN_DURATION: Duration = Duration::from_millis(1);

    assert!(n.is_power_of_two());
    let i = n.trailing_zeros() as usize;
    assert!(i < MAX_EXP);

    let align = 64;
    let stack_req = StackReq::new_aligned::<c64>(2 * n, align) // twiddles
        .and(StackReq::new_aligned::<c64>(n, align)) // buffer
        .and(fft_scratch(n).unwrap()); // scratch

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
            0 => FftAlgo::Dif4,
            1 => FftAlgo::Dit4,
            2 => FftAlgo::Dif8,
            3 => FftAlgo::Dit8,
            4 => FftAlgo::Dif16,
            5 => FftAlgo::Dit16,
            _ => unreachable!(),
        }
    };

    for (i, avg) in (0..N_ALGOS).zip(&mut avg_durations) {
        let algo = discriminant_to_algo(i);

        let (init_n_runs, approx_duration) = {
            let mut n_runs: u128 = 1;

            loop {
                let duration =
                    measure_n_runs(n_runs, algo, direction, &mut buf, &twiddles, stack.rb_mut());

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
            let duration =
                measure_n_runs(n_runs, algo, direction, &mut buf, &twiddles, stack.rb_mut());
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

#[derive(Clone)]
pub struct Plan {
    fn_ptr: unsafe fn(*mut c64, *mut c64, *const c64),
    twiddles: AlignedBox,
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

impl Plan {
    pub fn new(n: usize, method: Method, direction: Direction) -> Self {
        assert!(n.is_power_of_two());
        assert!((n.trailing_zeros() as usize) < MAX_EXP);
        let algo = match method {
            Method::UserProvided(algo) => algo,
            Method::Measure(duration) => measure_fastest(duration, n, direction),
        };

        let fn_ptr = match algo {
            FftAlgo::Dif4 => dif4::get_fn_ptr(direction, n),
            FftAlgo::Dit4 => dit4::get_fn_ptr(direction, n),
            FftAlgo::Dif8 => dif8::get_fn_ptr(direction, n),
            FftAlgo::Dit8 => dit8::get_fn_ptr(direction, n),
            FftAlgo::Dif16 => dif16::get_fn_ptr(direction, n),
            FftAlgo::Dit16 => dit16::get_fn_ptr(direction, n),
        };

        let mut twiddles = AlignedBox::new_zeroed(2 * n);
        init_twiddles(algo, direction, n, &mut *twiddles);
        Self {
            fn_ptr,
            twiddles,
            algo,
        }
    }

    pub fn len(&self) -> usize {
        self.twiddles.len() / 2
    }

    pub fn algo(&self) -> FftAlgo {
        self.algo
    }

    pub fn fft_scratch(&self) -> Result<StackReq, SizeOverflow> {
        fft_scratch(self.len())
    }

    pub fn fft(&self, buf: &mut [c64], stack: DynStack) {
        let n = self.len();
        assert_eq!(n, buf.len());
        let (mut scratch, _) = stack.make_aligned_uninit::<c64>(n, 64);
        let buf = buf.as_mut_ptr();
        let scratch = scratch.as_mut_ptr();
        unsafe { (self.fn_ptr)(buf, scratch as *mut c64, self.twiddles.ptr) }
    }
}
