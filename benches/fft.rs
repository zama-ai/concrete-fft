use concrete_fft::{c64, unordered::MonomialPlan};
use core::ptr::NonNull;
use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::{PodStack, ReborrowMut, StackReq};

struct FftwAlloc {
    bytes: NonNull<core::ffi::c_void>,
}

impl Drop for FftwAlloc {
    fn drop(&mut self) {
        unsafe {
            fftw_sys::fftw_free(self.bytes.as_ptr());
        }
    }
}

impl FftwAlloc {
    pub fn new(size_bytes: usize) -> FftwAlloc {
        unsafe {
            let bytes = fftw_sys::fftw_malloc(size_bytes);
            if bytes.is_null() {
                use std::alloc::{handle_alloc_error, Layout};
                handle_alloc_error(Layout::from_size_align_unchecked(size_bytes, 1));
            }
            FftwAlloc {
                bytes: NonNull::new_unchecked(bytes),
            }
        }
    }
}

pub struct PlanInterleavedC64 {
    plan: fftw_sys::fftw_plan,
    n: usize,
}

impl Drop for PlanInterleavedC64 {
    fn drop(&mut self) {
        unsafe {
            fftw_sys::fftw_destroy_plan(self.plan);
        }
    }
}

pub enum Sign {
    Forward,
    Backward,
}

impl PlanInterleavedC64 {
    pub fn new(n: usize, sign: Sign) -> Self {
        let size_bytes = n.checked_mul(core::mem::size_of::<c64>()).unwrap();
        let src = FftwAlloc::new(size_bytes);
        let dst = FftwAlloc::new(size_bytes);
        unsafe {
            let p = fftw_sys::fftw_plan_dft_1d(
                n.try_into().unwrap(),
                src.bytes.as_ptr() as _,
                dst.bytes.as_ptr() as _,
                match sign {
                    Sign::Forward => fftw_sys::FFTW_FORWARD as _,
                    Sign::Backward => fftw_sys::FFTW_BACKWARD as _,
                },
                fftw_sys::FFTW_MEASURE,
            );
            PlanInterleavedC64 { plan: p, n }
        }
    }

    pub fn print(&self) {
        unsafe {
            fftw_sys::fftw_print_plan(self.plan);
        }
    }

    pub fn execute(&self, src: &mut [c64], dst: &mut [c64]) {
        assert_eq!(src.len(), self.n);
        assert_eq!(dst.len(), self.n);
        let src = src.as_mut_ptr();
        let dst = dst.as_mut_ptr();
        unsafe {
            use fftw_sys::{fftw_alignment_of, fftw_execute_dft};
            assert_eq!(fftw_alignment_of(src as _), 0);
            assert_eq!(fftw_alignment_of(dst as _), 0);
            fftw_execute_dft(self.plan, src as _, dst as _);
        }
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    for n in [
        1 << 8,
        1 << 9,
        1 << 10,
        1 << 11,
        1 << 12,
        1 << 13,
        1 << 14,
        1 << 15,
        1 << 16,
    ] {
        let mut mem = dyn_stack::GlobalPodBuffer::new(StackReq::all_of([
            StackReq::new_aligned::<c64>(2 * n, 256), // scratch
            StackReq::new_aligned::<c64>(n, 256),     // src
            StackReq::new_aligned::<c64>(n, 256),     // dst
        ]));
        let mut stack = PodStack::new(&mut mem);
        let z = c64::new(0.0, 0.0);

        use rustfft::FftPlannerAvx;
        let mut scratch = [];

        let bench_duration = std::time::Duration::from_millis(10);
        let unordered = concrete_fft::unordered::Plan::new(
            n,
            concrete_fft::unordered::Method::Measure(bench_duration),
        );

        let (mut dst, stack) = stack.rb_mut().make_aligned_with::<c64, _>(n, 64, |_| z);
        let (mut src, mut stack) = stack.make_aligned_with::<c64, _>(n, 64, |_| z);

        c.bench_function(&format!("rustfft-fwd-{n}"), |b| {
            let mut planner = FftPlannerAvx::<f64>::new().unwrap();
            let fwd_rustfft = planner.plan_fft_forward(n);
            b.iter(|| fwd_rustfft.process_outofplace_with_scratch(&mut src, &mut dst, &mut scratch))
        });

        c.bench_function(&format!("fftw-fwd-{n}"), |b| {
            let fwd_fftw = PlanInterleavedC64::new(n, Sign::Forward);
            b.iter(|| {
                fwd_fftw.execute(&mut src, &mut dst);
            })
        });
        if n <= 1024 {
            let ordered = concrete_fft::ordered::Plan::new(
                n,
                concrete_fft::ordered::Method::Measure(bench_duration),
            );

            c.bench_function(&format!("concrete-fwd-{n}"), |b| {
                b.iter(|| ordered.fwd(&mut dst, stack.rb_mut()))
            });
        }
        c.bench_function(&format!("unordered-fwd-{n}"), |b| {
            b.iter(|| unordered.fwd(&mut dst, stack.rb_mut()));
        });
        c.bench_function(&format!("unordered-inv-{n}"), |b| {
            b.iter(|| unordered.inv(&mut dst, stack.rb_mut()));
        });

        // memcpy
        c.bench_function(&format!("memcpy-{n}"), |b| {
            b.iter(|| unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), n);
            })
        });

        let monomial_plan = MonomialPlan::new(n, unordered.algo().1);
        c.bench_function(&format!("fwd-monomial-{n}"), |b| {
            let mut degree = 0;
            b.iter(|| {
                degree += 1;
                if degree == n {
                    degree = 0;
                }
                monomial_plan.fwd_monomial(degree, &mut dst);
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
