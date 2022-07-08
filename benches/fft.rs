use binfft::c64;
use core::ptr::NonNull;
use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::{DynStack, ReborrowMut, StackReq};

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
    for n in [128, 256, 512, 1024, 2048, 4096, 8192, 16384] {
        let mut mem = dyn_stack::uninit_mem_in_global(
            binfft::fft_scratch(n)
                .unwrap()
                .and(
                    StackReq::new_aligned::<c64>(2 * n, 64).or(StackReq::new_aligned::<c64>(n, 64)), // src | twiddles
                )
                .and(StackReq::new_aligned::<c64>(n, 64)), // dst
        );
        let mut stack = DynStack::new(&mut mem);
        let z = c64::new(0.0, 0.0);

        // rustfft
        {
            use rustfft::FftPlannerAvx;
            let mut planner = FftPlannerAvx::<f64>::new().unwrap();
            let fwd = planner.plan_fft_forward(n);
            let inv = planner.plan_fft_inverse(n);
            let mut scratch = [];

            let (mut dst, stack) = stack.rb_mut().make_aligned_with::<c64, _>(n, 64, |_| z);
            let (mut src, _) = stack.make_aligned_with::<c64, _>(n, 64, |_| z);

            c.bench_function(&format!("rustfft-fwd-{}", n), |b| {
                b.iter(|| fwd.process_outofplace_with_scratch(&mut src, &mut dst, &mut scratch))
            });
            c.bench_function(&format!("rustfft-inv-{}", n), |b| {
                b.iter(|| inv.process_outofplace_with_scratch(&mut src, &mut dst, &mut scratch))
            });
        }

        // fftw
        {
            let fwd = PlanInterleavedC64::new(n, Sign::Forward);
            let inv = PlanInterleavedC64::new(n, Sign::Backward);

            let (mut dst, stack) = stack.rb_mut().make_aligned_with::<c64, _>(n, 64, |_| z);
            let (mut src, _) = stack.make_aligned_with::<c64, _>(n, 64, |_| z);

            c.bench_function(&format!("fftw-fwd-{}", n), |b| {
                b.iter(|| {
                    fwd.execute(&mut src, &mut dst);
                })
            });
            c.bench_function(&format!("fftw-inv-{}", n), |b| {
                b.iter(|| {
                    inv.execute(&mut src, &mut dst);
                })
            });
        }

        // binfft
        {
            let (mut dst, mut stack) = stack.rb_mut().make_aligned_with::<c64, _>(n, 64, |_| z);
            let (w, mut stack) = stack.rb_mut().make_aligned_with::<c64, _>(2 * n, 64, |_| z);

            c.bench_function(&format!("dif4-fwd-{}", n), |b| {
                b.iter(|| binfft::dif4::fwd(&mut dst, &w, stack.rb_mut()))
            });
            c.bench_function(&format!("dif4-inv-{}", n), |b| {
                b.iter(|| binfft::dif4::inv(&mut dst, &w, stack.rb_mut()))
            });

            c.bench_function(&format!("dit4-fwd-{}", n), |b| {
                b.iter(|| binfft::dit4::fwd(&mut dst, &w, stack.rb_mut()))
            });
            c.bench_function(&format!("dit4-inv-{}", n), |b| {
                b.iter(|| binfft::dit4::inv(&mut dst, &w, stack.rb_mut()))
            });

            c.bench_function(&format!("dif8-fwd-{}", n), |b| {
                b.iter(|| binfft::dif8::fwd(&mut dst, &w, stack.rb_mut()))
            });
            c.bench_function(&format!("dif8-inv-{}", n), |b| {
                b.iter(|| binfft::dif8::inv(&mut dst, &w, stack.rb_mut()))
            });
        }

        // memcpy
        {
            let (mut dst, stack) = stack.rb_mut().make_aligned_with::<c64, _>(n, 64, |_| z);
            let (src, _) = stack.make_aligned_with::<c64, _>(n, 64, |_| z);

            c.bench_function(&format!("memcpy-{}", n), |b| {
                b.iter(|| unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), n);
                })
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
