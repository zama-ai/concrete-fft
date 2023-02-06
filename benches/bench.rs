use concrete_fft::c64;
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
        let mut mem = dyn_stack::GlobalMemBuffer::new(
            StackReq::new_aligned::<c64>(n, 64) // scratch
                .and(
                    StackReq::new_aligned::<c64>(2 * n, 64).or(StackReq::new_aligned::<c64>(n, 64)), // src | twiddles
                )
                .and(StackReq::new_aligned::<c64>(n, 64)), // dst
        );
        let mut stack = DynStack::new(&mut mem);
        let z = c64::new(0.0, 0.0);

        {
            let mut scratch = [];
            let bench_duration = std::time::Duration::from_millis(10);

            {
                let (mut dst, stack) = stack.rb_mut().make_aligned_with::<c64, _>(n, 64, |_| z);
                let (mut src, _) = stack.make_aligned_with::<c64, _>(n, 64, |_| z);

                c.bench_function(&format!("rustfft-fwd-{n}"), |b| {
                    use rustfft::FftPlannerAvx;
                    let mut planner = FftPlannerAvx::<f64>::new().unwrap();
                    let fwd_rustfft = planner.plan_fft_forward(n);

                    b.iter(|| {
                        fwd_rustfft.process_outofplace_with_scratch(
                            &mut src,
                            &mut dst,
                            &mut scratch,
                        )
                    })
                });

                c.bench_function(&format!("fftw-fwd-{n}"), |b| {
                    let fwd_fftw = PlanInterleavedC64::new(n, Sign::Forward);

                    b.iter(|| {
                        fwd_fftw.execute(&mut src, &mut dst);
                    })
                });
            }
            {
                let (mut dst, mut stack) = stack.rb_mut().make_aligned_with::<c64, _>(n, 64, |_| z);

                c.bench_function(&format!("concrete-fwd-{n}"), |b| {
                    let ordered = concrete_fft::ordered::Plan::new(
                        n,
                        concrete_fft::ordered::Method::Measure(bench_duration),
                    );

                    b.iter(|| ordered.fwd(&mut dst, stack.rb_mut()))
                });
            }
            {
                let (mut dst, mut stack) = stack.rb_mut().make_aligned_with::<c64, _>(n, 64, |_| z);

                c.bench_function(&format!("unordered-fwd-{n}"), |b| {
                    let unordered = concrete_fft::unordered::Plan::new(
                        n,
                        concrete_fft::unordered::Method::Measure(bench_duration),
                    );

                    b.iter(|| unordered.fwd(&mut dst, stack.rb_mut()));
                });
            }
            {
                let (mut dst, mut stack) = stack.rb_mut().make_aligned_with::<c64, _>(n, 64, |_| z);

                c.bench_function(&format!("unordered-inv-{n}"), |b| {
                    let unordered = concrete_fft::unordered::Plan::new(
                        n,
                        concrete_fft::unordered::Method::Measure(bench_duration),
                    );

                    b.iter(|| unordered.inv(&mut dst, stack.rb_mut()));
                });
            }
        }

        // memcpy
        {
            let (mut dst, stack) = stack.rb_mut().make_aligned_with::<c64, _>(n, 64, |_| z);
            let (src, _) = stack.make_aligned_with::<c64, _>(n, 64, |_| z);

            c.bench_function(&format!("memcpy-{n}"), |b| {
                b.iter(|| unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), n);
                })
            });
        }
    }

    #[cfg(feature = "fft128")]
    for n in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384] {
        use concrete_fft::fft128::*;
        let twid_re0 = vec![0.0; n];
        let twid_re1 = vec![0.0; n];
        let twid_im0 = vec![0.0; n];
        let twid_im1 = vec![0.0; n];

        let mut data_re0 = vec![0.0; n];
        let mut data_re1 = vec![0.0; n];
        let mut data_im0 = vec![0.0; n];
        let mut data_im1 = vec![0.0; n];

        c.bench_function(&format!("concrete-fft128-fwd-{n}"), |bench| {
            bench.iter(|| {
                negacyclic_fwd_fft_scalar(
                    &mut data_re0,
                    &mut data_re1,
                    &mut data_im0,
                    &mut data_im1,
                    &twid_re0,
                    &twid_re1,
                    &twid_im0,
                    &twid_im1,
                );
            });
        });

        c.bench_function(&format!("concrete-fft128-inv-{n}"), |bench| {
            bench.iter(|| {
                negacyclic_inv_fft_scalar(
                    &mut data_re0,
                    &mut data_re1,
                    &mut data_im0,
                    &mut data_im1,
                    &twid_re0,
                    &twid_re1,
                    &twid_im0,
                    &twid_im1,
                );
            });
        });

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(simd) = Avx::try_new() {
            c.bench_function(&format!("concrete-fft128-avx-fwd-{n}"), |bench| {
                bench.iter(|| {
                    negacyclic_fwd_fft_avxfma(
                        simd,
                        &mut data_re0,
                        &mut data_re1,
                        &mut data_im0,
                        &mut data_im1,
                        &twid_re0,
                        &twid_re1,
                        &twid_im0,
                        &twid_im1,
                    );
                });
            });
            c.bench_function(&format!("concrete-fft128-avx-inv-{n}"), |bench| {
                bench.iter(|| {
                    negacyclic_inv_fft_avxfma(
                        simd,
                        &mut data_re0,
                        &mut data_re1,
                        &mut data_im0,
                        &mut data_im1,
                        &twid_re0,
                        &twid_re1,
                        &twid_im0,
                        &twid_im1,
                    );
                });
            });
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg(feature = "nightly")]
        if let Some(simd) = Avx512::try_new() {
            c.bench_function(&format!("concrete-fft128-avx512-fwd-{n}"), |bench| {
                bench.iter(|| {
                    negacyclic_fwd_fft_avx512(
                        simd,
                        &mut data_re0,
                        &mut data_re1,
                        &mut data_im0,
                        &mut data_im1,
                        &twid_re0,
                        &twid_re1,
                        &twid_im0,
                        &twid_im1,
                    );
                });
            });
            c.bench_function(&format!("concrete-fft128-avx512-inv-{n}"), |bench| {
                bench.iter(|| {
                    negacyclic_inv_fft_avx512(
                        simd,
                        &mut data_re0,
                        &mut data_re1,
                        &mut data_im0,
                        &mut data_im1,
                        &twid_re0,
                        &twid_re1,
                        &twid_im0,
                        &twid_im1,
                    );
                });
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
