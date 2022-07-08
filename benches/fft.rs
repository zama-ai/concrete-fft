use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::{DynStack, ReborrowMut, StackReq};
use binfft::c64;

pub fn criterion_benchmark(c: &mut Criterion) {
    for n in [128, 256, 512, 1024, 2048, 4096, 8192, 16384] {
        let mut mem = dyn_stack::uninit_mem_in_global(
            binfft::fft_req(n)
                .unwrap()
                .and(StackReq::new_aligned::<c64>(2 * n, 64)) // src | twiddles
                .and(StackReq::new_aligned::<c64>(n, 64)), // dst
        );
        let mut stack = DynStack::new(&mut mem);
        let z = c64::new(0.0, 0.0);

        // rustfft
        {
            use rustfft::FftPlannerAvx;
            let mut planner = FftPlannerAvx::<f64>::new().unwrap();
            let fft = planner.plan_fft_forward(n);
            let mut scratch = [];

            let (mut dst, stack) = stack.rb_mut().make_aligned_with::<c64, _>(n, 64, |_| z);
            let (mut src, _) = stack.make_aligned_with::<c64, _>(n, 64, |_| z);

            c.bench_function(&format!("rustfft-{}", n), |b| {
                b.iter(|| fft.process_outofplace_with_scratch(&mut src, &mut dst, &mut scratch))
            });
        }

        // fftw
        {
            use binfft::{PlanInterleavedC64, Sign};
            let plan = PlanInterleavedC64::new(n, Sign::Forward);

            let (mut dst, stack) = stack.rb_mut().make_aligned_with::<c64, _>(n, 64, |_| z);
            let (mut src, _) = stack.make_aligned_with::<c64, _>(n, 64, |_| z);

            c.bench_function(&format!("fftw-{}", n), |b| {
                b.iter(|| {
                    plan.execute(&mut src, &mut dst);
                })
            });
        }

        // dif4
        {
            let (mut dst, mut stack) = stack.rb_mut().make_aligned_with::<c64, _>(n, 64, |_| z);
            let (w, mut stack) = stack.rb_mut().make_aligned_with::<c64, _>(2 * n, 64, |_| z);

            c.bench_function(&format!("dif4-{}", n), |b| {
                b.iter(|| binfft::dif4::fwd(&mut dst, &w, stack.rb_mut()))
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
