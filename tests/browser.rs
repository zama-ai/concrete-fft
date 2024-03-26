use wasm_bindgen_test::wasm_bindgen_test_configure;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test::wasm_bindgen_test]
pub fn measure_n_runs_infinite_loop() {
    use concrete_fft::c64;
    use concrete_fft::unordered::{Method, Plan};
    use core::time::Duration;
    use dyn_stack::{GlobalPodBuffer, PodStack};

    let plan = Plan::new(4, Method::Measure(Duration::from_millis(10)));

    let mut memory = GlobalPodBuffer::new(plan.fft_scratch().unwrap());
    let stack = PodStack::new(&mut memory);

    let mut buf = [c64::default(); 4];
    plan.fwd(&mut buf, stack);
}
