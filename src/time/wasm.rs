#[cfg(target_family = "wasm")]
pub struct Instant {
    start: f64,
}

#[cfg(target_family = "wasm")]
impl Instant {
    pub fn now() -> Self {
        let window = web_sys::window().unwrap();
        let performance = window.performance().unwrap();
        let start: f64 = performance.now();
        Self { start }
    }

    pub fn elapsed(&self) -> core::time::Duration {
        let window = web_sys::window().unwrap();
        let performance = window.performance().unwrap();
        let start: f64 = self.start;
        let now = performance.now();
        let mut elapsed = now - start;
        // Performance api is not always exact, it will shift the time for a few milliseconds every minute
        if elapsed < 0. {
            // pick a relatively low value, 1us
            elapsed = 0.001;
        }
        core::time::Duration::from_micros((elapsed * 1_000.) as u64)
    }
}
