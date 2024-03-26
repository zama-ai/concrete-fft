#[cfg(target_family = "wasm")]
pub struct Instant {
    start: f64,
}

#[cfg(target_family = "wasm")]
impl Instant {
    pub fn now() -> Self {
        let now = js_sys::Date::new_0().get_time();
        Self { start: now }
    }

    pub fn elapsed(&self) -> core::time::Duration {
        let start: f64 = self.start;
        let now = js_sys::Date::new_0().get_time();

        let mut elapsed = now - start;
        if elapsed < 0. {
            elapsed = 0.00001;
        }
        core::time::Duration::from_micros((elapsed * 1_000.) as u64)
    }
}
