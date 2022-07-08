use crate::c64;

#[inline(always)]
pub unsafe fn twid(r: usize, big_n: usize, k: usize, w: *const c64, p: usize) -> *const c64 {
    w.add(p + (k - 1) * (big_n / r))
}

#[inline(always)]
pub unsafe fn twid_t(r: usize, big_n: usize, k: usize, w: *const c64, p: usize) -> *const c64 {
    &*w.add(r * p + (big_n + k))
}

pub unsafe fn init_wt(r: usize, big_n: usize, w: *mut c64) {
    if big_n < r {
        return;
    }

    let nr = big_n / r;
    let theta = -2.0 * core::f64::consts::PI / big_n as f64;

    for i in 0..2 * big_n {
        (*w.add(i)).re = f64::NAN;
        (*w.add(i)).im = f64::NAN;
    }

    for p in 0..nr {
        for k in 1..r {
            let (s, c) = (theta * (k * p) as f64).sin_cos();
            let z = c64::new(c, s);
            *w.add(p + (k - 1) * nr) = z;
            *w.add(big_n + r * p + k) = z;
        }
    }
}
