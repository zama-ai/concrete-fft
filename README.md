This crate computes the forward and inverse Fourier transform of a given array of some size
that is a small-ish power of `2`.

# Example

```rust
use binfft::{c64, init_twiddles, fwd, inv, fft_scratch};
use dyn_stack::{uninit_mem_in_global, DynStack, ReborrowMut};
use num_complex::ComplexFloat;

const N: usize = 4;

let mut mem = uninit_mem_in_global(fft_scratch(N).unwrap());
let mut stack = DynStack::new(&mut mem);

let mut twiddles = [c64::new(0.0, 0.0); 2 * N];
init_twiddles(N, &mut twiddles);

let data = [
    c64::new(1.0, 0.0),
    c64::new(2.0, 0.0),
    c64::new(3.0, 0.0),
    c64::new(4.0, 0.0),
];

let mut transformed_fwd = data;
fwd(&mut transformed_fwd, &twiddles, stack.rb_mut());

let mut transformed_inv = transformed_fwd;
inv(&mut transformed_inv, &twiddles, stack.rb_mut());

for (expected, actual) in transformed_inv.iter().map(|z| z / N as f64).zip(data) {
    let diff = (expected - actual).abs();
    assert!(diff < 1e-9);
}
```
