use eyre::{eyre, Result};
use std::env;
use std::fmt::Write;
use std::fs;
use std::path::Path;

static BOILERPLATE: &str = r#"
macro_rules! impl_main_fn {
    ($(#[$attr: meta])* $name: ident, $array_expr: expr) => {
        $(#[$attr])*
        pub fn $name(data: &mut [crate::c64], twiddles: &[crate::c64], stack: dyn_stack::DynStack) {
            let n = data.len();
            let i = n.trailing_zeros() as usize;

            assert!(n.is_power_of_two());
            assert!(i < crate::MAX_EXP);
            assert_eq!(twiddles.len(), 2 * n);

            let (mut scratch, _) = stack.make_aligned_uninit::<crate::c64>(n, 64);
            let scratch = scratch.as_mut_ptr();
            let data = data.as_mut_ptr();
            let w = twiddles.as_ptr();

            unsafe {
                ($array_expr)[i](data, scratch as *mut crate::c64, w);
            }
        }
    };
}

pub fn get_fn_ptr(
    direction: crate::Direction,
    n: usize,
) -> unsafe fn(*mut crate::c64, *mut crate::c64, *const crate::c64) {
    assert!(n.is_power_of_two());
    let array = match direction {
        crate::Direction::Forward => &*FWD_FN_ARRAY,
        crate::Direction::Inverse => &*INV_FN_ARRAY,
    };
    array[n.trailing_zeros() as usize]
}

impl_main_fn!(fwd, &*FWD_FN_ARRAY);
impl_main_fn!(inv, &*INV_FN_ARRAY);

impl_main_fn!(
    #[cfg(target_feature = "fma")]
    #[allow(dead_code)]
    fwd_fma,
    fwd_fn_array_fma()
);
impl_main_fn!(
    #[cfg(target_feature = "fma")]
    #[allow(dead_code)]
    inv_fma,
    inv_fn_array_fma()
);

impl_main_fn!(
    #[cfg(target_feature = "avx")]
    #[allow(dead_code)]
    fwd_avx,
    fwd_fn_array_avx()
);
impl_main_fn!(
    #[cfg(target_feature = "avx")]
    #[allow(dead_code)]
    inv_avx,
    inv_fn_array_avx()
);

impl_main_fn!(
    #[cfg(target_feature = "sse3")]
    #[allow(dead_code)]
    fwd_sse3,
    fwd_fn_array_sse3()
);
impl_main_fn!(
    #[cfg(target_feature = "sse3")]
    #[allow(dead_code)]
    inv_sse3,
    inv_fn_array_sse3()
);

impl_main_fn!(
    #[cfg(target_feature = "sse2")]
    #[allow(dead_code)]
    fwd_sse2,
    fwd_fn_array_sse2()
);
impl_main_fn!(
    #[cfg(target_feature = "sse2")]
    #[allow(dead_code)]
    inv_sse2,
    inv_fn_array_sse2()
);

impl_main_fn!(
    #[cfg(target_feature = "neon")]
    #[allow(dead_code)]
    fwd_neon,
    fwd_fn_array_neon()
);
impl_main_fn!(
    #[cfg(target_feature = "neon")]
    #[allow(dead_code)]
    inv_neon,
    inv_fn_array_neon()
);

impl_main_fn!(
    #[cfg(target_feature = "simd128")]
    #[allow(dead_code)]
    fwd_simd128,
    fwd_fn_array_simd128()
);
impl_main_fn!(
    #[cfg(target_feature = "simd128")]
    #[allow(dead_code)]
    inv_simd128,
    inv_fn_array_simd128()
);

impl_main_fn!(
    #[allow(dead_code)]
    fwd_scalar,
    fwd_fn_array_scalar()
);
impl_main_fn!(
    #[allow(dead_code)]
    inv_scalar,
    inv_fn_array_scalar()
);

#[cfg(test)]
mod tests {
    use super::*;
    use dyn_stack::ReborrowMut;

    fn test_fft_generic(
        n: usize,
        fwd_fn: fn(&mut [crate::c64], &[crate::c64], dyn_stack::DynStack),
        inv_fn: fn(&mut [crate::c64], &[crate::c64], dyn_stack::DynStack),
    ) {
        let z = crate::c64::new(0.0, 0.0);

        let mut arr_fwd = vec![z; n];

        for z in &mut arr_fwd {
            z.re = rand::random();
            z.im = rand::random();
        }

        let mut arr_inv = arr_fwd.clone();
        let mut arr_fwd_expected = arr_fwd.clone();
        let mut arr_inv_expected = arr_fwd.clone();

        let mut w = vec![z; 2 * n];

        let mut mem = dyn_stack::GlobalMemBuffer::new(crate::fft_scratch(n).unwrap());
        let mut stack = dyn_stack::DynStack::new(&mut mem);

        init_twiddles(true, n, &mut w);
        fwd_fn(&mut arr_fwd, &w, stack.rb_mut());
        init_twiddles(false, n, &mut w);
        inv_fn(&mut arr_inv, &w, stack);

        #[cfg(not(miri))]
        {
            let mut fft = rustfft::FftPlanner::new();
            let fwd = fft.plan_fft_forward(n);
            let inv = fft.plan_fft_inverse(n);

            fwd.process(&mut arr_fwd_expected);
            inv.process(&mut arr_inv_expected);

            use num_complex::ComplexFloat;
            for (actual, expected) in arr_fwd.iter().zip(&arr_fwd_expected) {
                assert!((*actual - *expected).abs() < 1e-6);
            }
            for (actual, expected) in arr_inv.iter().zip(&arr_inv_expected) {
                assert!((*actual - *expected).abs() < 1e-6);
            }
        }
    }

    fn test_roundtrip_generic(
        n: usize,
        fwd_fn: fn(&mut [crate::c64], &[crate::c64], dyn_stack::DynStack),
        inv_fn: fn(&mut [crate::c64], &[crate::c64], dyn_stack::DynStack),
    ) {
        let z = crate::c64::new(0.0, 0.0);

        let mut arr_orig = vec![z; n];

        for z in &mut arr_orig {
            z.re = rand::random();
            z.im = rand::random();
        }

        let mut arr_roundtrip = arr_orig.clone();

        let mut w = vec![z; 2 * n];

        let mut mem = dyn_stack::GlobalMemBuffer::new(crate::fft_scratch(n).unwrap());
        let mut stack = dyn_stack::DynStack::new(&mut mem);

        init_twiddles(true, n, &mut w);
        fwd_fn(&mut arr_roundtrip, &w, stack.rb_mut());
        init_twiddles(false, n, &mut w);
        inv_fn(&mut arr_roundtrip, &w, stack);

        for z in &mut arr_roundtrip {
            *z /= n as f64;
        }

        use num_complex::ComplexFloat;
        for (actual, expected) in arr_roundtrip.iter().zip(&arr_orig) {
            assert!((*actual - *expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_fft() {
        for i in 0..crate::MAX_EXP {
            let n = 1usize << i;
            test_fft_generic(n, fwd_scalar, inv_scalar);

            #[cfg(not(miri))]
            {
                #[cfg(target_feature = "fma")]
                test_fft_generic(n, fwd_fma, inv_fma);
                #[cfg(target_feature = "avx")]
                test_fft_generic(n, fwd_avx, inv_avx);
                #[cfg(target_feature = "sse3")]
                test_fft_generic(n, fwd_sse3, inv_sse3);
                #[cfg(target_feature = "sse2")]
                test_fft_generic(n, fwd_sse2, inv_sse2);

                #[cfg(target_feature = "neon")]
                test_fft_generic(n, fwd_neon, inv_neon);

                #[cfg(target_feature = "simd128")]
                test_fft_generic(n, fwd_simd128, inv_simd128);
            }
        }
    }

    #[test]
    fn test_roundtrip() {
        for i in 0..crate::MAX_EXP {
            let n = 1usize << i;
            test_roundtrip_generic(n, fwd_scalar, inv_scalar);

            #[cfg(not(miri))]
            {
                #[cfg(target_feature = "fma")]
                test_roundtrip_generic(n, fwd_fma, inv_fma);
                #[cfg(target_feature = "avx")]
                test_roundtrip_generic(n, fwd_avx, inv_avx);
                #[cfg(target_feature = "sse3")]
                test_roundtrip_generic(n, fwd_sse3, inv_sse3);
                #[cfg(target_feature = "sse2")]
                test_roundtrip_generic(n, fwd_sse2, inv_sse2);

                #[cfg(target_feature = "neon")]
                test_roundtrip_generic(n, fwd_neon, inv_neon);

                #[cfg(target_feature = "simd128")]
                test_roundtrip_generic(n, fwd_simd128, inv_simd128);
            }
        }
    }
}
"#;

const MAX_POW: usize = 17;
const DIRECTIONS: [&str; 2] = ["fwd", "inv"];

fn ascii_to_titlecase(s: &str) -> String {
    s[0..1].to_uppercase() + &s[1..]
}

type GenFn = fn(
    body: &mut String,
    n: usize,
    s: usize,
    base_case: usize,
    factor: usize,
    eo: bool,
    first: &str,
    second: &str,
    direction: &str,
) -> Result<()>;

#[allow(clippy::too_many_arguments)]
fn gen_generic_dif_body(
    body: &mut String,
    n: usize,
    s: usize,
    base_case: usize,
    factor: usize,
    eo: bool,
    first: &str,
    second: &str,
    direction: &str,
) -> Result<()> {
    if n <= base_case {
        if n > 1 {
            write!(
                body,
                "\n    {direction}end_{n}_{}::<S>({n}, {s}, {eo}, {first}, {second});",
                if s == 1 { "1" } else { "s" },
            )?;
        }
    } else {
        write!(
            body,
            "\n    {direction}core_{}::<S>({n}, {s}, {first}, {second}, w);",
            if s == 1 { "1" } else { "s" }
        )?;
        gen_generic_dif_body(
            body,
            n / factor,
            s * factor,
            base_case,
            factor,
            !eo,
            second,
            first,
            direction,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gen_generic_dit_body(
    body: &mut String,
    n: usize,
    s: usize,
    base_case: usize,
    factor: usize,
    eo: bool,
    first: &str,
    second: &str,
    direction: &str,
) -> Result<()> {
    if n <= base_case {
        if n > 1 {
            write!(
                body,
                "\n    {direction}end_{n}_{}::<S>({n}, {s}, {eo}, {first}, {second});",
                if s == 1 { "1" } else { "s" },
            )?;
        }
    } else {
        gen_generic_dit_body(
            body,
            n / factor,
            s * factor,
            base_case,
            factor,
            !eo,
            second,
            first,
            direction,
        )?;
        write!(
            body,
            "\n    {direction}core_{}::<S>({n}, {s}, {first}, {second}, w);",
            if s == 1 { "1" } else { "s" }
        )?;
    }

    Ok(())
}

fn gen_generic_fn(n: usize, factor: usize, gen_fn: GenFn, direction: &str) -> Result<String> {
    assert!(n.is_power_of_two());
    let sig = format!(
        "#[inline(always)]#[allow(unused_variables)]\nunsafe fn {direction}fft_{n}<S: FftSimd16>\
                (x: *mut c64, y: *mut c64, w: *const c64)"
    );

    let mut body = String::new();
    gen_fn(&mut body, n, 1, factor, factor, false, "x", "y", direction)?;

    Ok(format!("{} {{{}\n}}", sig, body))
}

fn gen_runtime_fn(n: usize, direction: &str, arch_module: &str, feat: &str) -> String {
    let feat_struct = ascii_to_titlecase(feat);
    format!(
        "
    {}
    unsafe fn {direction}fft_{n}_{feat}(x: *mut c64, y: *mut c64, w: *const c64) {{
        {direction}fft_{n}::<crate::{arch_module}::{feat_struct}>(x, y, w);
    }}",
        if feat == "scalar" {
            String::new()
        } else {
            format!("#[target_feature(enable = \"{feat}\")]")
        }
    )
}

fn gen_fn_ptr_array(direction: &str, feat: &str) -> Result<String> {
    let mut array = String::new();
    for i in 0..MAX_POW {
        let n = 1usize << i;
        write!(array, "{direction}fft_{n}_{feat}, ")?;
    }
    Ok(format!("[{array}]"))
}

fn gen_runtime_fn_ptr(direction: &str, feat: &str) -> Result<String> {
    Ok(format!(
        "
            fn {direction}_fn_array_{feat}()
            -> [unsafe fn(*mut c64, *mut c64, *const c64); {MAX_POW}] {{ {} }}",
        gen_fn_ptr_array(direction, feat)?,
    ))
}

fn gen_runtime_dispatch_fn_ptr(arch_module: &str, direction: &str) -> Result<String> {
    let big_dir = direction.to_uppercase();

    match arch_module{
        "x86" => Ok(format!(
                "
                static ref {big_dir}_FN_ARRAY: [unsafe fn(*mut c64, *mut c64, *const c64); {MAX_POW}] = {{
                    if is_x86_feature_detected!(\"fma\") {{
                        {direction}_fn_array_fma()
                    }} else if is_x86_feature_detected!(\"avx\") {{
                        {direction}_fn_array_avx()
                    }} else if is_x86_feature_detected!(\"sse3\") {{
                        {direction}_fn_array_sse3()
                    }} else if is_x86_feature_detected!(\"sse2\") {{
                        {direction}_fn_array_sse2()
                    }} else {{
                        {direction}_fn_array_scalar()
                    }}
                }};",
                )),
        "aarch64" => Ok(format!(
                "
                static ref {big_dir}_FN_ARRAY: [unsafe fn(*mut c64, *mut c64, *const c64); {MAX_POW}] = {{
                    if is_aarch64_feature_detected!(\"neon\") {{
                        {direction}_fn_array_neon()
                    }} else {{
                        {direction}_fn_array_scalar()
                    }}
                }};",
                )),
        "wasm32" => Ok(format!(
                "
                static ref {big_dir}_FN_ARRAY: [unsafe fn(*mut c64, *mut c64, *const c64); {MAX_POW}] = {{
                    #[cfg(target_feature = \"simd128\")]
                    {{ {direction}_fn_array_simd128() }}
                    #[cfg(not(target_feature = \"simd128\"))]
                    {{ {direction}_fn_array_scalar() }}
                }};",
                )),
        _ => unreachable!(),
    }
}

fn write_to(
    dest_path: &Path,
    factor: usize,
    gen_fn: GenFn,
    arch_module: &str,
    features: &[&str],
) -> Result<()> {
    let mut code = String::new();

    for i in 0..MAX_POW {
        let n = 1usize << i;

        for direction in DIRECTIONS {
            writeln!(code, "{}", gen_generic_fn(n, factor, gen_fn, direction)?)?;
            for feat in features.iter().cloned() {
                writeln!(code, "{}", gen_runtime_fn(n, direction, arch_module, feat))?;
            }
        }
    }

    for direction in DIRECTIONS {
        for feat in features.iter().cloned() {
            code.push_str(&gen_runtime_fn_ptr(direction, feat)?);
        }
    }

    code.push_str("\nlazy_static::lazy_static! {");
    for direction in DIRECTIONS {
        code.push_str(&gen_runtime_dispatch_fn_ptr(arch_module, direction)?);
    }
    code.push_str("\n}");
    code.push_str(BOILERPLATE);

    fs::write(dest_path, &code)?;

    Ok(())
}

fn main() -> Result<()> {
    let out_dir = env::var_os("OUT_DIR").ok_or_else(|| eyre!("couldn't find OUT_DIR"))?;
    let out_dir = Path::new(&out_dir);
    let arch_name =
        env::var_os("CARGO_CFG_TARGET_ARCH").ok_or_else(|| eyre!("couldn't detect target arch"))?;

    let (arch_module, features) = match arch_name
        .to_str()
        .ok_or_else(|| eyre!("invalid target arch"))?
    {
        "x86_64" | "x86" => ("x86", vec!["scalar", "sse2", "sse3", "avx", "fma"]),
        "aarch64" => ("aarch64", vec!["scalar", "neon"]),
        "wasm32" => ("wasm32", vec!["scalar", "simd128"]),
        _ => ("fft_simd", vec!["scalar"]),
    };

    write_to(
        &out_dir.join("dif4.rs"),
        4,
        gen_generic_dif_body,
        arch_module,
        &features,
    )?;

    write_to(
        &out_dir.join("dit4.rs"),
        4,
        gen_generic_dit_body,
        arch_module,
        &features,
    )?;

    write_to(
        &out_dir.join("dif8.rs"),
        8,
        gen_generic_dif_body,
        arch_module,
        &features,
    )?;

    write_to(
        &out_dir.join("dit8.rs"),
        8,
        gen_generic_dit_body,
        arch_module,
        &features,
    )?;

    write_to(
        &out_dir.join("dif16.rs"),
        16,
        gen_generic_dif_body,
        arch_module,
        &features,
    )?;

    write_to(
        &out_dir.join("dit16.rs"),
        16,
        gen_generic_dit_body,
        arch_module,
        &features,
    )?;

    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
