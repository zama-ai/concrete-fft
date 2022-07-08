use eyre::{eyre, Result};
use std::env;
use std::fmt::Write;
use std::fs;
use std::path::Path;

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
