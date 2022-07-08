use eyre::{eyre, Result};
use std::env;
use std::ffi::OsStr;
use std::fmt::Write;
use std::fs;
use std::mem::swap;
use std::path::Path;

const MAX_POW: usize = 17;
const DIRECTIONS: [&str; 2] = ["fwd", "inv"];
const ARCHS: [&str; 5] = ["scalar", "sse2", "sse3", "avx", "fma"];

fn ascii_to_titlecase(s: &str) -> String {
    s[0..1].to_uppercase() + &s[1..]
}

fn gen_generic_dif4_fn(n: usize, direction: &str) -> Result<String> {
    assert!(n.is_power_of_two());
    let sig = format!(
        "#[inline(always)]#[allow(unused_variables)]\nunsafe fn {direction}fft_{n}<S: FftSimd16>\
                (x: *mut c64, y: *mut c64, w: *const c64)"
    );

    let mut body = String::new();
    let mut first = "x";
    let mut second = "y";
    let mut eo = false;
    let mut n = n;
    let mut s: usize = 1;

    while n > 4 {
        write!(
            body,
            "\n    {direction}core_{}::<S>({n}, {s}, {first}, {second}, w);",
            if s == 1 { "1" } else { "s" }
        )?;

        n /= 4;
        s *= 4;
        swap(&mut first, &mut second);
        eo = !eo;
    }

    if n > 1 {
        write!(
            body,
            "\n    {direction}end_{n}_{}::<S>({n}, {s}, {eo}, {first}, {second});",
            if s == 1 { "1" } else { "s" },
        )?;
    }

    Ok(format!("{} {{{}\n}}", sig, body))
}

fn gen_runtime_dif4_fn(n: usize, direction: &str, arch: &str) -> String {
    let arch_struct = ascii_to_titlecase(arch);
    format!(
        "
    {}
    unsafe fn {direction}fft_{n}_{arch}(x: *mut c64, y: *mut c64, w: *const c64) {{
        {direction}fft_{n}::<crate::x86::{arch_struct}>(x, y, w);
    }}",
        if arch == "scalar" {
            String::new()
        } else {
            format!("#[target_feature(enable = \"{arch}\")]")
        }
    )
}

fn gen_fn_ptr_array(direction: &str, arch: &str) -> Result<String> {
    let mut array = String::new();
    for i in 0..MAX_POW {
        let n = 1usize << i;
        write!(array, "{direction}fft_{n}_{arch}, ")?;
    }
    Ok(format!("[{array}]"))
}

fn gen_runtime_fn_ptr(direction: &str, arch: &str) -> Result<String> {
    Ok(format!(
        "
            fn {direction}_fn_array_{arch}()
            -> [unsafe fn(*mut c64, *mut c64, *const c64); {MAX_POW}] {{ {} }}",
        gen_fn_ptr_array(direction, arch)?,
    ))
}

fn gen_runtime_dispatch_fn_ptr(direction: &str) -> String {
    let big_dir = direction.to_uppercase();
    format!(
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
    )
}

fn write_dif4(out_dir: &OsStr) -> Result<()> {
    let dest_path = Path::new(out_dir).join("dif4.rs");

    let mut code = String::new();

    for i in 0..MAX_POW {
        let n = 1usize << i;

        for direction in DIRECTIONS {
            writeln!(code, "{}", gen_generic_dif4_fn(n, direction)?)?;
            for arch in ARCHS {
                writeln!(code, "{}", gen_runtime_dif4_fn(n, direction, arch))?;
            }
        }
    }

    for direction in DIRECTIONS {
        for arch in ARCHS {
            code.push_str(&gen_runtime_fn_ptr(direction, arch)?);
        }
    }

    code.push_str("\nlazy_static::lazy_static! {");
    for direction in DIRECTIONS {
        code.push_str(&gen_runtime_dispatch_fn_ptr(direction));
    }
    code.push_str("\n}");

    fs::write(&dest_path, &code)?;

    Ok(())
}

fn main() -> Result<()> {
    let out_dir = env::var_os("OUT_DIR").ok_or_else(|| eyre!("couldn't find OUT_DIR"))?;

    write_dif4(&out_dir)?;

    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
