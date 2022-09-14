use rustc_version::{version_meta, Channel};

fn main() {
    if matches!(version_meta().map(|x| x.channel), Ok(Channel::Nightly)) {
        println!("cargo:rustc-cfg=feature=\"nightly\"");
    }
}
