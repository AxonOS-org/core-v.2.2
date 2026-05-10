use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let linker_script = env::var("AXONOS_LINKER_SCRIPT").unwrap_or_else(|_| "link.x".into());
    fs::copy(&linker_script, out.join("link.x")).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed={}", linker_script);
}
