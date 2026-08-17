//! What the shaders cost, in instructions, on any machine.
//!
//!     cargo run --example shader_stats                # this branch's shaders
//!     cargo run --example shader_stats -- a.wgsl      # a WGSL source
//!     cargo run --example shader_stats -- other.rs    # a shaders.rs, WGSL lifted out
//!
//! "What does the shape lane cost against the atlas lane" is now the
//! FIRST form: `fs_shape` and `fs_main` ship in the same module, so
//! one report holds both and the ratio is read off it directly.
//!
//! The third form is for a shader this tree does not carry — an older
//! commit, a branch, a sketch — and it needs neither checkout nor a
//! `.wgsl` file, because the WGSL is lifted out of the Rust source:
//!
//!     git show <rev>:src/shaders.rs > /tmp/other.rs
//!     cargo run --example shader_stats -- /tmp/other.rs
//!
//! No Vulkan, no device, no display: the road is WGSL -> naga ->
//! SPIR-V, exactly the one `shaders::compile()` walks at startup, and
//! then the words are read back and counted. See `spirvstat`'s header
//! for what such a number does and does not mean.

use nacelle_renderer::spirvstat;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        let module = spirvstat::builtin().expect("the renderer's own shaders");
        println!("=== built in (this branch's src/shaders.rs) ===");
        print!("{}", spirvstat::report(&module));
        return;
    }

    let mut failed = false;
    for path in &args {
        println!("=== {path} ===");
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                println!("cannot read: {e}\n");
                failed = true;
                continue;
            }
        };
        // A Rust source carries the shader as a raw string; a .wgsl
        // file is already the shader. Deciding by content rather than
        // by extension means a file piped in under any name works.
        let src = spirvstat::wgsl_from_rust(&text).unwrap_or(&text);
        match spirvstat::from_wgsl(src) {
            Ok(module) => print!("{}", spirvstat::report(&module)),
            Err(e) => {
                println!("{e}");
                failed = true;
            }
        }
        println!();
    }

    if failed {
        std::process::exit(1);
    }
}
