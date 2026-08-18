//! nacelle-renderer — the Vulkan renderer of the nacelle project.
//!
//! It eats the toolkit's draw list — one vertex stream partitioned
//! into runs by texture — and turns it into frames: the glyph atlas
//! pipeline for text and solid shapes, the image pipeline for
//! registered RGBA textures, an optional 3D grading LUT applied last
//! in every fragment path, and a swapchain that honours a preferred
//! bit depth (8, 10, 12 or 16) as far as the surface allows.
//!
//! The renderer knows no windowing library: it takes raw display and
//! window handles and pixel sizes, which is what lets the same crate
//! serve a winit application today and the project's own compositor
//! tomorrow.
//!
//! What it does NOT do is end the process on the machine's behalf.
//! [`Gfx::new`] fails with a [`GfxError`] a caller can print, and a
//! surface that dies mid-session — an unplugged monitor, a restarted
//! compositor — is rebuilt from the handles the constructor was given.

mod gfx;
mod shaders;
/// The shape fragment's own arithmetic, evaluated over naga's IR and
/// checked twice without a device: against plain geometry, and against
/// `nacelle::sdf::d_record`, the specification it claims to mirror.
/// Test-only; nothing ships it.
#[cfg(test)]
mod shape_field;
mod timing;

/// What the shaders cost, counted in instructions rather than in
/// seconds: the same WGSL -> naga -> SPIR-V road the renderer walks at
/// startup, read back and tallied. Public because the measurement is
/// meant to be run — `cargo run --example shader_stats` — and because
/// it answers questions about shaders this crate does not itself carry.
pub mod spirvstat;

pub use gfx::{parse_cube, Gfx, GfxError};
