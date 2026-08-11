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

mod gfx;
mod shaders;

pub use gfx::{parse_cube, Gfx};
