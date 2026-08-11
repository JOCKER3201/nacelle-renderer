# nacelle-renderer

The Vulkan renderer of the nacelle project. It consumes the draw list
of [libnacelle](https://github.com/JOCKER3201/libnacelle) — one
vertex stream partitioned into runs by texture — and renders it with
two pipelines: the glyph atlas (text and solid shapes) and registered
RGBA images. On top of that: a preferred swapchain bit depth
(8/10/12/16, falling back to what the surface offers), and an optional
3D grading LUT (.cube) applied in every fragment path.

The renderer takes raw display/window handles and pixel sizes — no
windowing library — so the same crate serves the desktop application
today and the project's own compositor later.

Part of the nacelle project:
[libnacelle](https://github.com/JOCKER3201/libnacelle) ·
[nacelle-widgets](https://github.com/JOCKER3201/nacelle-widgets) ·
[nacelle-themes](https://github.com/JOCKER3201/nacelle-themes) ·
[nacelle-desktop](https://github.com/JOCKER3201/nacelle-desktop)

Developed with the assistance of Claude (Anthropic) — AI-generated
code, reviewed and directed by the project author.
