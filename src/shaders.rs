//! WGSL -> SPIR-V compilation at startup using naga (pure Rust,
//! no external tools like glslc). One module, four entry points: the
//! shared vertex stage, the atlas fragment (coverage modulates alpha),
//! the image fragment (the texture IS the color) and the frosted-glass
//! fragment (the blurred scene, sampled by screen position).

pub const WGSL_SRC: &str = r#"
struct Push {
    screen: vec2<f32>,
    // 0 = no grading; otherwise the LUT's edge size N, needed to
    // sample voxel centres: c * (N-1)/N + 0.5/N.
    lut: f32,
    // Exponent on atlas coverage before the alpha multiply (the theme's
    // render.text_gamma). The blend happens in the swapchain's own encoding,
    // not in linear light, so light-on-dark glyph edges land thinner than
    // the same glyph inverted; this is the one knob that lets a theme
    // re-weigh them. 1.0 = raw coverage, exactly yesterday's picture.
    text_gamma: f32,
};

var<push_constant> pc: Push;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) a_pos: vec2<f32>,
    @location(1) a_uv: vec2<f32>,
    @location(2) a_color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    let ndc = a_pos / pc.screen * 2.0 - vec2<f32>(1.0, 1.0);
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = a_uv;
    out.color = a_color;
    return out;
}

@group(0) @binding(0) var t_atlas: texture_2d<f32>;
@group(0) @binding(1) var s_atlas: sampler;
@group(1) @binding(0) var t_lut: texture_3d<f32>;
@group(1) @binding(1) var s_lut: sampler;

// The grading LUT, applied last in every fragment path. Identity when
// pc.lut is zero.
fn grade(c: vec4<f32>) -> vec4<f32> {
    if (pc.lut < 1.5) {
        return c;
    }
    let n = pc.lut;
    let coord = clamp(c.rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * ((n - 1.0) / n)
        + vec3<f32>(0.5 / n);
    let g = textureSample(t_lut, s_lut, coord);
    return vec4<f32>(g.rgb, c.a);
}

@fragment
fn fs_main(
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
) -> @location(0) vec4<f32> {
    let coverage = pow(textureSample(t_atlas, s_atlas, uv).r, pc.text_gamma);
    return grade(vec4<f32>(color.rgb, color.a * coverage));
}

// The frosted-glass fragment: what lies behind, pre-rendered and
// blurred by the pyramid, sampled by SCREEN position — so the glass
// quad may ride any animation and the frost stays put on the picture.
@fragment
fn fs_blur(
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
) -> @location(0) vec4<f32> {
    let suv = pos.xy / pc.screen;
    return grade(textureSample(t_atlas, s_atlas, suv) * color);
}

@fragment
fn fs_image(
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
) -> @location(0) vec4<f32> {
    return grade(textureSample(t_atlas, s_atlas, uv) * color);
}
"#;

pub fn compile() -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(WGSL_SRC)
        .unwrap_or_else(|e| panic!("WGSL compilation error: {e}"));

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("shader validation error: {e:?}"));

    // The shader computes NDC in Vulkan convention (Y down), so we disable
    // the default WebGPU->Vulkan coordinate-space conversion (Y flip).
    let mut options = naga::back::spv::Options::default();
    options
        .flags
        .remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);

    naga::back::spv::write_vec(&module, &info, &options, None)
        .unwrap_or_else(|e| panic!("SPIR-V write error: {e:?}"))
}
