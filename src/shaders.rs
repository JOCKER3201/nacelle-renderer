//! WGSL -> SPIR-V compilation at startup using naga (pure Rust,
//! no external tools like glslc). One module, six entry points: the
//! shared vertex stage, the atlas fragment (coverage modulates alpha),
//! the image fragment (the texture IS the color), the frosted-glass
//! fragment (the blurred scene, sampled by screen position), the
//! shape fragment (the vector core: an analytic distance field read
//! from the record a vertex points at) and, since K3b, the frosted
//! band of that same core — the one fragment that reads both.

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
    // D0's homography, in PIXELS: the cube's per-face perspective moves
    // here from the CPU loop, eventually. Identity today, and identity
    // is bit-for-bit yesterday's positions: p.z is then exactly 1.0 and
    // x / 1.0 is lossless in IEEE 754.
    xform: mat3x3<f32>,
};

var<push_constant> pc: Push;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    // Which Shape record this vertex belongs to; NO_SHAPE (all ones)
    // outside the shape lane. Flat: a quad must not interpolate
    // between two records.
    @location(2) @interpolate(flat) shape: u32,
};

@vertex
fn vs_main(
    @location(0) a_pos: vec2<f32>,
    @location(1) a_uv: vec2<f32>,
    @location(2) a_color: vec4<f32>,
    @location(3) a_shape: u32,
) -> VsOut {
    var out: VsOut;
    // The divide is MANUAL, on purpose (D0's third finding): handing
    // p.z to the hardware as w would switch interpolation to
    // perspective-correct and silently change the picture the
    // CPU-projected cube draws today. Here uv and color interpolate
    // linearly in screen space, exactly as they always have.
    let p = pc.xform * vec3<f32>(a_pos, 1.0);
    let px = p.xy / p.z;
    let ndc = px / pc.screen * 2.0 - vec2<f32>(1.0, 1.0);
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = a_uv;
    out.color = a_color;
    out.shape = a_shape;
    return out;
}

@group(0) @binding(0) var t_atlas: texture_2d<f32>;
@group(0) @binding(1) var s_atlas: sampler;
@group(1) @binding(0) var t_lut: texture_3d<f32>;
@group(1) @binding(1) var s_lut: sampler;

// One shape record, the mirror of nacelle::draw::Shape (f3 2.5):
// 80 bytes, array stride 80. `half` is a reserved WGSL word, hence
// half_size. The last field arrived with K3b (3.3): a frosted band
// composes three colours in one fragment and only two of them had a
// home — the tint that multiplies the blurred scene had none.
struct Shape {
    half_size: vec2<f32>,
    stroke: f32,
    feather: f32,
    corner: vec4<f32>,
    stroke_c: vec4<f32>,
    flags: u32,
    arc_half: f32,
    arc_dir: f32,
    pad: f32,
    tint: vec4<f32>,
};
@group(2) @binding(0) var<storage, read> shapes: array<Shape>;

// cos 45, the chamfer plane's normalisation. Module scope: naga 0.19
// does not parse `const` inside a function body.
const SQRT1_2: f32 = 0.70710678118654752;

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

// The vector core (f3 2.2/2.3/2.10): every record drawn as its Box
// distance — round and chamfered corners per corner, bed and border
// composed by area in ONE record. `uv` carries the LOCAL position in px
// from the record's centre; the atlas is not sampled here at all.
//
// nacelle::sdf is the specification this implements, function for
// function: d_box, d_round, d_chamfer, coverage, band_coverage, over,
// compose. A change here that is not also a change there is wrong by
// definition — that file is where the mathematics is proved, on the
// CPU, without a device.
//
// The three functions below carry the whole of it, and they are
// functions rather than a body because K3b gave the lane a SECOND entry
// point (fs_shape_glass). Two copies of one field would be two answers
// waiting to differ, which is exactly the defect this project spent a
// milestone removing from its easing resolvers.

// The Box family's distance under four per-corner treatments — the
// mirror of nacelle::sdf::d_shape. No derivatives here: this is a pure
// function of the local point, and that is what lets the caller take
// its screen gradient.
fn shape_distance(s: Shape, p: vec2<f32>) -> f32 {
    let b = s.half_size;

    // The exact box distance.
    let q = abs(p) - b;
    let d_box = min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0)));

    // Which corner rules this fragment: 0 tl, 1 tr, 2 br, 3 bl —
    // ring_points' order, y down. The quadrant seam sits mid-edge,
    // deep inside, where the treatment switch cannot show.
    let ix = select(select(0u, 1u, p.x >= 0.0),
                    select(3u, 2u, p.x >= 0.0),
                    p.y >= 0.0);
    let top = select(s.corner.x, s.corner.y, p.x >= 0.0);
    let bot = select(s.corner.w, s.corner.z, p.x >= 0.0);
    let k = select(top, bot, p.y >= 0.0);
    let st = (s.flags >> (2u * ix)) & 3u;

    // The rounded corner: the exact rounded-box distance at radius k.
    let qk = q + vec2<f32>(k);
    let d_rnd = min(max(qk.x, qk.y), 0.0) + length(max(qk, vec2<f32>(0.0))) - k;

    // The chamfer: the 45-degree half-plane |x| + |y| = b.x + b.y - k,
    // composed with the box. Exact near the boundary, which is the
    // only place coverage reads it.
    let d_cut = (abs(p.x) + abs(p.y) - (b.x + b.y - k)) * SQRT1_2;
    let d_chm = max(d_box, d_cut);

    // Composed by select, not by branch: d must stay a smooth function
    // of position inside every 2x2 derivative block (f3 2.2).
    var d = d_box;
    d = select(d, d_rnd, st == 1u);
    d = select(d, d_chm, st == 2u);
    return d;
}

// Coverage of the silhouette (x) and area of the inward band (y).
// Called from the top level of an entry point and from nowhere else:
// the derivatives below need uniform control flow.
fn shape_cover(s: Shape, d: f32) -> vec2<f32> {
    // AA width from the SCREEN derivatives of the field itself — the
    // one form correct under any transform the vertices rode through.
    // length(vec2(dpdx, dpdy)), never fwidth: fwidth over-reads by
    // sqrt(2) on a 45-degree slope (f3 2.3).
    let g = vec2<f32>(dpdx(d), dpdy(d));
    let w = max(length(g), 1e-6);
    let cov = clamp(0.5 - d / w, 0.0, 1.0);

    // The stroke band, INWARD from the boundary (the project's
    // convention), as an AREA: the interior LESS the interior inset by
    // the stroke. One ramp minus the other, never the product of two —
    // the band's outer boundary IS the silhouette, and a ramp weighted
    // by itself puts 0.25 on an edge whose area is 0.5, and keeps
    // reading a half at the centre of a hairline however thin it gets.
    // The mirror of `nacelle::sdf::band_coverage`.
    let has_stroke = f32((s.flags >> 13u) & 1u);
    let cov_in = clamp(0.5 - (d + s.stroke) / w, 0.0, 1.0);
    let a_band = max(cov - cov_in, 0.0) * has_stroke;
    return vec2<f32>(cov, a_band);
}

// `top` over `bottom`, straight alpha in and out — the mirror of
// nacelle::sdf::over, and the identity that lets a wash drawn as its
// own quad become a term in somebody else's fragment (3.3).
fn over(top: vec4<f32>, bottom: vec4<f32>) -> vec4<f32> {
    let a = top.a + bottom.a * (1.0 - top.a);
    let premul = top.rgb * top.a + bottom.rgb * bottom.a * (1.0 - top.a);
    // Nothing laid at all: the colour is unobservable, and the top's is
    // as good a nothing as any.
    return vec4<f32>(select(top.rgb, premul / max(a, 1e-5), a > 0.0), a);
}

// 2.10's one composition, by area: the band is the stroke OVER the
// bed — `ring_fill` draws on the original rect and the border stands
// on it — the rest of the covered pixel is the bed alone, and the
// two are averaged by the areas they hold. Bed and edge share this
// record because they share a silhouette, so that silhouette blends
// exactly once. Straight alpha out, as every path here returns.
fn shape_compose(fill: vec4<f32>, stroke_c: vec4<f32>, cov: f32, a_band: f32) -> vec4<f32> {
    let s_a = a_band * stroke_c.a;
    let f_a = fill.a;
    let alpha = cov * f_a + s_a * (1.0 - f_a);
    let premul = s_a * stroke_c.rgb + f_a * (cov - s_a) * fill.rgb;
    // No band under this fragment: the bed's own colour, untouched. The
    // divide would return it to within an ulp; this returns it exactly.
    let rgb = select(fill.rgb, premul / max(alpha, 1e-5), s_a > 0.0);
    return vec4<f32>(rgb, alpha);
}

@fragment
fn fs_shape(
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) shape: u32,
) -> @location(0) vec4<f32> {
    // Overflowing shapes are clipped by the uploader (the MAX_SHAPES
    // idiom); a vertex past the clip must read SOME record rather than
    // out of bounds.
    let s = shapes[min(shape, arrayLength(&shapes) - 1u)];
    let p = uv;
    let d = shape_distance(s, p);
    let cb = shape_cover(s, d);
    let has_fill = f32((s.flags >> 12u) & 1u);
    let f_a = color.a * has_fill;
    return grade(shape_compose(vec4<f32>(color.rgb, f_a), s.stroke_c, cb.x, cb.y));
}

// The FROSTED band of the vector lane (f3 3.3, K3b): the same record,
// the same silhouette, the same composition — with the blurred scene
// under it.
//
// Three layers reach one fragment here, and that is the whole point of
// the entry point existing. Drawn as three quads they would blend the
// SAME outline three times, and on a half-covered pixel each pass adds
// alpha the surface does not have — `c·(1 − c)·a·b` of it, which is R4
// under another name and reads as a heavy rim on exactly the arcs the
// eye goes to. Here the frost, the wash and the border are folded
// first and `cov` is applied once.
//
// The frost is sampled by SCREEN position, like `fs_blur` and for the
// same reason: the blurred copy belongs to the picture, not to the
// quad, so a surface may ride any animation without the frost sliding
// under it. The rank it samples is the RUN's — the renderer binds one
// pyramid target per run, which is why the toolkit has three
// SHAPE_GLASS handles and the record has no rank field.
//
// AND THE GRADE GOES ON THE LAYERS, WHICH IS THE ONE LINE HERE THAT
// LOOKS LIKE A STYLE CHOICE AND IS NOT. A frosted surface is drawn in
// two pieces: this band, and a CORE of two ordinary quads — the frost
// through fs_blur, the wash through fs_main — that the hardware
// blends. Each of those grades its own colour on the way out, so the
// core shows grade(wash) over grade(frost). OVER is associative, so
// this fragment reproduces it exactly by folding the two GRADED
// layers; folding first and grading the result is a different colour
// wherever the LUT is not affine, and the difference draws as a
// rectangle inside the panel, on the line where the core's cut falls.
// `fs_shape` above may grade its composite and does: its fill is one
// layer, and there is nothing in it for a fold to reassociate.
// nacelle::sdf::glass_base carries the identity and the proof; the
// LUT is off by default, so nothing here would have said otherwise.
@fragment
fn fs_shape_glass(
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) shape: u32,
) -> @location(0) vec4<f32> {
    let s = shapes[min(shape, arrayLength(&shapes) - 1u)];
    let p = uv;
    let d = shape_distance(s, p);
    let cb = shape_cover(s, d);
    // The blurred scene times the record's tint: the same product the
    // surface's own core draws through fs_blur, so core and band frost
    // alike. The tint can only darken; the wash over it is the only
    // layer that can brighten (the master's ladder at elev.*.glass).
    let frost = grade(textureSample(t_atlas, s_atlas, pos.xy / pc.screen) * s.tint);
    let has_fill = f32((s.flags >> 12u) & 1u);
    let wash = grade(vec4<f32>(color.rgb, color.a * has_fill));
    return shape_compose(over(wash, frost), grade(s.stroke_c), cb.x, cb.y);
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

#[cfg(test)]
mod tests {
    /// R2's smoke test and f3 K1's stated acceptance condition: naga
    /// 0.19 must carry `var<storage, read>` with a runtime-sized array
    /// AND the screen-space derivatives through wgsl-in -> spv-out —
    /// neither had ever crossed this tree before the shape lane.
    /// compile() panics on any of its three stages, so calling it is
    /// the assertion; the source checks keep a refactor from hollowing
    /// the test out silently. Pure CPU, no GPU, no device.
    #[test]
    fn naga_carries_the_storage_array_and_the_derivatives() {
        assert!(super::WGSL_SRC.contains("var<storage, read>"));
        assert!(super::WGSL_SRC.contains("dpdx"));
        assert!(super::WGSL_SRC.contains("mat3x3<f32>"));
        let spv = super::compile();
        assert!(!spv.is_empty());
    }

    /// The two retired forms, kept out by name.
    ///
    /// `nacelle::sdf` proves the arithmetic on the CPU; nothing here can
    /// run WGSL, so what this side can guarantee is only that the shader
    /// has not drifted BACK — that the band is still a difference of two
    /// coverage ramps rather than the folded field times the silhouette
    /// (which put 0.25 on an edge covering 0.5), and that the AA width
    /// is still the gradient's length rather than `fwidth` (which
    /// over-reads √2 on a 45° slope). Both are one-line regressions that
    /// compile, validate and look almost right, which is exactly the
    /// class of change a test has to hold.
    ///
    /// Two anchors moved with K3b and only because the code did: the
    /// band and the composition now live in `shape_cover` and
    /// `shape_compose`, which the frosted entry point calls as well, so
    /// `s.stroke_c` reads `stroke_c` and `s.flags` reads `s.flags`
    /// through the record the caller passed. The forms they hold are
    /// the same forms, character for character otherwise.
    #[test]
    fn the_shape_fragment_keeps_the_reference_s_form() {
        let src = super::WGSL_SRC;
        // The retired forms are NAMED in the comments that retired them,
        // so the negative assertions read the code alone.
        let code: String = src
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!code.contains("fwidth"), "the AA width went back to fwidth");
        assert!(
            !code.contains("max(d, -d - s.stroke)"),
            "the folded band came back — see nacelle::sdf::band_coverage"
        );
        // The difference of ramps, and the area composition it feeds.
        assert!(src.contains("let cov_in = clamp(0.5 - (d + s.stroke) / w, 0.0, 1.0);"));
        assert!(src.contains("let a_band = max(cov - cov_in, 0.0) * has_stroke;"));
        assert!(src.contains("let alpha = cov * f_a + s_a * (1.0 - f_a);"));
        assert!(src.contains("let g = vec2<f32>(dpdx(d), dpdy(d));"));
        // The two gates the composition stands on. `has_fill` is the
        // one that reads like noise and is not: a STROKE-only record
        // carries the BAND's colour on its vertices (draw.rs, "the
        // stroke's when there is no fill"), so dropping the factor
        // fills every borderless ring solid in its own border colour —
        // a focus ring becomes a plate. It compiles, it validates, and
        // nothing else in this file would notice.
        assert!(
            src.contains("let f_a = color.a * has_fill;"),
            "the fill gate went missing: a stroke-only record would fill solid"
        );
        assert!(src.contains("let s_a = a_band * stroke_c.a;"));
        assert!(src.contains("let has_fill = f32((s.flags >> 12u) & 1u);"));
        assert!(src.contains("let has_stroke = f32((s.flags >> 13u) & 1u);"));
        // …and ONE copy of each, which is what the functions are for:
        // two entry points reading two fields of the same mathematics
        // is how a lane grows two answers (the easing resolvers, three
        // of them, two with the same bug — motion.rs).
        assert_eq!(src.matches("fn shape_distance(").count(), 1);
        assert_eq!(src.matches("dpdx(d)").count(), 1);
        assert_eq!(src.matches("let alpha = cov * f_a + s_a * (1.0 - f_a);").count(), 1);
    }

    /// The FROSTED entry point's own contract (f3 §3.3), and every line
    /// of it is a thing that would compile while drawing the wrong
    /// picture.
    ///
    /// * It samples by SCREEN position — the blurred copy belongs to
    ///   the picture, not to the quad, so a panel riding an animation
    ///   must not carry its frost along. `fs_blur` has said this since
    ///   glass existed; this is the same sentence in the second place
    ///   it now has to be true.
    /// * The frost MULTIPLIES the sample by the record's tint and the
    ///   wash lies OVER the result. The other order is a different
    ///   picture and an equally short line: the master's ladder says
    ///   the tint can only darken and the wash is the only knob that
    ///   brightens, which is precisely the asymmetry of these two
    ///   operators.
    /// * The three layers reach ONE `shape_compose` call. Two calls
    ///   would be R4 rebuilt by hand.
    /// * The grade is applied to each LAYER and never to their fold.
    ///   The core of the same surface is two ordinary draws the
    ///   hardware blends after each has graded itself, and OVER is
    ///   associative — so folding graded layers reproduces the core
    ///   exactly and grading the fold does not. That one is worth a
    ///   test of its own because it is INVISIBLE by default: `grade()`
    ///   is the identity until a user loads a colour LUT, and the day
    ///   they do, a rectangle appears inside every frosted panel on the
    ///   line where the core's cut falls. `nacelle::sdf` states the
    ///   identity and rasterises both variants to prove it; this side
    ///   can only hold the shape of the expression, which is exactly
    ///   the thing a refactor would tidy away.
    #[test]
    fn the_frosted_fragment_samples_the_screen_and_lays_the_wash_over_the_tint() {
        let src = super::WGSL_SRC;
        let glass = src
            .split("fn fs_shape_glass(")
            .nth(1)
            .expect("fs_shape_glass went missing");
        assert!(glass.contains("textureSample(t_atlas, s_atlas, pos.xy / pc.screen)"));
        assert!(glass.contains("* s.tint"), "the tint stopped multiplying the sample");
        assert!(
            glass.contains("over(wash, frost)"),
            "the wash went under the frost, or the fold went away"
        );
        assert!(glass.contains("let p = uv;"), "the field lost its local point");
        assert_eq!(glass.matches("shape_compose(").count(), 1, "two compositions");
        // Three grades, one per layer, and NONE around the composite.
        assert_eq!(glass.matches("grade(").count(), 3, "a layer lost its grade");
        assert!(
            !glass.contains("grade(shape_compose("),
            "the grade moved onto the fold: the band no longer matches its own core"
        );
        assert!(glass.contains("let frost = grade("), "the frost is graded where it is made");
        assert!(glass.contains("let wash = grade("), "the wash is graded where it is made");
        assert!(glass.contains("grade(s.stroke_c)"), "the border stopped being graded at all");
        // The record's fourth colour, in both languages.
        assert!(src.contains("tint: vec4<f32>,"));
        // And OVER is the identity nacelle::sdf::over states, not a mix.
        assert!(src.contains("let a = top.a + bottom.a * (1.0 - top.a);"));
        assert!(
            src.contains("let premul = top.rgb * top.a + bottom.rgb * bottom.a * (1.0 - top.a);")
        );
    }

    /// **Why f3 K4 cost this crate nothing.** The diagonal lane — every
    /// tick, chevron, sort arrow, chart stroke and joint disc — is a
    /// Box record read along its own axes, and the only thing that
    /// makes it one is where the CPU puts four vertices and what it
    /// writes in their `uv`. That works because of two properties of
    /// the fragment below, and both are one careless line from being
    /// lost:
    ///
    /// * **The local point comes from `uv`** — an interpolated varying,
    ///   which the rasteriser has already run backwards through
    ///   whatever affine map the vertices described. Reading
    ///   `@builtin(position)` instead would compile, would look
    ///   identical on every axis-aligned shape K3 draws, and would draw
    ///   every diagonal in the wrong frame.
    /// * **The AA width comes from the field's own screen
    ///   derivatives** — so a rotation (|∇d| = 1) and a shear (|∇d|
    ///   whatever it is) both come out right without the shader being
    ///   told which it got. A constant width, or one read off a push
    ///   constant, would be correct only on the screen's own axes.
    ///
    /// The first is asserted here, the second by the test above. Neither
    /// can be checked by running WGSL in this tree; both can be checked
    /// by reading it, which is what a contract between two crates is
    /// for.
    ///
    /// The slice is cut at the NEXT entry point on purpose. K3b gave
    /// the lane a second one, and it reads `@builtin(position)` because
    /// its frost belongs to the screen — a slice running to the end of
    /// the file would find that word and call it a regression in the
    /// wrong fragment.
    #[test]
    fn the_shape_fragment_reads_its_local_point_and_not_the_pixel_s() {
        let shape = super::WGSL_SRC
            .split("fn fs_shape(")
            .nth(1)
            .expect("fs_shape went missing")
            .split("@fragment")
            .next()
            .expect("fs_shape ran to the end of the file");
        let code: String = shape
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(code.contains("let p = uv;"), "the local point stopped being the varying");
        assert!(
            !code.contains("builtin(position)") && !code.contains("frag_coord"),
            "the shape fragment started asking where the pixel is"
        );
        // …and the width still comes from the field — through
        // `shape_cover`, which both entry points call and which is
        // where `dpdx` now lives, so an oriented frame still needs no
        // push constant.
        let cover = super::WGSL_SRC
            .split("fn shape_cover(")
            .nth(1)
            .expect("shape_cover went missing");
        assert!(cover.contains("dpdx(d)") && cover.contains("dpdy(d)"));
        assert!(code.contains("shape_cover(s, d)"));
    }
}

#[cfg(test)]
mod layout_tests {
    /// **The two Shapes are one Shape.** The record is written by the
    /// toolkit in Rust and read by this module in WGSL, and nothing but
    /// agreement between two hand-written declarations keeps a
    /// fragment from reading a corner where a colour is. Two crews are
    /// editing the vector lane on two branches; this asks naga for the
    /// std430 size of the struct it just parsed and compares it with
    /// the Rust one, which carries its own `size_of` assertion.
    ///
    /// A size match is not a field-order match, and this cannot claim
    /// to be one. It is the check that catches the change that actually
    /// happens — a field added on one side and not the other — at the
    /// moment it happens rather than at the first frosted panel.
    #[test]
    fn the_record_is_the_same_size_in_both_languages() {
        let module = naga::front::wgsl::parse_str(super::WGSL_SRC).unwrap();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let (handle, _) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("Shape"))
            .expect("the WGSL module declares a Shape");
        assert_eq!(
            layouter[handle].size as usize,
            std::mem::size_of::<nacelle::draw::Shape>(),
            "the record grew on one side of the seam only"
        );
    }
}
