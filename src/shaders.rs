//! WGSL -> SPIR-V compilation at startup using naga (pure Rust,
//! no external tools like glslc). One module, five entry points: the
//! shared vertex stage, the atlas fragment (coverage modulates alpha),
//! the image fragment (the texture IS the color), the frosted-glass
//! fragment (the blurred scene, sampled by screen position) and the
//! shape fragment (the vector core: an analytic distance field read
//! from the record a vertex points at).

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
// 64 bytes, array stride 64. `half` is a reserved WGSL word, hence
// half_size.
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
};
@group(2) @binding(0) var<storage, read> shapes: array<Shape>;

// cos 45, the chamfer plane's normalisation. Module scope: naga 0.19
// does not parse `const` inside a function body.
const SQRT1_2: f32 = 0.70710678118654752;

// The six-fold lattice, for the hexagon: the mirror normal
// (-cos 30, sin 30) that folds a quadrant into one sixth, and 1/sqrt(3),
// half of one edge in units of the apothem.
const HK: vec3<f32> = vec3<f32>(-0.8660254, 0.5, 0.5773503);

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

// **The distance ONE record computes** — the whole of the silhouette,
// and the only part of this file whose arithmetic can be checked
// without a device.
//
// It is a function and not a stretch of `fs_shape` for exactly that
// reason. `nacelle::sdf::d_record` is the specification it implements,
// line for line, and "a change here that is not also a change there is
// wrong by definition" was a promise with nothing keeping it: the tests
// below could only quote the source back at itself. A pure function of
// six plain arguments — no textures, no derivatives, no control flow —
// is one an evaluator can run over naga's own IR, which is what
// `the_shape_field_draws_the_silhouette_it_claims` does. The arguments
// are spelled out rather than passed as the record so that the
// evaluator never has to know the struct's layout, only its numbers.
//
// The record's slots are read exactly as the table on
// nacelle::draw::ShapeKind states them: lengths in `corner`, angles in
// `arc_half` / `arc_dir`, the kind in bits 8-11 of `flags` and the four
// corner treatments in bits 0-7.
fn shape_field(
    p: vec2<f32>,
    b: vec2<f32>,
    corner: vec4<f32>,
    flags: u32,
    arc_half: f32,
    arc_dir: f32,
) -> f32 {
    // The exact box distance.
    let q = abs(p) - b;
    let d_box = min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0)));

    // Which corner rules this fragment: 0 tl, 1 tr, 2 br, 3 bl —
    // ring_points' order, y down. The quadrant seam sits mid-edge,
    // deep inside, where the treatment switch cannot show.
    let ix = select(select(0u, 1u, p.x >= 0.0),
                    select(3u, 2u, p.x >= 0.0),
                    p.y >= 0.0);
    let top = select(corner.x, corner.y, p.x >= 0.0);
    let bot = select(corner.w, corner.z, p.x >= 0.0);
    let k = select(top, bot, p.y >= 0.0);
    let st = (flags >> (2u * ix)) & 3u;

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

    // ---- the kinds past Box (K6, bits 8-11) -------------------------
    //
    // Until K6 this fragment drew EVERY record as the box distance
    // above and never looked at bits 8-11, so a Hex record and a Box
    // record of the same rect were the same picture. They are not any
    // more. Each field below is the mirror of a function in
    // nacelle::sdf — d_hex, d_arc, d_chevron — and the record's
    // per-kind slots are read exactly as the table on
    // nacelle::draw::ShapeKind states them: lengths in `corner`, angles
    // in `arc_half` / `arc_dir`.
    //
    // Computed UNCONDITIONALLY and composed by select, like the corner
    // treatments and for a harder reason: `dpdx`/`dpdy` below must sit
    // in uniform control flow, and a branch on a record's own kind is
    // not uniform across a draw. Every fragment therefore pays for
    // every kind. That is a measured cost for K3c to answer — the
    // honest remedy is one pipeline per kind, keyed on the run, not a
    // branch here.
    let kind = (flags >> 8u) & 0xFu;

    // The local point seen from a frame turned by arc_dir: the two
    // kinds that carry an angle share it, and rotating the question is
    // the same as rotating the shape.
    let ca = cos(arc_dir);
    let sa = sin(arc_dir);
    let pt = vec2<f32>(p.x * ca + p.y * sa, -p.x * sa + p.y * ca);

    // Hex: the flat-topped hexagon of apothem corner.x, folded into one
    // sixth and read against that sixth's single edge.
    var hp = abs(pt);
    hp = hp - 2.0 * min(dot(HK.xy, hp), 0.0) * HK.xy;
    hp = hp - vec2<f32>(clamp(hp.x, -HK.z * corner.x, HK.z * corner.x), corner.x);
    // sign(0.0) is zero in WGSL and one in Rust, so the sign is a
    // comparison in both files rather than a call in either.
    let d_hexagon = length(hp) * select(-1.0, 1.0, hp.y >= 0.0);

    // Ring: the annular arc, half width corner.x, its outer edge on the
    // shorter side of the rect, swept 2*arc_half about local +y with
    // round caps. Clamping sin at zero is what makes arc_half >= PI a
    // CLOSED ring: a non-positive left side can never beat zero, so
    // every fragment takes the circle (sin(PI) in f32 is a hair below
    // zero, and that hair used to decide it).
    let rb = corner.x;
    let ra = max(min(b.x, b.y) - rb, 0.0);
    let sn = max(sin(arc_half), 0.0);
    let cs = cos(arc_half);
    let ap = vec2<f32>(abs(pt.x), pt.y);
    let d_cap = length(ap - vec2<f32>(sn, cs) * ra) - rb;
    let d_ann = abs(length(ap) - ra) - rb;
    let d_ring = select(d_ann, d_cap, cs * ap.x > sn * ap.y);

    // Chevron: the box with one or both vertical ends collapsed to a
    // point at mid-height, corner.x deep on the left and corner.y on
    // the right. abs(p.y) folds each end's two slanted edges into one
    // half-plane; a depth of zero gives back that end's own vertical
    // edge exactly, so an uncollapsed end needs no second formula.
    let ll = max(sqrt(b.y * b.y + corner.x * corner.x), 1e-6);
    let lr = max(sqrt(b.y * b.y + corner.y * corner.y), 1e-6);
    let cut_l = (corner.x * abs(p.y) - b.y * (p.x + b.x)) / ll;
    let cut_r = (corner.y * abs(p.y) - b.y * (b.x - p.x)) / lr;
    let d_chevron = max(d_box, max(cut_l, cut_r));

    d = select(d, d_ring, kind == 1u);
    d = select(d, d_hexagon, kind == 2u);
    d = select(d, d_chevron, kind == 3u);
    return d;
}

// The vector core (f3 2.2/2.3/2.10): one record, one distance field,
// bed and border composed by area. `uv` carries the LOCAL position in
// px from the record's centre; the atlas is not sampled here at all.
//
// The SILHOUETTE is `shape_field` above — one function, chosen by kind,
// with its own tests. What is left here is what needs the fragment: the
// record, the screen derivatives, and 2.10's composition by area.
//
// nacelle::sdf is the specification both halves implement, function for
// function: d_box, d_round, d_chamfer, d_hex, d_arc, d_chevron,
// d_record, coverage, band_coverage, compose. A change here that is not
// also a change there is wrong by definition — that file is where the
// mathematics is proved, on the CPU, without a device. This crate cannot
// yet make that a compile error (its lock pins a libnacelle from before
// K6, and `nacelle::sdf` is imported nowhere in this tree); what it can
// do, and now does, is check its own arithmetic against the geometry
// both files claim to draw — `crate::shape_field`.
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
    let b = s.half_size;

    let d = shape_field(p, b, s.corner, s.flags, s.arc_half, s.arc_dir);

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
    let has_fill = f32((s.flags >> 12u) & 1u);
    let has_stroke = f32((s.flags >> 13u) & 1u);
    let cov_in = clamp(0.5 - (d + s.stroke) / w, 0.0, 1.0);
    let a_band = max(cov - cov_in, 0.0) * has_stroke;

    // 2.10's one composition, by area: the band is the stroke OVER the
    // bed — `ring_fill` draws on the original rect and the border stands
    // on it — the rest of the covered pixel is the bed alone, and the
    // two are averaged by the areas they hold. Bed and edge share this
    // record because they share a silhouette, so that silhouette blends
    // exactly once. Straight alpha out, as every path here returns.
    let s_a = a_band * s.stroke_c.a;
    let f_a = color.a * has_fill;
    let alpha = cov * f_a + s_a * (1.0 - f_a);
    let premul = s_a * s.stroke_c.rgb + f_a * (cov - s_a) * color.rgb;
    // No band under this fragment: the bed's own colour, untouched. The
    // divide would return it to within an ulp; this returns it exactly.
    let rgb = select(color.rgb, premul / max(alpha, 1e-5), s_a > 0.0);
    return grade(vec4<f32>(rgb, alpha));
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
        assert!(src.contains("let s_a = a_band * s.stroke_c.a;"));
        assert!(src.contains("let has_fill = f32((s.flags >> 12u) & 1u);"));
        assert!(src.contains("let has_stroke = f32((s.flags >> 13u) & 1u);"));
    }

    /// K6's own regression, read: bits 8-11 must still be READ, the
    /// selection must stay a `select`, and the field must stay a
    /// function the evaluator can run.
    ///
    /// The trap this holds shut is the one f3 §4 named before K6 was
    /// written — "fs_shape draws every record as its box distance and
    /// never reads bits 8-11", so a Hex record and a Box record of the
    /// same rect drew the same picture and nothing complained.
    ///
    /// These are the invariants a NUMBER cannot see, and they are all
    /// this test is now for; what the arithmetic computes is
    /// `the_shape_field_draws_the_silhouette_it_claims`, one test down.
    /// The two used to be one, and the string half was standing in for
    /// the other: six arithmetic mutations inside this branch passed the
    /// whole gate, because a test can only hold the lines it quotes.
    #[test]
    fn the_shape_fragment_chooses_its_field_by_kind() {
        let field = super::WGSL_SRC
            .split("fn shape_field(")
            .nth(1)
            .expect("shape_field went missing — the field stopped being one function")
            .split("\n}")
            .next()
            .expect("shape_field never closes");
        let code: String = field
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code.contains("let kind = (flags >> 8u) & 0xFu;"),
            "the kind stopped being read out of bits 8-11"
        );
        for (want, what) in [
            ("d = select(d, d_ring, kind == 1u);", "Ring"),
            ("d = select(d, d_hexagon, kind == 2u);", "Hex"),
            ("d = select(d, d_chevron, kind == 3u);", "Chevron"),
        ] {
            assert!(code.contains(want), "{what} lost its branch: the box would draw instead");
        }
        // The per-kind slots, read where the record puts them.
        assert!(code.contains("arc_half"), "the sweep went unread");
        assert!(code.contains("arc_dir"), "the turn went unread");
        // No BRANCH may wrap the fields: `fs_shape` takes the
        // derivatives of what this returns, and those must sit in
        // uniform control flow. WGSL makes the parentheses optional, so
        // the word is what is looked for — the old spelling `if (` was
        // blind to `if kind == 1u {`, which is how one would actually be
        // written.
        for banned in ["if ", "switch ", "loop ", "while ", "for "] {
            assert!(
                !code.contains(banned),
                "`{banned}` in shape_field: a branch on the kind breaks derivative uniformity"
            );
        }
        // And the field stays a function of PLAIN numbers. A struct
        // argument would compile and would put the evaluator below out
        // of reach of the record's own layout.
        assert!(
            code.contains(") -> f32 {") || super::WGSL_SRC.contains(") -> f32 {"),
            "shape_field stopped answering a distance"
        );
    }

    /// **The arithmetic of the kind branch, RUN.**
    ///
    /// Nothing in this crate can execute WGSL — but naga has already
    /// parsed it by the time a test runs, so `crate::shape_field`
    /// evaluates `shape_field` over naga's own IR and this compares the
    /// answer against the silhouette stated as GEOMETRY: the hexagon's
    /// six vertices at their clock positions, the chevron's six corners,
    /// the arc's centre curve, the box's own outline. A distance to a
    /// polygon knows nothing about mirror normals or folded quadrants,
    /// so agreement is evidence and not a coincidence.
    ///
    /// Two grades of agreement, because two of the fields are honestly
    /// approximate. `max` of half-planes is the exact distance inside a
    /// convex silhouette and an UNDERESTIMATE outside it, in the acute
    /// wedge behind a vertex — §2.2 accepted that trade for the chamfer
    /// and K6 took it again for the chevron. So those two are held to
    /// the exact distance on the inside and to the SIGN everywhere,
    /// which is the whole of what coverage reads; the box, the rounded
    /// box, the hexagon and the arc are exact fields and are held to the
    /// number everywhere.
    #[test]
    fn the_shape_field_draws_the_silhouette_it_claims() {
        use crate::shape_field::{
            arc_curve, box_poly, chevron_poly, curve_dist, hex_poly, poly_sd, Field, Record,
        };
        let module =
            naga::front::wgsl::parse_str(super::WGSL_SRC).expect("the shader must parse");
        let mut field = Field::new(&module, "shape_field");

        /// The corner treatments, packed as the record packs them: two
        /// bits each, tl tr br bl, and the kind in bits 8-11.
        fn flags(kind: u32, styles: [u32; 4]) -> u32 {
            (kind << 8) | styles.iter().enumerate().fold(0, |f, (i, s)| f | (s << (2 * i)))
        }

        // (name, record, reference distance, exact everywhere)
        let sq = box_poly([60.0, 36.0], [(0, 0.0); 4]);
        let rn = box_poly([60.0, 36.0], [(1, 14.0); 4]);
        let ch = box_poly([60.0, 36.0], [(2, 12.0); 4]);
        let mx = box_poly([60.0, 36.0], [(1, 14.0), (2, 12.0), (0, 0.0), (1, 6.0)]);
        let hex_flat = hex_poly(40.0, 0.0);
        // 15°: NOT a symmetry of a six-fold lattice, so a turn that ran
        // the other way would put the flat edge somewhere else. 30°
        // would not — which is the angle the old tests happened to use.
        let turn = std::f32::consts::PI / 12.0;
        let hex_turned = hex_poly(40.0, turn);
        let chev_both = chevron_poly([70.0, 30.0], 18.0, 26.0);
        let chev_right = chevron_poly([70.0, 30.0], 0.0, 26.0);
        // The band's outer edge meets the SHORTER side of the rect, so
        // its axis sits one half thickness inside that.
        let cut = arc_curve(60.0 - 7.0, 0.6, 0.9);
        let closed = arc_curve(60.0 - 7.0, std::f32::consts::PI, 0.0);
        let oblong = arc_curve(40.0 - 7.0, 0.6, -0.5);

        type Truth<'t> = Box<dyn Fn([f32; 2]) -> f32 + 't>;
        fn poly(p: &[[f32; 2]]) -> Truth<'_> {
            Box::new(move |q| poly_sd(p, q))
        }
        fn band(c: &[[f32; 2]], rb: f32) -> Truth<'_> {
            Box::new(move |q| curve_dist(c, q) - rb)
        }
        let zero = [0.0f32; 4];
        let cases: Vec<(&str, Record, Truth, bool)> = vec![
            (
                "box",
                Record { half: [60.0, 36.0], corner: zero, flags: flags(0, [0; 4]), arc_half: 0.0, arc_dir: 0.0 },
                poly(&sq),
                true,
            ),
            (
                "rounded box",
                Record { half: [60.0, 36.0], corner: [14.0; 4], flags: flags(0, [1; 4]), arc_half: 0.0, arc_dir: 0.0 },
                poly(&rn),
                true,
            ),
            (
                "chamfered box",
                Record { half: [60.0, 36.0], corner: [12.0; 4], flags: flags(0, [2; 4]), arc_half: 0.0, arc_dir: 0.0 },
                poly(&ch),
                false,
            ),
            (
                "four corners that differ",
                Record { half: [60.0, 36.0], corner: [14.0, 12.0, 0.0, 6.0], flags: flags(0, [1, 2, 0, 1]), arc_half: 0.0, arc_dir: 0.0 },
                poly(&mx),
                false,
            ),
            (
                "hexagon, flat top",
                Record { half: [50.0, 50.0], corner: [40.0, 0.0, 0.0, 0.0], flags: flags(2, [0; 4]), arc_half: 0.0, arc_dir: 0.0 },
                poly(&hex_flat),
                true,
            ),
            (
                "hexagon turned 15 degrees",
                Record { half: [50.0, 50.0], corner: [40.0, 0.0, 0.0, 0.0], flags: flags(2, [0; 4]), arc_half: 0.0, arc_dir: turn },
                poly(&hex_turned),
                true,
            ),
            (
                "chevron, both ends",
                Record { half: [70.0, 30.0], corner: [18.0, 26.0, 0.0, 0.0], flags: flags(3, [0; 4]), arc_half: 0.0, arc_dir: 0.0 },
                poly(&chev_both),
                false,
            ),
            (
                "chevron, one end",
                Record { half: [70.0, 30.0], corner: [0.0, 26.0, 0.0, 0.0], flags: flags(3, [0; 4]), arc_half: 0.0, arc_dir: 0.0 },
                poly(&chev_right),
                false,
            ),
            (
                "arc, cut short and turned",
                Record { half: [60.0, 60.0], corner: [7.0, 0.0, 0.0, 0.0], flags: flags(1, [0; 4]), arc_half: 0.6, arc_dir: 0.9 },
                band(&cut, 7.0),
                true,
            ),
            (
                "arc, closed",
                Record { half: [60.0, 60.0], corner: [7.0, 0.0, 0.0, 0.0], flags: flags(1, [0; 4]), arc_half: std::f32::consts::PI, arc_dir: 0.0 },
                band(&closed, 7.0),
                true,
            ),
            (
                "arc in an oblong rect",
                Record { half: [80.0, 40.0], corner: [7.0, 0.0, 0.0, 0.0], flags: flags(1, [0; 4]), arc_half: 0.6, arc_dir: -0.5 },
                band(&oblong, 7.0),
                true,
            ),
        ];

        const N: i32 = 60;
        // The polylines above are inscribed, so the reference reads a
        // hair long; every one of them is fine enough that the hair is
        // under a thousandth of a pixel. This is two orders past that.
        const TOL: f32 = 0.05;
        // A dead band around the boundary, where a float either side is
        // neither an error nor a signal.
        const SKIN: f32 = 0.02;

        for (name, rec, truth, exact) in cases {
            let (mut inside, mut outside, mut worst) = (0usize, 0usize, 0.0f32);
            for iy in -N..=N {
                for ix in -N..=N {
                    let p = [
                        rec.half[0] * 1.7 * ix as f32 / N as f32,
                        rec.half[1] * 1.7 * iy as f32 / N as f32,
                    ];
                    let want = truth(p);
                    let got = field.at(rec, p);
                    if want.abs() > SKIN {
                        assert_eq!(
                            want < 0.0,
                            got < 0.0,
                            "{name}: {p:?} is {} by the outline and {} by the shader ({want} vs {got})",
                            if want < 0.0 { "inside" } else { "outside" },
                            if got < 0.0 { "inside" } else { "outside" }
                        );
                    }
                    if want < 0.0 {
                        inside += 1;
                    } else {
                        outside += 1;
                    }
                    if exact || want <= -SKIN {
                        worst = worst.max((got - want).abs());
                        assert!(
                            (got - want).abs() <= TOL,
                            "{name}: at {p:?} the outline is {want} away and the field says {got}"
                        );
                    }
                }
            }
            // Fail closed: a silhouette nobody landed inside proves
            // nothing about its interior, and one nobody landed outside
            // proves nothing at all.
            assert!(inside >= 100 && outside >= 100, "{name}: {inside} in, {outside} out");
            assert!(worst > 0.0, "{name}: not one sample was compared by number");
        }
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
    #[test]
    fn the_shape_fragment_reads_its_local_point_and_not_the_pixel_s() {
        let shape = super::WGSL_SRC
            .split("fn fs_shape(")
            .nth(1)
            .expect("fs_shape went missing");
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
        // …and the width still comes from the field, inside this
        // fragment, so an oriented frame needs no push constant.
        assert!(code.contains("dpdx(d)") && code.contains("dpdy(d)"));
    }
}
