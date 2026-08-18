//! **What `shape_field` actually computes, run without a GPU.**
//!
//! The header of `fs_shape` promises that `nacelle::sdf` is the
//! specification the WGSL implements and that "a change here that is not
//! also a change there is wrong by definition". Until this module the
//! promise had nothing keeping it: the only tests on the shape fragment
//! asserted that certain LINES OF TEXT were present, so every line they
//! did not quote — the hexagon's mirror normal, the sense of the
//! rotation, the ring's axis radius, the chevron's second end, the sign
//! of the hexagon's field — could be changed arbitrarily and the whole
//! renderer gate still passed.
//!
//! What is missing on this side is not a proof; it is an INTERPRETER.
//! The crate already parses its own WGSL through naga, so the shader's
//! arithmetic is sitting in an expression arena at test time and the
//! only thing needed is something to walk it. That is [`Field`]: a few
//! hundred lines that evaluate the subset of WGSL `shape_field` is
//! written in — literals, constants, vectors, `select`, and the dozen
//! `MathFunction`s it calls. Naga does the parsing, so nothing here can
//! mis-read an expression that naga read correctly.
//!
//! And what it is compared AGAINST is geometry, not a second copy of the
//! same formulas. Every case below states its silhouette as an outline —
//! six hexagon vertices at their clock positions, the chevron's own six
//! corners, the arc's centre curve — and the reference distance is the
//! plain "how far to the nearest point of that outline, and which side
//! of it are we on". A folding trick and a polygon cannot agree by
//! accident.
//!
//! **What this is, and what it is not.** On its own it is not a
//! cross-check against `nacelle::sdf` — it is the shader against
//! GEOMETRY, which is the stronger half of the two, because an outline
//! and a folding trick cannot agree by accident.
//!
//! The other half now exists as well. When this module was written the
//! lock pinned a libnacelle from before K6, which had no `d_record` to
//! call; the merge order has since run (libnacelle, then the pin, then
//! this crate) and `the_field_answers_what_the_reference_answers` takes
//! the check this paragraph said was worth taking — `Field::at` on one
//! side, `nacelle::sdf::d_record` on the other, the same records at the
//! same points. The two answer identically, to the bit.
//!
//! Neither test replaces the other. The reference could be wrong in the
//! same way as the shader and only the geometry would notice; the
//! geometry is stated per case and only the reference covers the whole
//! of `d_record`'s own dispatch, including the kind CODES.
//!
//! Test-only: nothing in the shipping renderer walks the IR.

use naga::{
    BinaryOperator as Bin, Expression as E, Handle, MathFunction as M, Module,
    ScalarKind, Statement as St, SwizzleComponent as Sw, TypeInner, UnaryOperator as Un,
};

// ---------------------------------------------------------------- values

/// A value the evaluator can hold. Vectors carry their length so one
/// arithmetic path serves scalars and vectors alike; WGSL's own rules
/// for mixing them are broadcast rules, which is what [`broadcast`]
/// implements.
#[derive(Clone, Copy, Debug, PartialEq)]
enum V {
    F(f32),
    U(u32),
    B(bool),
    Vec(usize, [f32; 4]),
}

impl V {
    fn f(self) -> f32 {
        match self {
            V::F(x) => x,
            other => panic!("expected a float, got {other:?}"),
        }
    }
    fn u(self) -> u32 {
        match self {
            V::U(x) => x,
            other => panic!("expected a u32, got {other:?}"),
        }
    }
    fn b(self) -> bool {
        match self {
            V::B(x) => x,
            other => panic!("expected a bool, got {other:?}"),
        }
    }
    /// How many float lanes: one for a scalar, `n` for a vector.
    fn lanes(self) -> usize {
        match self {
            V::F(_) => 1,
            V::Vec(n, _) => n,
            other => panic!("{other:?} has no float lanes"),
        }
    }
    fn lane(self, i: usize) -> f32 {
        match self {
            V::F(x) => x,
            V::Vec(_, c) => c[i],
            other => panic!("{other:?} has no float lanes"),
        }
    }
    /// A scalar stays a scalar; anything wider becomes a vector.
    fn pack(n: usize, c: [f32; 4]) -> V {
        if n == 1 {
            V::F(c[0])
        } else {
            V::Vec(n, c)
        }
    }
}

/// Componentwise `op` over two operands, either of which may be a
/// scalar standing in for every lane — WGSL's own mixed arithmetic.
fn broadcast(a: V, b: V, op: impl Fn(f32, f32) -> f32) -> V {
    let n = a.lanes().max(b.lanes());
    let mut out = [0.0f32; 4];
    for (i, slot) in out.iter_mut().enumerate().take(n) {
        *slot = op(a.lane(i.min(a.lanes() - 1)), b.lane(i.min(b.lanes() - 1)));
    }
    V::pack(n, out)
}

// ------------------------------------------------------------- evaluator

/// One run of a WGSL function over naga's IR.
///
/// Expressions are evaluated where the IR says they are — on the `Emit`
/// statement that makes them available — and cached by handle, which is
/// the only way a `Load` of a variable that is stored to three times can
/// answer three different numbers. A lazy evaluator would read every one
/// of them as the last value written.
pub struct Field<'a> {
    module: &'a Module,
    func: &'a naga::Function,
    args: Vec<V>,
    ready: Vec<Option<V>>,
    locals: Vec<Option<V>>,
}

impl<'a> Field<'a> {
    /// The named function of `module`, ready to be called.
    pub fn new(module: &'a Module, name: &str) -> Field<'a> {
        let (_, func) = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("the shader declares no `{name}`"));
        Field {
            module,
            func,
            args: Vec::new(),
            ready: vec![None; func.expressions.len()],
            locals: vec![None; func.local_variables.len()],
        }
    }

    /// Calls it. `args` are in declaration order and must match the
    /// signature's types; the result is the returned f32.
    fn call(&mut self, args: Vec<V>) -> f32 {
        assert_eq!(args.len(), self.func.arguments.len(), "wrong argument count");
        self.args = args;
        self.ready.iter_mut().for_each(|s| *s = None);
        self.locals.iter_mut().for_each(|s| *s = None);
        for (h, l) in self.func.local_variables.iter() {
            if let Some(init) = l.init {
                let v = self.ev(&self.func.expressions, init);
                self.locals[h.index()] = Some(v);
            }
        }
        let body = &self.func.body;
        self.run(body).expect("the function returned nothing")
    }

    fn run(&mut self, block: &naga::Block) -> Option<f32> {
        for st in block.iter() {
            match st {
                // The IR's own evaluation order: these expressions become
                // available HERE, with the variables holding what they
                // hold at this point in the block.
                St::Emit(range) => {
                    for h in range.clone() {
                        // A range is a span of HANDLES and may take in
                        // expressions that are not computed at all — a
                        // pointer to a local, a constant. Naga names
                        // those itself.
                        if self.func.expressions[h].needs_pre_emit()
                            || self.is_place(&self.func.expressions, h)
                        {
                            continue;
                        }
                        let v = self.ev(&self.func.expressions, h);
                        self.ready[h.index()] = Some(v);
                    }
                }
                St::Block(inner) => {
                    if let Some(v) = self.run(inner) {
                        return Some(v);
                    }
                }
                St::Store { pointer, value } => {
                    let v = self.ev(&self.func.expressions, *value);
                    match self.func.expressions[*pointer] {
                        E::LocalVariable(l) => self.locals[l.index()] = Some(v),
                        ref other => panic!("store into {other:?}"),
                    }
                }
                St::Return { value } => {
                    let h = value.expect("a value-returning function must return one");
                    return Some(self.ev(&self.func.expressions, h).f());
                }
                other => panic!(
                    "`{}` grew a statement this evaluator cannot run: {other:?}",
                    self.func.name.as_deref().unwrap_or("?")
                ),
            }
        }
        None
    }

    /// Whether `h` names a STORAGE LOCATION rather than a value — a
    /// local, or a component of one. Such an expression has no value of
    /// its own to cache: it answers whatever the variable holds at the
    /// moment it is read, which is the whole point of a variable.
    fn is_place(&self, arena: &naga::Arena<E>, h: Handle<E>) -> bool {
        match arena[h] {
            E::LocalVariable(_) | E::GlobalVariable(_) => true,
            E::AccessIndex { base, .. } | E::Access { base, .. } => self.is_place(arena, base),
            _ => false,
        }
    }

    /// What a place currently holds. A pointer chain (`hp.x`) is walked
    /// to the variable at its root and read there.
    fn place(&self, arena: &naga::Arena<E>, h: Handle<E>) -> V {
        match arena[h] {
            E::LocalVariable(l) => {
                self.locals[l.index()].expect("read of a local that was never written")
            }
            E::FunctionArgument(i) => self.args[i as usize],
            E::AccessIndex { base, index } => V::F(self.place(arena, base).lane(index as usize)),
            E::Access { base, index } => {
                let i = self.ev(arena, index).u() as usize;
                V::F(self.place(arena, base).lane(i))
            }
            // Not a place at all: an ordinary value, read as one.
            _ => self.ev(arena, h),
        }
    }

    fn ev(&self, arena: &naga::Arena<E>, h: Handle<E>) -> V {
        // Only the function's own arena is cached: const expressions are
        // a different arena whose handles share the numbering.
        if std::ptr::eq(arena, &self.func.expressions) {
            if let Some(v) = self.ready[h.index()] {
                return v;
            }
        }
        let ev = |x: Handle<E>| self.ev(arena, x);
        match arena[h] {
            E::Literal(l) => match l {
                naga::Literal::F32(x) => V::F(x),
                naga::Literal::F64(x) => V::F(x as f32),
                naga::Literal::U32(x) => V::U(x),
                naga::Literal::I32(x) => V::U(x as u32),
                naga::Literal::I64(x) => V::U(x as u32),
                naga::Literal::Bool(x) => V::B(x),
                naga::Literal::AbstractInt(x) => V::U(x as u32),
                naga::Literal::AbstractFloat(x) => V::F(x as f32),
            },
            E::Constant(c) => {
                self.ev(&self.module.const_expressions, self.module.constants[c].init)
            }
            E::ZeroValue(ty) => match self.module.types[ty].inner {
                TypeInner::Scalar(s) if s.kind == ScalarKind::Uint => V::U(0),
                TypeInner::Scalar(_) => V::F(0.0),
                TypeInner::Vector { size, .. } => V::Vec(size as usize, [0.0; 4]),
                ref other => panic!("no zero for {other:?}"),
            },
            E::Compose { ref components, .. } => {
                let mut out = [0.0f32; 4];
                let mut n = 0;
                for c in components {
                    let v = ev(*c);
                    for i in 0..v.lanes() {
                        out[n] = v.lane(i);
                        n += 1;
                    }
                }
                V::pack(n, out)
            }
            E::Splat { size, value } => {
                let x = ev(value).f();
                V::Vec(size as usize, [x; 4])
            }
            E::Swizzle { size, vector, pattern } => {
                let v = ev(vector);
                let mut out = [0.0f32; 4];
                for (i, slot) in out.iter_mut().enumerate().take(size as usize) {
                    *slot = v.lane(match pattern[i] {
                        Sw::X => 0,
                        Sw::Y => 1,
                        Sw::Z => 2,
                        Sw::W => 3,
                    });
                }
                V::pack(size as usize, out)
            }
            E::AccessIndex { base, index } => V::F(self.place(arena, base).lane(index as usize)),
            E::Access { base, index } => {
                let i = ev(index).u() as usize;
                V::F(self.place(arena, base).lane(i))
            }
            E::FunctionArgument(i) => self.args[i as usize],
            E::Load { pointer } => self.place(arena, pointer),
            E::Unary { op, expr } => {
                let v = ev(expr);
                match op {
                    Un::Negate => broadcast(v, V::F(-1.0), |a, b| a * b),
                    other => panic!("unary {other:?}"),
                }
            }
            E::Binary { op, left, right } => {
                let (a, b) = (ev(left), ev(right));
                match (op, a, b) {
                    (Bin::Add, V::U(x), V::U(y)) => V::U(x.wrapping_add(y)),
                    (Bin::Subtract, V::U(x), V::U(y)) => V::U(x.wrapping_sub(y)),
                    (Bin::Multiply, V::U(x), V::U(y)) => V::U(x.wrapping_mul(y)),
                    (Bin::And, V::U(x), V::U(y)) => V::U(x & y),
                    (Bin::InclusiveOr, V::U(x), V::U(y)) => V::U(x | y),
                    (Bin::ShiftLeft, V::U(x), V::U(y)) => V::U(x << y),
                    (Bin::ShiftRight, V::U(x), V::U(y)) => V::U(x >> y),
                    (Bin::Equal, V::U(x), V::U(y)) => V::B(x == y),
                    (Bin::NotEqual, V::U(x), V::U(y)) => V::B(x != y),
                    (Bin::Equal, _, _) => V::B(a.f() == b.f()),
                    (Bin::NotEqual, _, _) => V::B(a.f() != b.f()),
                    (Bin::Less, _, _) => V::B(a.f() < b.f()),
                    (Bin::LessEqual, _, _) => V::B(a.f() <= b.f()),
                    (Bin::Greater, _, _) => V::B(a.f() > b.f()),
                    (Bin::GreaterEqual, _, _) => V::B(a.f() >= b.f()),
                    (Bin::Add, _, _) => broadcast(a, b, |x, y| x + y),
                    (Bin::Subtract, _, _) => broadcast(a, b, |x, y| x - y),
                    (Bin::Multiply, _, _) => broadcast(a, b, |x, y| x * y),
                    (Bin::Divide, _, _) => broadcast(a, b, |x, y| x / y),
                    (other, _, _) => panic!("binary {other:?}"),
                }
            }
            E::Select { condition, accept, reject } => {
                if ev(condition).b() {
                    ev(accept)
                } else {
                    ev(reject)
                }
            }
            E::Math { fun, arg, arg1, arg2, .. } => {
                let a = ev(arg);
                let b = arg1.map(ev);
                let c = arg2.map(ev);
                match fun {
                    M::Abs => broadcast(a, V::F(0.0), |x, _| x.abs()),
                    M::Sqrt => broadcast(a, V::F(0.0), |x, _| x.sqrt()),
                    M::Sin => V::F(a.f().sin()),
                    M::Cos => V::F(a.f().cos()),
                    M::Min => broadcast(a, b.expect("min takes two"), f32::min),
                    M::Max => broadcast(a, b.expect("max takes two"), f32::max),
                    M::Clamp => {
                        let lo = broadcast(a, b.expect("clamp takes three"), f32::max);
                        broadcast(lo, c.expect("clamp takes three"), f32::min)
                    }
                    M::Length => {
                        V::F((0..a.lanes()).map(|i| a.lane(i) * a.lane(i)).sum::<f32>().sqrt())
                    }
                    M::Dot => {
                        let b = b.expect("dot takes two");
                        V::F((0..a.lanes()).map(|i| a.lane(i) * b.lane(i)).sum())
                    }
                    other => panic!("math {other:?}"),
                }
            }
            E::As { expr, kind, .. } => {
                let v = ev(expr);
                match (kind, v) {
                    (ScalarKind::Float, V::U(x)) => V::F(x as f32),
                    (ScalarKind::Uint, V::F(x)) => V::U(x as u32),
                    (ScalarKind::Sint, V::F(x)) => V::U(x as i32 as u32),
                    _ => v,
                }
            }
            ref other => panic!("expression {other:?}"),
        }
    }
}

// ------------------------------------------------------- the record's face

/// The six numbers `shape_field` takes, in its own order.
#[derive(Clone, Copy, Debug)]
pub struct Record {
    pub half: [f32; 2],
    pub corner: [f32; 4],
    pub flags: u32,
    pub arc_half: f32,
    pub arc_dir: f32,
}

impl Field<'_> {
    /// The distance the shader computes for `rec` at local point `p`.
    pub fn at(&mut self, rec: Record, p: [f32; 2]) -> f32 {
        self.call(vec![
            V::Vec(2, [p[0], p[1], 0.0, 0.0]),
            V::Vec(2, [rec.half[0], rec.half[1], 0.0, 0.0]),
            V::Vec(4, rec.corner),
            V::U(rec.flags),
            V::F(rec.arc_half),
            V::F(rec.arc_dir),
        ])
    }
}

// ---------------------------------------------------------- the geometry

/// Exact distance from `p` to the segment `a`–`b`.
fn seg(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (vx, vy) = (b[0] - a[0], b[1] - a[1]);
    let (wx, wy) = (p[0] - a[0], p[1] - a[1]);
    let len2 = vx * vx + vy * vy;
    let t = if len2 > 0.0 { ((wx * vx + wy * vy) / len2).clamp(0.0, 1.0) } else { 0.0 };
    (wx - t * vx).hypot(wy - t * vy)
}

/// How far `p` is from an OPEN polyline — the arc's own centre curve.
pub fn curve_dist(curve: &[[f32; 2]], p: [f32; 2]) -> f32 {
    curve.windows(2).map(|s| seg(p, s[0], s[1])).fold(f32::INFINITY, f32::min)
}

/// Signed distance to a CLOSED outline: how far to the nearest edge,
/// negative inside. Inside is decided by counting crossings of a ray,
/// which knows nothing about how far anything is — so the sign and the
/// magnitude cannot be wrong together in the same way.
pub fn poly_sd(poly: &[[f32; 2]], p: [f32; 2]) -> f32 {
    let mut d = f32::INFINITY;
    let mut inside = false;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        d = d.min(seg(p, a, b));
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if p[0] < x {
                inside = !inside;
            }
        }
    }
    if inside {
        -d
    } else {
        d
    }
}

/// A point `radius` from the origin at `angle` MEASURED ON THE GLASS:
/// from +x toward +y, and +y is downward here, so a growing angle runs
/// 3 o'clock → 6 o'clock → 9 o'clock. Clockwise.
///
/// This is the fixed frame every expectation below is stated in, and it
/// is written out rather than borrowed from either implementation: a
/// rotation checked against a copy of itself agrees whichever way it
/// turns.
pub fn at(radius: f32, angle: f32) -> [f32; 2] {
    [radius * angle.cos(), radius * angle.sin()]
}

const ARC_STEPS: usize = 96;

/// The outline of the box family: four corners, each square, round or
/// chamfered, in `ring_points`' own order — tl, tr, br, bl — walked
/// clockwise on the glass.
pub fn box_poly(b: [f32; 2], corners: [(u32, f32); 4]) -> Vec<[f32; 2]> {
    let (bx, by) = (b[0], b[1]);
    // Corner point, then the centre of its arc, then the arc's start
    // angle: tl runs 180°→270°, and each next corner is a quarter turn
    // further round.
    let geom = [
        ([-bx, -by], [1.0f32, 1.0f32], std::f32::consts::PI),
        ([bx, -by], [-1.0, 1.0], std::f32::consts::PI * 1.5),
        ([bx, by], [-1.0, -1.0], 0.0),
        ([-bx, by], [1.0, -1.0], std::f32::consts::FRAC_PI_2),
    ];
    // Which way the outline ENTERS and LEAVES each corner, for the
    // chamfer's two points.
    let cuts = [
        ([0.0f32, -1.0f32], [1.0f32, 0.0f32]),
        ([1.0, 0.0], [0.0, 1.0]),
        ([0.0, 1.0], [-1.0, 0.0]),
        ([-1.0, 0.0], [0.0, -1.0]),
    ];
    let mut out = Vec::new();
    for (i, (p, inward, a0)) in geom.iter().enumerate() {
        let (style, k) = corners[i];
        let (din, dout) = cuts[i];
        match style {
            1 => {
                let c = [p[0] + inward[0] * k, p[1] + inward[1] * k];
                for s in 0..=ARC_STEPS {
                    let a = a0 + std::f32::consts::FRAC_PI_2 * s as f32 / ARC_STEPS as f32;
                    let q = at(k, a);
                    out.push([c[0] + q[0], c[1] + q[1]]);
                }
            }
            2 => {
                out.push([p[0] - din[0] * k, p[1] - din[1] * k]);
                out.push([p[0] + dout[0] * k, p[1] + dout[1] * k]);
            }
            _ => out.push(*p),
        }
    }
    out
}

/// The regular hexagon of apothem `r`, turned by `turn`.
///
/// Its six vertices sit one circumradius out at 0°, 60°, … on the glass
/// — which at `turn = 0` puts a VERTEX at 3 o'clock and therefore a flat
/// edge at 12, the orientation `nacelle::sdf::d_hex` is written about. A
/// turn moves every vertex clockwise by that much.
pub fn hex_poly(r: f32, turn: f32) -> Vec<[f32; 2]> {
    let circum = r * 2.0 / 3.0f32.sqrt();
    (0..6).map(|k| at(circum, turn + k as f32 * std::f32::consts::FRAC_PI_3)).collect()
}

/// The chevron: the rect with each vertical end drawn in to a point at
/// mid-height, `left` and `right` px deep. Six corners, clockwise.
pub fn chevron_poly(b: [f32; 2], left: f32, right: f32) -> Vec<[f32; 2]> {
    let (bx, by) = (b[0], b[1]);
    vec![
        [-bx + left, -by],
        [bx - right, -by],
        [bx, 0.0],
        [bx - right, by],
        [-bx + left, by],
        [-bx, 0.0],
    ]
}

/// The centre curve of an annular arc: the circle of radius `ra`, swept
/// `2·half_sweep` about its middle. The middle starts at 6 o'clock — the
/// silhouette's own +y, downward on the glass — and `dir` turns it
/// clockwise from there. Half a turn or more is the whole circle.
pub fn arc_curve(ra: f32, half_sweep: f32, dir: f32) -> Vec<[f32; 2]> {
    let mid = std::f32::consts::FRAC_PI_2 + dir;
    let half = half_sweep.min(std::f32::consts::PI);
    let n = 720;
    (0..=n)
        .map(|k| at(ra, mid - half + 2.0 * half * k as f32 / n as f32))
        .collect()
}
