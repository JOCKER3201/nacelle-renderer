//! What a shader costs, counted in instructions, without a device.
//!
//! The sibling of `timing`: that module measures the GPU's clock, this
//! one measures the code the GPU is handed. Neither needs the other,
//! and this one needs no Vulkan at all — it runs on any machine, in
//! `cargo test`, in a headless container, on a laptop with the display
//! server switched off.
//!
//! The road is the one the renderer already walks at startup: WGSL ->
//! naga -> SPIR-V. `shaders::compile()` hands the words to
//! `create_shader_module`; here the same words are read back and
//! counted, so the thing measured is the module that actually ships,
//! not a paraphrase of it.
//!
//! ## What a number from here means, and what it does not
//!
//! SPIR-V is the driver's INPUT, not the hardware's instruction set.
//! Between these counts and the wave that runs there is a whole
//! compiler: it promotes the locals naga emits for every `var` into
//! registers, folds constants, hoists what is uniform, and schedules
//! the rest. So:
//!
//! * **A count from here is an upper bound on ALU, and a good ratio.**
//!   Two shaders through the same front end, counted the same way,
//!   compare honestly against each other. `fs_shape` against `fs_main`
//!   is exactly that comparison.
//! * **`compute` is the number to quote**, not `total`. Loads, stores,
//!   access chains, labels and composite shuffles are naga's
//!   bookkeeping, and mem2reg deletes most of them. Arithmetic,
//!   `GLSL.std.450` calls, texture samples and derivatives are work
//!   that survives.
//! * **Every count is static.** A branch that never runs is still
//!   counted; `grade()` carries a `textureSample` that a theme without
//!   a LUT never executes. Static cost is what the register allocator
//!   sees, so it is worth knowing, but it is not a per-fragment
//!   dynamic count.
//! * **One `GLSL.std.450` call is one instruction here.** On hardware
//!   `length` is a dot plus a square root, `pow` is a log, a multiply
//!   and an exp. That is why the report also breaks the extended
//!   instructions out by name: the reader can weigh them, and the
//!   counter does not pretend to.
//!
//! ## Reading SPIR-V
//!
//! The format is a header of five words and then a flat stream of
//! instructions, each one word of `(word_count << 16) | opcode`
//! followed by its operands. That is the whole of what this module
//! needs to know; the opcode NAMES come from the `spirv` crate, which
//! is already in this tree as naga's own dependency, so nothing is
//! transcribed by hand and nothing new is downloaded.

use std::collections::{BTreeMap, BTreeSet};

use spirv::{GLOp, Op};

/// SPIR-V's magic number, first word of every module.
const MAGIC: u32 = 0x0723_0203;

/// The extended instruction set naga emits every builtin through.
const GLSL_STD_450: &str = "GLSL.std.450";

// ---------------------------------------------------------------- class

/// What kind of work an instruction is.
///
/// The grouping follows the SPIR-V specification's own chapters, by
/// opcode number, so it does not drift with a crate version: §3.42.14
/// Conversion, .15 Arithmetic, .16 Relational and Logical, .17 Bit are
/// all `Alu`; .10 Image is `Texture`; .18 Derivative is `Deriv`; and so
/// on. What is NOT from the specification is the split into "work that
/// survives the driver" and "bookkeeping that does not" — that is this
/// module's judgement, and it lives in [`Class::is_compute`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// Arithmetic, comparison, logic, bit twiddling, conversion,
    /// `OpSelect`. Real work, one instruction or a few.
    Alu,
    /// `OpExtInst` — a call into `GLSL.std.450`: `length`, `pow`,
    /// `clamp`, `sqrt`, `mix`. One here, several on the hardware.
    Math,
    /// The image chapter: sampling, fetching, querying. The expensive
    /// one, and the one with latency rather than just cost.
    Texture,
    /// `dpdx` / `dpdy` / `fwidth`. Cheap in issue slots, but they force
    /// the value to be live across the whole 2x2 quad.
    Deriv,
    /// Loads, stores, access chains, locals, `arrayLength`. Mostly
    /// naga's bookkeeping when the pointer is a function-local; a real
    /// memory trip when it is a storage buffer.
    Memory,
    /// Composite construct / extract / insert, vector shuffles, copies.
    /// Register naming on hardware, near-free.
    Composite,
    /// Labels, branches, merges, phis, calls, returns. Structure.
    Flow,
    /// Anything the classifier does not place. A shader of ours that
    /// produces one of these is a shader this module has stopped
    /// understanding — hence the test that the count is zero.
    Other,
}

impl Class {
    /// Every class, in report order.
    pub const ALL: [Class; 8] = [
        Class::Alu,
        Class::Math,
        Class::Texture,
        Class::Deriv,
        Class::Memory,
        Class::Composite,
        Class::Flow,
        Class::Other,
    ];

    /// The column head this class prints under.
    pub fn label(self) -> &'static str {
        match self {
            Class::Alu => "alu",
            Class::Math => "math",
            Class::Texture => "tex",
            Class::Deriv => "deriv",
            Class::Memory => "mem",
            Class::Composite => "comp",
            Class::Flow => "flow",
            Class::Other => "other",
        }
    }

    /// Does this class survive the driver's optimiser?
    ///
    /// The four that do are the ones worth quoting as "the cost of the
    /// shader"; the rest are how naga spells the source, not what the
    /// hardware runs.
    pub fn is_compute(self) -> bool {
        matches!(self, Class::Alu | Class::Math | Class::Texture | Class::Deriv)
    }
}

/// Place an opcode in its class.
///
/// Ranges, not a list of names: the specification assigns opcodes in
/// chapters, and a chapter is a class. Numbers that no instruction uses
/// simply never arrive.
pub fn classify(opcode: u32) -> Class {
    match opcode {
        // §3.42.12 Extension: OpExtInst, the GLSL.std.450 call.
        12 => Class::Math,
        // §3.42.9 Function: OpFunction, OpFunctionParameter,
        // OpFunctionEnd, OpFunctionCall.
        54..=57 => Class::Flow,
        // §3.42.8 Memory: OpVariable .. OpInBoundsPtrAccessChain.
        // OpImageTexelPointer (60) lives here in the specification but
        // is an image address, so it is left with the memory ops.
        59..=70 => Class::Memory,
        // §3.42.12 Composite: OpVectorExtractDynamic .. OpTranspose.
        77..=84 => Class::Composite,
        // §3.42.10 Image: OpSampledImage .. OpImageSparse*.
        86..=107 => Class::Texture,
        // §3.42.11 Conversion.
        109..=124 => Class::Alu,
        // §3.42.13 Arithmetic (incl. OpDot and the extended multiplies).
        126..=153 => Class::Alu,
        // §3.42.14 Relational and Logical, OpSelect (169) among them.
        154..=191 => Class::Alu,
        // §3.42.15 Bit.
        194..=205 => Class::Alu,
        // §3.42.16 Derivative.
        207..=215 => Class::Deriv,
        // §3.42.17 Control-Flow: OpPhi .. OpLifetimeStop.
        245..=257 => Class::Flow,
        // §3.42.1 Miscellaneous: OpNop, OpUndef, OpSizeOf.
        0..=1 => Class::Composite,
        // §3.42.3 Debug and §3.42.4 Annotation, if debug info is on.
        2..=8 | 71..=75 | 317 => Class::Other,
        _ => Class::Other,
    }
}

/// The printable name of an opcode, from the `spirv` crate's table.
pub fn op_name(opcode: u32) -> String {
    match Op::from_u32(opcode) {
        Some(op) => format!("{op:?}"),
        None => format!("Op#{opcode}"),
    }
}

/// The printable name of a `GLSL.std.450` instruction.
pub fn glsl_name(inst: u32) -> String {
    match GLOp::from_u32(inst) {
        Some(op) => format!("{op:?}"),
        None => format!("GL#{inst}"),
    }
}

// ---------------------------------------------------------------- tally

/// What a stretch of code contains, opcode by opcode.
///
/// Kept as counts against numbers rather than names so that summing two
/// tallies is arithmetic, not string work; the names are put on at the
/// moment of printing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    /// opcode -> how many times it appears.
    pub by_op: BTreeMap<u32, u32>,
    /// `GLSL.std.450` instruction -> how many times it is called.
    pub by_glsl: BTreeMap<u32, u32>,
    /// `OpExtInst` against a set that is not `GLSL.std.450`. None of
    /// our shaders has one; if one appears, it is visible rather than
    /// silently folded into `math`.
    pub foreign_ext: u32,
}

impl Tally {
    /// Every instruction, of every class.
    pub fn total(&self) -> u32 {
        self.by_op.values().sum()
    }

    /// Instructions of one class.
    pub fn class(&self, class: Class) -> u32 {
        self.by_op
            .iter()
            .filter(|(op, _)| classify(**op) == class)
            .map(|(_, n)| *n)
            .sum()
    }

    /// The number worth quoting: arithmetic, extended-math calls,
    /// texture and derivatives. See the module header.
    pub fn compute(&self) -> u32 {
        Class::ALL
            .iter()
            .filter(|c| c.is_compute())
            .map(|c| self.class(*c))
            .sum()
    }

    /// Trips to memory through a sampler: samples, fetches, gathers,
    /// reads.
    ///
    /// Not the same as the `tex` column. The image chapter also holds
    /// `OpSampledImage`, which only pairs a texture with a sampler and
    /// costs nothing, and the queries, which cost nearly nothing; one
    /// `textureSample` in WGSL is two instructions in that column and
    /// one trip here. When the question is latency, this is the number.
    pub fn samples(&self) -> u32 {
        self.by_op
            .iter()
            .filter(|(op, _)| {
                let name = op_name(**op);
                name.starts_with("Image")
                    && !name.starts_with("ImageQuery")
                    && name != "ImageTexelPointer"
            })
            .map(|(_, n)| *n)
            .sum()
    }

    /// One line of class counts, for a report or a commit message.
    pub fn line(&self) -> String {
        let mut s = format!(
            "{} instructions, {} compute (",
            self.total(),
            self.compute()
        );
        let parts: Vec<String> = Class::ALL
            .iter()
            .map(|c| format!("{}={}", c.label(), self.class(*c)))
            .collect();
        s.push_str(&parts.join(" "));
        s.push_str(&format!(", samples={})", self.samples()));
        s
    }

    /// Fold another tally into this one.
    pub fn add(&mut self, other: &Tally) {
        for (op, n) in &other.by_op {
            *self.by_op.entry(*op).or_insert(0) += n;
        }
        for (inst, n) in &other.by_glsl {
            *self.by_glsl.entry(*inst).or_insert(0) += n;
        }
        self.foreign_ext += other.foreign_ext;
    }

    /// Opcodes with their names, most frequent first, ties by name.
    pub fn ops(&self) -> Vec<(String, u32)> {
        let mut v: Vec<(String, u32)> =
            self.by_op.iter().map(|(op, n)| (op_name(*op), *n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    }

    /// `GLSL.std.450` calls with their names, most frequent first.
    pub fn glsl(&self) -> Vec<(String, u32)> {
        let mut v: Vec<(String, u32)> = self
            .by_glsl
            .iter()
            .map(|(inst, n)| (glsl_name(*inst), *n))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    }

    fn bump(&mut self, opcode: u32) {
        *self.by_op.entry(opcode).or_insert(0) += 1;
    }
}

// ------------------------------------------------------------- the module

/// One function of the module: its body, and who it calls.
#[derive(Clone, Debug)]
pub struct Function {
    /// The SPIR-V result id of `OpFunction`.
    pub id: u32,
    /// From `OpName`, when the writer emitted debug names — naga does,
    /// unless `WriterFlags::DEBUG` is cleared.
    pub name: Option<String>,
    /// Everything between `OpFunction` and `OpFunctionEnd`, those two
    /// excluded: the body, its label, its locals, its parameters.
    pub body: Tally,
    /// Ids called from this body, in the order first seen.
    pub calls: Vec<u32>,
}

impl Function {
    /// The name if there is one, `%id` if there is not.
    pub fn label(&self) -> String {
        match &self.name {
            Some(n) => n.clone(),
            None => format!("%{}", self.id),
        }
    }
}

/// One entry point: the name the pipeline asks for, and its stage.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    /// SPIR-V execution model: 0 vertex, 4 fragment, 5 compute.
    pub model: u32,
    /// The id of the function this entry point runs.
    pub func: u32,
}

impl Entry {
    /// The stage, spelled the way a pipeline description spells it.
    pub fn stage(&self) -> &'static str {
        match self.model {
            0 => "vertex",
            1 => "tess-control",
            2 => "tess-eval",
            3 => "geometry",
            4 => "fragment",
            5 => "compute",
            _ => "other",
        }
    }
}

/// A parsed SPIR-V module: its functions, its entry points, its size.
#[derive(Clone, Debug)]
pub struct Module {
    pub functions: Vec<Function>,
    pub entries: Vec<Entry>,
    /// Length of the binary in 32-bit words.
    pub words: usize,
}

/// An entry point costed: its own body, and its body plus everything it
/// reaches through calls.
#[derive(Clone, Debug)]
pub struct EntryCost {
    pub name: String,
    pub stage: &'static str,
    /// The entry function's own body.
    pub own: Tally,
    /// The body plus every function reachable from it, each counted
    /// once. There are no loops over calls in these shaders, so once is
    /// also the dynamic count; a call inside a loop would need the
    /// caveat spelled out.
    pub total: Tally,
    /// Names of the functions reached, in breadth-first order.
    pub callees: Vec<String>,
}

impl Module {
    /// Find a function by id.
    pub fn function(&self, id: u32) -> Option<&Function> {
        self.functions.iter().find(|f| f.id == id)
    }

    /// Cost one entry point, following its calls.
    ///
    /// Breadth-first with a visited set, so a recursive module (SPIR-V
    /// forbids recursion, but a hand-built binary can still contain it)
    /// terminates instead of hanging.
    pub fn cost(&self, entry: &Entry) -> EntryCost {
        let mut total = Tally::default();
        let mut callees = Vec::new();
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        let mut queue = vec![entry.func];
        seen.insert(entry.func);

        let own = self
            .function(entry.func)
            .map(|f| f.body.clone())
            .unwrap_or_default();

        while let Some(id) = queue.pop() {
            let Some(f) = self.function(id) else { continue };
            total.add(&f.body);
            if id != entry.func {
                callees.push(f.label());
            }
            for c in &f.calls {
                if seen.insert(*c) {
                    queue.push(*c);
                }
            }
        }
        callees.sort();

        EntryCost {
            name: entry.name.clone(),
            stage: entry.stage(),
            own,
            total,
            callees,
        }
    }

    /// Every entry point, costed, in the order the module declares them.
    pub fn costs(&self) -> Vec<EntryCost> {
        self.entries.iter().map(|e| self.cost(e)).collect()
    }
}

// ---------------------------------------------------------------- parsing

/// Read a SPIR-V binary into functions and entry points.
///
/// Nothing here validates the module — that is the driver's job, and
/// naga's before it. This walks the instruction stream, which is all
/// that counting needs, and refuses only what would make the walk
/// meaningless: a wrong magic number, a word count of zero (which would
/// never advance), an instruction running past the end.
pub fn parse(words: &[u32]) -> Result<Module, String> {
    if words.len() < 5 {
        return Err(format!("too short for a SPIR-V header: {} words", words.len()));
    }
    if words[0] != MAGIC {
        return Err(format!("not SPIR-V: first word is {:#010x}", words[0]));
    }

    let mut functions: Vec<Function> = Vec::new();
    let mut entries: Vec<Entry> = Vec::new();
    let mut names: BTreeMap<u32, String> = BTreeMap::new();
    let mut ext_sets: BTreeMap<u32, String> = BTreeMap::new();
    let mut open: Option<Function> = None;

    let mut i = 5usize;
    while i < words.len() {
        let head = words[i];
        let count = (head >> 16) as usize;
        let opcode = head & 0xffff;
        if count == 0 {
            return Err(format!("instruction at word {i} has a word count of zero"));
        }
        if i + count > words.len() {
            return Err(format!(
                "instruction at word {i} runs {count} words past the end of {} ",
                words.len()
            ));
        }
        let inst = &words[i..i + count];

        match opcode {
            // OpExtInstImport: id, then the set's name.
            11 => {
                if let (Some(id), Some(name)) = (inst.get(1), literal_string(&inst[2..])) {
                    ext_sets.insert(*id, name);
                }
            }
            // OpEntryPoint: execution model, function id, name, interface.
            15 => {
                if let (Some(model), Some(func), Some(name)) =
                    (inst.get(1), inst.get(2), literal_string(&inst[3..]))
                {
                    entries.push(Entry {
                        name,
                        model: *model,
                        func: *func,
                    });
                }
            }
            // OpName: target id, then the name.
            5 => {
                if let (Some(id), Some(name)) = (inst.get(1), literal_string(&inst[2..])) {
                    names.insert(*id, name);
                }
            }
            // OpFunction: result type, result id, control, function type.
            54 => {
                if let Some(id) = inst.get(2) {
                    open = Some(Function {
                        id: *id,
                        name: None,
                        body: Tally::default(),
                        calls: Vec::new(),
                    });
                }
            }
            // OpFunctionEnd.
            56 => {
                if let Some(f) = open.take() {
                    functions.push(f);
                }
            }
            _ => {}
        }

        // The body is everything strictly between OpFunction and
        // OpFunctionEnd; the two brackets are structure, not work.
        if let Some(f) = open.as_mut() {
            if opcode != 54 {
                f.body.bump(opcode);
                // OpExtInst: result type, result id, set id, instruction.
                if opcode == 12 {
                    match (inst.get(3), inst.get(4)) {
                        (Some(set), Some(which))
                            if ext_sets.get(set).map(String::as_str) == Some(GLSL_STD_450) =>
                        {
                            *f.body.by_glsl.entry(*which).or_insert(0) += 1;
                        }
                        _ => f.body.foreign_ext += 1,
                    }
                }
                // OpFunctionCall: result type, result id, function id.
                if opcode == 57 {
                    if let Some(callee) = inst.get(3) {
                        if !f.calls.contains(callee) {
                            f.calls.push(*callee);
                        }
                    }
                }
            }
        }

        i += count;
    }

    if open.is_some() {
        return Err("the binary ends inside a function".to_string());
    }

    for f in &mut functions {
        f.name = names.get(&f.id).cloned();
    }

    Ok(Module {
        functions,
        entries,
        words: words.len(),
    })
}

/// Decode a SPIR-V literal string: UTF-8, four bytes to a word, little
/// endian, terminated by a zero byte and padded to the word.
fn literal_string(words: &[u32]) -> Option<String> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for w in words {
        for shift in [0, 8, 16, 24] {
            let b = ((w >> shift) & 0xff) as u8;
            if b == 0 {
                return String::from_utf8(bytes).ok();
            }
            bytes.push(b);
        }
    }
    // No terminator: not a literal string, so nothing to report.
    None
}

// -------------------------------------------------------------- the road in

/// Compile WGSL and count what comes out.
///
/// The writer options MIRROR `shaders::compile` — same flags, same
/// coordinate space — so a foreign source measured here is measured the
/// way this renderer would ship it. The test
/// `the_foreign_road_and_the_shipping_road_agree` holds the two
/// together.
pub fn from_wgsl(src: &str) -> Result<Module, String> {
    let module = naga::front::wgsl::parse_str(src).map_err(|e| format!("WGSL: {e}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("validation: {e:?}"))?;

    let mut options = naga::back::spv::Options::default();
    options
        .flags
        .remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);

    let words = naga::back::spv::write_vec(&module, &info, &options, None)
        .map_err(|e| format!("SPIR-V: {e:?}"))?;
    parse(&words)
}

/// The shaders this renderer actually uploads, counted.
pub fn builtin() -> Result<Module, String> {
    parse(&crate::shaders::compile())
}

/// Pull the WGSL out of a Rust source that carries it as a raw string.
///
/// This is what makes the tool answer questions across branches: the
/// shader of another branch is a `pub const WGSL_SRC: &str = r#"..."#;`
/// inside `shaders.rs`, and `git show <branch>:src/shaders.rs` prints
/// it. Handing that file here needs no checkout of anyone's tree.
pub fn wgsl_from_rust(src: &str) -> Option<&str> {
    let start = src.find("r#\"")? + 3;
    let rest = &src[start..];
    let end = rest.find("\"#")?;
    Some(&rest[..end])
}

// ---------------------------------------------------------------- report

/// The table: one block per entry point, then the shared functions.
pub fn report(module: &Module) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "SPIR-V: {} words ({} B), {} functions, {} entry points\n",
        module.words,
        module.words * 4,
        module.functions.len(),
        module.entries.len()
    ));
    out.push_str(
        "\ncost = own body + everything it calls; `compute` is alu+math+tex+deriv\n\n",
    );

    out.push_str(&format!(
        "{:<10} {:<9} {:>6} {:>8}",
        "entry", "stage", "total", "compute"
    ));
    for c in Class::ALL {
        out.push_str(&format!(" {:>6}", c.label()));
    }
    out.push_str(&format!(" {:>6}\n", "samp"));

    for cost in module.costs() {
        out.push_str(&format!(
            "{:<10} {:<9} {:>6} {:>8}",
            cost.name,
            cost.stage,
            cost.total.total(),
            cost.total.compute()
        ));
        for c in Class::ALL {
            out.push_str(&format!(" {:>6}", cost.total.class(c)));
        }
        out.push_str(&format!(" {:>6}\n", cost.total.samples()));
    }

    for cost in module.costs() {
        out.push_str(&format!("\n{} ({})\n", cost.name, cost.stage));
        out.push_str(&format!("  own body: {}\n", cost.own.line()));
        if !cost.callees.is_empty() {
            out.push_str(&format!(
                "  with {}: {}\n",
                cost.callees.join(", "),
                cost.total.line()
            ));
        }
        let glsl = cost.total.glsl();
        if !glsl.is_empty() {
            let names: Vec<String> = glsl.iter().map(|(n, c)| format!("{n}x{c}")).collect();
            out.push_str(&format!("  GLSL.std.450: {}\n", names.join(" ")));
        }
        let ops = cost.total.ops();
        let head: Vec<String> = ops
            .iter()
            .take(8)
            .map(|(n, c)| format!("{n}x{c}"))
            .collect();
        out.push_str(&format!("  top opcodes: {}\n", head.join(" ")));
    }

    out
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    /// A module built by hand, word by word, so the walker can be
    /// tested without a shader: two functions, one calling the other,
    /// one entry point, one GLSL.std.450 import and one call into it.
    fn hand_built() -> Vec<u32> {
        let mut w = vec![MAGIC, 0x0001_0500, 0, 20, 0];
        // OpExtInstImport %1 "GLSL.std.450"
        let name = string_words("GLSL.std.450");
        w.push(inst_head(11, 2 + name.len()));
        w.push(1);
        w.extend_from_slice(&name);
        // OpEntryPoint Fragment %10 "fs_probe"
        let ep = string_words("fs_probe");
        w.push(inst_head(15, 3 + ep.len()));
        w.push(4); // fragment
        w.push(10);
        w.extend_from_slice(&ep);
        // OpName %11 "helper"
        let hn = string_words("helper");
        w.push(inst_head(5, 2 + hn.len()));
        w.push(11);
        w.extend_from_slice(&hn);
        // %10 = OpFunction ... { OpLabel; OpFAdd; OpFunctionCall %11; OpReturn }
        w.extend_from_slice(&[inst_head(54, 5), 0, 10, 0, 0]);
        w.extend_from_slice(&[inst_head(248, 2), 100]); // OpLabel
        w.extend_from_slice(&[inst_head(129, 5), 0, 101, 0, 0]); // OpFAdd
        w.extend_from_slice(&[inst_head(57, 4), 0, 102, 11]); // OpFunctionCall
        w.extend_from_slice(&[inst_head(253, 1)]); // OpReturn
        w.extend_from_slice(&[inst_head(56, 1)]); // OpFunctionEnd
        // %11 = OpFunction { OpLabel; OpExtInst Length; OpImageSample; OpReturn }
        w.extend_from_slice(&[inst_head(54, 5), 0, 11, 0, 0]);
        w.extend_from_slice(&[inst_head(248, 2), 200]);
        w.extend_from_slice(&[inst_head(12, 6), 0, 201, 1, GLOp::Length as u32, 0]);
        w.extend_from_slice(&[inst_head(87, 5), 0, 202, 0, 0]); // OpImageSampleImplicitLod
        w.extend_from_slice(&[inst_head(253, 1)]);
        w.extend_from_slice(&[inst_head(56, 1)]);
        w
    }

    fn inst_head(opcode: u32, words: usize) -> u32 {
        ((words as u32) << 16) | opcode
    }

    fn string_words(s: &str) -> Vec<u32> {
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        bytes
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// The walk itself: names decoded, bodies bounded by their
    /// brackets, the entry point's stage read from its execution model.
    #[test]
    fn the_walker_finds_the_functions_between_their_brackets() {
        let m = parse(&hand_built()).expect("hand-built module parses");
        assert_eq!(m.functions.len(), 2);
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.entries[0].name, "fs_probe");
        assert_eq!(m.entries[0].stage(), "fragment");

        let entry = m.function(10).expect("the entry function");
        // Label, FAdd, FunctionCall, Return — OpFunction and
        // OpFunctionEnd are brackets, not body.
        assert_eq!(entry.body.total(), 4);
        assert_eq!(entry.body.class(Class::Alu), 1);
        assert_eq!(entry.calls, vec![11]);

        let helper = m.function(11).expect("the called function");
        assert_eq!(helper.label(), "helper");
        assert_eq!(helper.body.class(Class::Math), 1);
        assert_eq!(helper.body.class(Class::Texture), 1);
    }

    /// The number a reader is given is the entry point PLUS what it
    /// calls — a shader whose work sits in a helper must not report as
    /// cheap.
    #[test]
    fn an_entry_point_costs_what_it_calls_as_well() {
        let m = parse(&hand_built()).unwrap();
        let cost = m.cost(&m.entries[0]);
        assert_eq!(cost.own.total(), 4);
        // Four of its own plus four of the helper's.
        assert_eq!(cost.total.total(), 8);
        // FAdd, the GLSL call and the sample; the label, the call, the
        // two returns are structure.
        assert_eq!(cost.total.compute(), 3);
        assert_eq!(cost.callees, vec!["helper".to_string()]);
        assert_eq!(cost.total.glsl(), vec![("Length".to_string(), 1)]);
        // One trip through a sampler, in the helper.
        assert_eq!(cost.total.samples(), 1);
        assert_eq!(cost.own.samples(), 0);
    }

    /// SPIR-V forbids recursion, but this parser eats binaries from
    /// anywhere, and a walk that hangs on one is worse than a walk that
    /// reports a wrong number.
    #[test]
    fn a_call_cycle_terminates_instead_of_hanging() {
        let mut w = vec![MAGIC, 0x0001_0500, 0, 20, 0];
        let ep = string_words("loop");
        w.push(inst_head(15, 3 + ep.len()));
        w.push(4);
        w.push(10);
        w.extend_from_slice(&ep);
        for (id, callee) in [(10u32, 11u32), (11, 10)] {
            w.extend_from_slice(&[inst_head(54, 5), 0, id, 0, 0]);
            w.extend_from_slice(&[inst_head(248, 2), id * 10]);
            w.extend_from_slice(&[inst_head(57, 4), 0, id * 10 + 1, callee]);
            w.extend_from_slice(&[inst_head(56, 1)]);
        }
        let m = parse(&w).unwrap();
        let cost = m.cost(&m.entries[0]);
        assert_eq!(cost.total.total(), 4);
    }

    /// A stream that cannot be walked is refused with a reason, never
    /// walked half way and reported as a small shader.
    #[test]
    fn a_binary_that_cannot_be_walked_is_refused_with_a_reason() {
        assert!(parse(&[1, 2, 3]).unwrap_err().contains("too short"));
        assert!(parse(&[0, 0, 0, 0, 0]).unwrap_err().contains("not SPIR-V"));

        // A word count of zero would never advance the cursor.
        let stuck = vec![MAGIC, 0, 0, 1, 0, 0x0000_0037];
        assert!(parse(&stuck).unwrap_err().contains("word count of zero"));

        // An instruction claiming more words than the binary holds.
        let over = vec![MAGIC, 0, 0, 1, 0, inst_head(5, 40), 1];
        assert!(parse(&over).unwrap_err().contains("past the end"));

        // A function nobody closed.
        let open = vec![MAGIC, 0, 0, 1, 0, inst_head(54, 5), 0, 9, 0, 0];
        assert!(parse(&open).unwrap_err().contains("inside a function"));
    }

    /// The classifier is the whole meaning of the report, so it is
    /// pinned against the `spirv` crate's own opcode numbers rather
    /// than against numbers typed here.
    #[test]
    fn every_kind_of_instruction_lands_in_its_own_class() {
        let cases = [
            (Op::FAdd, Class::Alu),
            (Op::FDiv, Class::Alu),
            (Op::Select, Class::Alu),
            (Op::FOrdGreaterThanEqual, Class::Alu),
            (Op::ShiftRightLogical, Class::Alu),
            (Op::BitwiseAnd, Class::Alu),
            (Op::Bitcast, Class::Alu),
            (Op::Dot, Class::Alu),
            (Op::ExtInst, Class::Math),
            (Op::ImageSampleImplicitLod, Class::Texture),
            (Op::SampledImage, Class::Texture),
            (Op::DPdx, Class::Deriv),
            (Op::DPdy, Class::Deriv),
            (Op::Fwidth, Class::Deriv),
            (Op::Load, Class::Memory),
            (Op::Store, Class::Memory),
            (Op::AccessChain, Class::Memory),
            (Op::ArrayLength, Class::Memory),
            (Op::Variable, Class::Memory),
            (Op::CompositeConstruct, Class::Composite),
            (Op::CompositeExtract, Class::Composite),
            (Op::VectorShuffle, Class::Composite),
            (Op::Label, Class::Flow),
            (Op::Branch, Class::Flow),
            (Op::BranchConditional, Class::Flow),
            (Op::SelectionMerge, Class::Flow),
            (Op::Phi, Class::Flow),
            (Op::Return, Class::Flow),
            (Op::ReturnValue, Class::Flow),
            (Op::FunctionCall, Class::Flow),
        ];
        for (op, class) in cases {
            assert_eq!(
                classify(op as u32),
                class,
                "{op:?} ({}) should be {class:?}",
                op as u32
            );
        }
        // The split that decides which number gets quoted.
        assert!(Class::Alu.is_compute() && Class::Math.is_compute());
        assert!(!Class::Memory.is_compute() && !Class::Flow.is_compute());
    }

    /// The names come from the crate's table, not from this file.
    #[test]
    fn opcodes_and_glsl_calls_are_named_from_the_spirv_table() {
        assert_eq!(op_name(Op::FMul as u32), "FMul");
        assert_eq!(op_name(60000), "Op#60000");
        assert_eq!(glsl_name(GLOp::Sqrt as u32), "Sqrt");
        assert_eq!(glsl_name(60000), "GL#60000");
    }

    /// The real thing: the module this renderer uploads, read back.
    /// Six entry points, each reachable, each with a body — a report
    /// of zero would mean the walker missed the function section. The
    /// last two are the vector lane's: `fs_shape` and, since K3b, the
    /// frosted `fs_shape_glass`. Both are dark in the picture
    /// (`render.vector = false`) but their code ships in the module,
    /// and the measurement is of what ships.
    #[test]
    fn the_shipping_module_reads_back_with_all_of_its_entry_points() {
        let m = builtin().expect("the renderer's own SPIR-V parses");
        let names: Vec<&str> = m.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["vs_main", "fs_main", "fs_blur", "fs_image", "fs_shape", "fs_shape_glass"]
        );
        for cost in m.costs() {
            assert!(cost.total.total() > 0, "{} has an empty body", cost.name);
            assert_eq!(cost.total.foreign_ext, 0);
        }
        // Nothing in these shaders may fall outside the classifier: an
        // opcode this module cannot place is a report that lies by
        // omission.
        for cost in m.costs() {
            assert_eq!(
                cost.total.class(Class::Other),
                0,
                "{} contains an instruction the classifier cannot place",
                cost.name
            );
        }
    }

    /// What the source says, the count must show: the atlas fragment
    /// samples a texture, exactly one fragment takes a screen
    /// derivative and it takes two of them, and the vertex stage
    /// touches no texture at all.
    #[test]
    fn the_counts_answer_to_what_the_shader_source_says() {
        let m = builtin().unwrap();
        let cost = |name: &str| {
            m.costs()
                .into_iter()
                .find(|c| c.name == name)
                .unwrap_or_else(|| panic!("{name} is missing"))
        };

        // fs_main: its own sample of the atlas plus the LUT sample
        // inside grade(), which is static cost whether or not a theme
        // has a LUT. Two trips, four instructions of the image chapter
        // — the pairing of texture with sampler is the other two.
        let fs_main = cost("fs_main");
        assert_eq!(fs_main.total.samples(), 2);
        assert_eq!(fs_main.own.samples(), 1);
        assert_eq!(fs_main.total.class(Class::Texture), 4);
        assert!(fs_main.callees.iter().any(|c| c.contains("grade")));
        // pow() is the text_gamma exponent, and it is the reason
        // fs_main has any extended math at all.
        assert!(fs_main.total.glsl().iter().any(|(n, _)| n == "Pow"));

        let vs_main = cost("vs_main");
        assert_eq!(vs_main.stage, "vertex");
        assert_eq!(vs_main.total.class(Class::Texture), 0);
        assert_eq!(vs_main.total.samples(), 0);

        // The derivatives are the shape lane's and nobody else's. Two
        // of them, because the AA width is `length(vec2(dpdx(d),
        // dpdy(d)))` — the gradient of the field, read on both screen
        // axes. `fwidth` would be ONE instruction of this chapter and
        // the wrong number on a rotated frame, so the count is what
        // tells the two apart from here, without reading the WGSL.
        //
        // The lane has TWO entry points since K3b and one `shape_cover`
        // between them, so both report the pair and neither reports
        // four: the count is per entry point INCLUDING what it calls,
        // which is exactly how a second copy of the field would show up
        // here — as a third pair, in a third fragment.
        for name in ["fs_shape", "fs_shape_glass"] {
            assert_eq!(
                cost(name).total.class(Class::Deriv),
                2,
                "{name} stopped reading the gradient on both axes"
            );
        }
        for c in m.costs() {
            if c.name.starts_with("fs_shape") {
                continue;
            }
            assert_eq!(
                c.total.class(Class::Deriv),
                0,
                "{} takes a derivative, and only the shape lane may",
                c.name
            );
        }
        // The frosted fragment samples: the pyramid, plus grade()'s
        // LUT. Its own sample is the frost and there is exactly one —
        // a second would mean a second target, which one run cannot
        // bind (the reason the rank rides the handle).
        let glass = cost("fs_shape_glass");
        assert_eq!(glass.own.samples(), 1);
        assert_eq!(glass.total.samples(), 2);
        // …and the plain shape fragment still reads no image but the
        // LUT: the record IS its picture.
        assert_eq!(cost("fs_shape").own.samples(), 0);
    }

    /// **What K3b's split into functions cost the shape lane, and what
    /// it did not.** The field's mathematics moved out of `fs_shape`'s
    /// body into `shape_field`/`shape_cover`/`over`/`shape_compose`
    /// so that the frosted entry point could call the same forms
    /// instead of carrying a second copy of them. That moved a number
    /// every measurement of this lane is quoted from, and a moved
    /// number with no note is a number somebody re-reads as a
    /// regression.
    ///
    /// Measured on both sides of the split, with this file's own
    /// walker: `fs_shape.own.compute()` 78 → 6 and `own.class(Alu)`
    /// 57 → 5, because the body is now a call; `total.total()`
    /// 174 → 199, the 25 being OpFunction / OpFunctionParameter /
    /// OpLabel / OpReturnValue / OpFunctionCall and the locals they
    /// need. THE WORK WAS UNCHANGED TO THE INSTRUCTION by that move:
    /// reachable compute stayed at 87 and reachable ALU at 63.
    ///
    /// **K6 then changed the work itself, which is what the numbers
    /// below now say.** `shape_field` computes the arc, the hexagon and
    /// the chevron for every fragment of every record, because a branch
    /// on the record's own kind is not uniform across a draw and
    /// `shape_cover` takes derivatives of what it returns. Reachable
    /// compute 87 → 161 and ALU 63 → 114 are the price of the KINDS,
    /// not of the split — the two are told apart by `own`, which stayed
    /// exactly where K3b left it. The remedy the field's own comment
    /// names is one pipeline per kind, keyed on the run; until that
    /// exists a Box record pays for a hexagon it will never draw.
    ///
    /// Which is also the lesson for whoever pins the next one: on an
    /// entry point that calls anything, `own` measures where the code
    /// was WRITTEN and `total` measures what the fragment DOES. A
    /// budget belongs on `total`.
    #[test]
    fn the_shape_lane_computes_what_it_computed_before_it_was_split_up() {
        let m = builtin().expect("the renderer's own SPIR-V parses");
        let shape = m
            .costs()
            .into_iter()
            .find(|c| c.name == "fs_shape")
            .expect("fs_shape is missing");
        assert_eq!(shape.total.compute(), 161, "the field's work changed");
        assert_eq!(shape.total.class(Class::Alu), 114, "the field's ALU changed");
        // And the body itself is now scaffolding: a handful of loads
        // and four calls. If this grows, the mathematics came back —
        // which is the second copy K3b existed to prevent.
        assert!(shape.own.compute() < 20, "the field moved back into the body");
        assert_eq!(
            shape.callees,
            vec!["grade", "shape_compose", "shape_cover", "shape_field"],
            "the plain lane's callees are the shared forms and the grade"
        );
        // The frosted entry point calls the SAME field, and its extra
        // cost is its own three layers rather than a second silhouette.
        // A copied field would show up here as a number near twice this
        // one, which is precisely the failure K3b's split prevents and
        // no string search can see.
        let glass = m
            .costs()
            .into_iter()
            .find(|c| c.name == "fs_shape_glass")
            .expect("fs_shape_glass is missing");
        assert_eq!(glass.total.compute(), 177, "the frosted lane's work changed");
        assert!(
            glass.callees.contains(&"shape_field".to_string()),
            "the frosted entry point stopped sharing the field"
        );
    }

    /// The road for a foreign source must be the road for ours, or a
    /// measurement of another branch's shader would be a measurement of
    /// different writer options.
    #[test]
    fn the_foreign_road_and_the_shipping_road_agree() {
        let shipped = builtin().unwrap();
        let through_wgsl = from_wgsl(crate::shaders::WGSL_SRC).unwrap();
        assert_eq!(shipped.words, through_wgsl.words);
        let a: Vec<(String, u32, u32)> = shipped
            .costs()
            .iter()
            .map(|c| (c.name.clone(), c.total.total(), c.total.compute()))
            .collect();
        let b: Vec<(String, u32, u32)> = through_wgsl
            .costs()
            .iter()
            .map(|c| (c.name.clone(), c.total.total(), c.total.compute()))
            .collect();
        assert_eq!(a, b);
    }

    /// The shader of another branch arrives as `shaders.rs`, not as a
    /// `.wgsl` file — the extraction is what lets one command compare
    /// two branches.
    #[test]
    fn the_wgsl_is_lifted_out_of_a_rust_source_that_carries_it() {
        let rust = "pub const WGSL_SRC: &str = r#\"\n@fragment\nfn f() {}\n\"#;\n";
        assert_eq!(
            wgsl_from_rust(rust),
            Some("\n@fragment\nfn f() {}\n")
        );
        assert_eq!(wgsl_from_rust("no raw string here"), None);
        // Our own file is the case that has to work.
        let lifted = wgsl_from_rust(include_str!("shaders.rs")).expect("shaders.rs carries WGSL");
        assert!(lifted.contains("fn fs_main("));
        assert_eq!(from_wgsl(lifted).unwrap().entries.len(), 6);
    }

    /// A refusal must arrive as a message, not as a panic in the middle
    /// of a measurement run over several files.
    #[test]
    fn a_broken_source_comes_back_as_a_message() {
        let err = from_wgsl("fn nonsense(").unwrap_err();
        assert!(err.starts_with("WGSL:"), "{err}");
    }

    /// The report is what a human reads; it has to name the entry
    /// points and both numbers.
    #[test]
    fn the_report_names_the_entry_points_and_both_numbers() {
        let text = report(&builtin().unwrap());
        for name in ["vs_main", "fs_main", "fs_blur", "fs_image", "fs_shape"] {
            assert!(text.contains(name), "the report skips {name}");
        }
        assert!(text.contains("compute"));
        assert!(text.contains("GLSL.std.450"));
        // The body alone and the body with its calls are different
        // numbers, and a reader who is given only one of them will
        // quote the wrong ratio.
        assert!(text.contains("own body"));
        assert!(text.contains("with grade"));
        assert!(text.contains("samples="));
    }

    /// **K3c, measurement 1: what one fragment of the vector lane costs
    /// against one fragment of the lane it replaces.**
    ///
    /// The numbers are pinned rather than printed, because the question
    /// K3d has to answer — "is the switch worth flipping" — is answered
    /// on these, and a figure that drifts silently is worse than none.
    /// A shader edit that moves them is meant to fail here and be
    /// re-argued in `.gap-program/pomiar-wektor-k3c.md`.
    ///
    /// # Where the pins sit, and why they moved off `own`
    ///
    /// They were written on `own` when the whole lane was one fragment
    /// body. K3b split it into functions and K6 grew the field, and
    /// after both `own` measures only the four call instructions the
    /// body has left. So the pins sit on `total`, which is what this
    /// module's own header says a budget must be taken on, and the
    /// bodies are pinned separately as the scaffolding they now are.
    ///
    /// # The ratio the plan quotes, and the ratio a frame actually pays
    ///
    /// f3 §7b compares the two lanes as `~101` against `~5`. Measured
    /// through this module, before K6, the whole fragments were 87
    /// against 13 — a ratio of **6.7**, the number
    /// `.gap-program/pomiar-wektor-k3c.md` carries and the number that
    /// corrected §7b's own 19.5 (which compared bodies and forgot that
    /// both lanes pay `grade()`: nine compute instructions and a LUT
    /// sample neither can avoid).
    ///
    /// **K6 moved it again, and this is the note §7b asked for.** The
    /// kinds are computed unconditionally — a branch on the record's
    /// kind would break the derivative uniformity `shape_cover` needs —
    /// so every fragment of every record now evaluates the arc, the
    /// hexagon and the chevron as well as the box. The whole fragments
    /// are **161 against 13, a ratio of 12.4**, and that is the number
    /// to hold up against a frame time. Everything the measurement note
    /// derives per silhouette (the 4.25x, the perimeter-to-area
    /// argument) was computed on the 6.7 and has NOT been re-run here.
    ///
    /// Two things the ratio hides, both in the vector lane's favour:
    ///
    /// * `fs_shape` takes ONE texture sample where `fs_main` takes two
    ///   — it reads no atlas. A sample is a memory trip, not an ALU
    ///   slot, so the trade is 148 arithmetic instructions against one
    ///   fetch, and which side that lands on is a property of the
    ///   device's occupancy rather than of this count.
    /// * Of the fragment's 330 instructions only 161 are compute; the
    ///   rest are loads, access chains and composite shuffles that
    ///   `mem2reg` deletes before the hardware sees them. The count is
    ///   an upper bound on ALU (this module's own header says so), and
    ///   43 of the 161 are `GLSL.std.450` calls the hardware weighs its
    ///   own way — six of them `length`, a dot and a square root each —
    ///   so the hardware-weighted figure moves in both directions and
    ///   cannot be settled from here at all.
    #[test]
    fn the_shape_lane_costs_what_the_measurement_says_it_costs() {
        let m = builtin().unwrap();
        let cost = |name: &str| {
            m.costs()
                .into_iter()
                .find(|c| c.name == name)
                .unwrap_or_else(|| panic!("{name} is missing"))
        };
        let fs_main = cost("fs_main");
        let fs_shape = cost("fs_shape");

        // The whole-fragment comparison: what the GPU actually runs,
        // grade included, because both lanes run it.
        assert_eq!(fs_main.total.compute(), 13, "the plain fill lane moved");
        assert_eq!(fs_shape.total.compute(), 161, "the shape lane moved");
        assert_eq!(fs_main.total.total(), 54);
        assert_eq!(fs_shape.total.total(), 330);

        // The bodies, which are no longer the lanes: one reads a
        // texture and grades it, the other is four calls.
        assert_eq!(fs_main.own.compute(), 4, "the plain fill body moved");
        assert_eq!(fs_shape.own.compute(), 6, "the shape body stopped being a call");

        // The trade that the ratio hides: the shape lane reads no atlas.
        assert_eq!(fs_main.total.samples(), 2);
        assert_eq!(fs_shape.total.samples(), 1);

        // And the shape of the shape lane's cost, so a rewrite that
        // keeps the total and moves the mix is still visible.
        assert_eq!(fs_shape.total.class(Class::Alu), 114);
        assert_eq!(fs_shape.total.class(Class::Math), 43);
        assert_eq!(fs_shape.total.class(Class::Deriv), 2);
        assert_eq!(fs_shape.own.class(Class::Texture), 0);
        assert_eq!(
            fs_shape.total.glsl().iter().find(|(n, _)| n == "Length").map(|(_, c)| *c),
            Some(6),
            "six square roots: the box distance, the rounded corner, the AA \
             gradient, the hexagon's fold and the arc's two branches"
        );
    }
}
