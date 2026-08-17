//! GPU timing — where the frame's milliseconds actually go.
//!
//! The renderer's frame is not one lump of work: it is a base scene
//! rendered offscreen, a pyramid of downsamples that IS the blur, a
//! composite of that pyramid back onto the swapchain, and the main
//! pass over the top. Until this module existed, `gfx.rs` held not a
//! single `VkQueryPool`, so every claim about what a pass costs was
//! arithmetic on an assumed device. This measures it instead, per
//! pass, on the machine actually running.
//!
//! # The four passes, and the two extra numbers
//!
//! Six timestamps per frame, written at fixed points in the command
//! buffer, give five adjacent spans plus the whole:
//!
//! ```text
//!   0 frame begin ─┬─ uploads   (atlas rows, textures, LUT)
//!   1 uploads end ─┼─ base      (the scene under the glass)
//!   2 base end    ─┼─ pyramid   (half / quarter / eighth, and back up)
//!   3 pyramid end ─┼─ composite (the blurred scene as one graded quad)
//!   4 main start  ─┴─ main      (everything above the glass)
//!   5 frame end
//! ```
//!
//! `frame` is 5 − 0, and `rest` is whatever `frame` has left over once
//! the named spans are subtracted — the render-pass begins and the
//! bookkeeping nobody bracketed. On a frame with glass every span is
//! present and `rest` is zero; on a frame without glass only
//! `uploads`, `main` and `frame` are, and `rest` holds the swapchain
//! pass's own setup.
//!
//! # What a span honestly means
//!
//! Every end timestamp is written at `BOTTOM_OF_PIPE`, which the spec
//! defines as "everything submitted before this point has completed",
//! and the frame's first at `TOP_OF_PIPE`. So a span is *when this
//! stretch of work finished, minus when the previous stretch
//! finished*. That is the right number for "what did the blur cost me
//! this frame", and it is NOT a claim that the hardware ran the two
//! stretches in strict isolation: GPUs overlap work across pass
//! boundaries, and a cheap pass wedged between two expensive ones can
//! read as cheaper than it is. The spans sum to the frame exactly,
//! which is the property that makes them worth reading.
//!
//! # The delayed read, and why it is not the swapchain's depth
//!
//! Timestamps are written by the GPU, asynchronously; reading them in
//! the frame that wrote them either blocks the pipeline (with
//! `WAIT_BIT`) or reads whatever was in the pool before (without it).
//! The delay that fixes this is the number of frames the CPU can be
//! *ahead* of the GPU — frames in flight — not the number of
//! swapchain images. Those differ: this renderer asks for
//! `min_image_count + 1` images (usually three), yet owns exactly one
//! command buffer, one fence and one semaphore pair, and blocks on
//! that fence at the top of every frame. So one frame is in flight,
//! and frame N's timestamps are complete the moment frame N+1's fence
//! wait returns.
//!
//! Hard-coding "1" would rot the day the renderer grows a second
//! command buffer, so the depth is a parameter: [`Schedule`] takes the
//! in-flight count, delays the read by exactly that many frames, and
//! keeps one more ring of queries than that so the frame being read is
//! never the frame being overwritten. Belt and braces: every read asks
//! for the availability bit as well and drops any span whose endpoints
//! the GPU has not published, so a wrong depth would cost samples, not
//! correctness.
//!
//! # Cost when switched off
//!
//! `NACELLE_GPU_TIMING` unset means [`GpuTiming::from_env`] returns
//! `None` before it has created a pool or allocated anything, and the
//! renderer's every timing call is a `None` check on an `Option` field.
//! No queries are recorded, no memory is held, nothing is printed.

use ash::vk;
use std::collections::VecDeque;

/// The frame's command buffer has started; nothing has run yet.
pub const SLOT_FRAME_BEGIN: usize = 0;
/// Atlas rows, application textures and the LUT have been copied.
pub const SLOT_UPLOADS_END: usize = 1;
/// The offscreen base scene — everything below the first glass run.
pub const SLOT_BASE_END: usize = 2;
/// The downsample pyramid, and the smoothing step back up.
pub const SLOT_PYRAMID_END: usize = 3;
/// The swapchain pass is open and the composite quad, if any, is
/// drawn: what follows is the main pass proper.
pub const SLOT_MAIN_START: usize = 4;
/// The swapchain pass has ended; the frame is recorded.
pub const SLOT_FRAME_END: usize = 5;

/// Timestamps per frame.
pub const SLOTS: usize = 6;

/// A named stretch of the frame, as a pair of timestamp slots.
pub struct SpanDef {
    pub name: &'static str,
    pub begin: usize,
    pub end: usize,
}

/// The spans, in the order they are reported. The whole frame comes
/// last on purpose: everything before it is a part of it, which is
/// what lets `rest` be computed by subtraction.
pub const SPANS: [SpanDef; 6] = [
    SpanDef { name: "uploads", begin: SLOT_FRAME_BEGIN, end: SLOT_UPLOADS_END },
    SpanDef { name: "base", begin: SLOT_UPLOADS_END, end: SLOT_BASE_END },
    SpanDef { name: "pyramid", begin: SLOT_BASE_END, end: SLOT_PYRAMID_END },
    SpanDef { name: "composite", begin: SLOT_PYRAMID_END, end: SLOT_MAIN_START },
    SpanDef { name: "main", begin: SLOT_MAIN_START, end: SLOT_FRAME_END },
    SpanDef { name: "frame", begin: SLOT_FRAME_BEGIN, end: SLOT_FRAME_END },
];

/// Index of the whole-frame span inside [`SPANS`].
pub const FRAME_SPAN: usize = SPANS.len() - 1;
/// Index of the synthetic leftover, which lives past [`SPANS`].
pub const REST_SPAN: usize = SPANS.len();
/// How many numbers a frame produces: the spans plus the leftover.
pub const MEASURES: usize = SPANS.len() + 1;

/// Report every this many frames when the variable says only "on".
pub const DEFAULT_INTERVAL: u32 = 120;

/// Name of the switch. One variable, three meanings: absent or off,
/// on with the default window, or on with a window in frames.
pub const ENV_VAR: &str = "NACELLE_GPU_TIMING";

/// How the environment variable's value reads.
///
/// Absent, empty, `0`, `off`, `no` or `false` — off, and the renderer
/// creates nothing. Anything else is on; a number above zero is also
/// the report interval in frames, so `NACELLE_GPU_TIMING=1` means "on,
/// default window" and `NACELLE_GPU_TIMING=600` means "on, report
/// every six hundred frames". A word that is neither off nor a number
/// is taken as on rather than as an error: the variable being set at
/// all is intent, and a typo that silently measured nothing would be
/// worse than one that measures with the default window.
pub fn interval_from_env(value: Option<&str>) -> Option<u32> {
    let raw = value?.trim();
    let lowered = raw.to_ascii_lowercase();
    if matches!(lowered.as_str(), "" | "0" | "off" | "no" | "false" | "none") {
        return None;
    }
    match lowered.parse::<u32>() {
        Ok(0) => None,
        Ok(1) => Some(DEFAULT_INTERVAL),
        Ok(n) => Some(n),
        Err(_) => Some(DEFAULT_INTERVAL),
    }
}

/// Whether this device can be timed at all, and if not, why not in
/// words the log can print.
///
/// `valid_bits` is the queue family's `timestampValidBits`, which is
/// the authoritative answer for the queue the frames actually go to;
/// `compute_and_graphics` is the device-wide promise that every
/// graphics or compute family has non-zero bits. The family is checked
/// because the promise is the weaker statement of the two: a device
/// may set it false and still time this particular family.
pub fn caps_verdict(period_ns: f32, valid_bits: u32) -> Result<(), String> {
    if valid_bits == 0 {
        return Err("the queue family writes no timestamp bits".into());
    }
    if !(period_ns > 0.0) || !period_ns.is_finite() {
        return Err(format!("timestampPeriod is {period_ns}, so ticks have no length"));
    }
    Ok(())
}

/// The mask of bits a timestamp of this device actually carries.
/// Everything above `timestampValidBits` is undefined and must be cut
/// away before two timestamps are subtracted.
pub fn mask_bits(valid_bits: u32) -> u64 {
    if valid_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    }
}

/// Ticks between two timestamps, counted inside the device's valid
/// width so a counter that wrapped between them still gives the true
/// (small) difference rather than a nonsense (enormous) one.
pub fn delta_ticks(begin: u64, end: u64, valid_bits: u32) -> u64 {
    let mask = mask_bits(valid_bits);
    (end & mask).wrapping_sub(begin & mask) & mask
}

/// Ticks to milliseconds through `timestampPeriod`, which the device
/// reports in nanoseconds per tick (1.0 on NVIDIA, ~38.4 on several
/// AMD parts, ~52.08 on some Intel — the reason no fixed constant
/// would do).
pub fn ticks_to_ms(ticks: u64, period_ns: f32) -> f64 {
    ticks as f64 * period_ns as f64 / 1.0e6
}

/// Which timestamps a frame of this shape writes.
///
/// A frame with glass runs every pass. A frame without one never
/// opens the offscreen pass and never draws the composite quad, so
/// slots 2 and 3 stay unwritten and the spans that need them are
/// dropped for that frame rather than counted as zero — an average
/// with zeros in it would quietly halve the cost of the blur.
pub fn frame_mask(has_glass: bool) -> u16 {
    let common = (1 << SLOT_FRAME_BEGIN)
        | (1 << SLOT_UPLOADS_END)
        | (1 << SLOT_MAIN_START)
        | (1 << SLOT_FRAME_END);
    if has_glass {
        common | (1 << SLOT_BASE_END) | (1 << SLOT_PYRAMID_END)
    } else {
        common
    }
}

/// Turns one frame's raw timestamps into milliseconds per span.
///
/// A slot is `None` when the frame never wrote it or the GPU has not
/// published it yet; a span survives only if both its endpoints are
/// there. `rest` is the frame minus the named parts, floored at zero
/// (the parts are adjacent subintervals of the frame, so the true
/// value cannot be negative — the floor only guards against a device
/// whose timestamps do not increase monotonically).
pub fn spans_ms(
    vals: &[Option<u64>; SLOTS],
    valid_bits: u32,
    period_ns: f32,
) -> [Option<f64>; MEASURES] {
    let mut out = [None; MEASURES];
    for (i, s) in SPANS.iter().enumerate() {
        if let (Some(b), Some(e)) = (vals[s.begin], vals[s.end]) {
            out[i] = Some(ticks_to_ms(delta_ticks(b, e, valid_bits), period_ns));
        }
    }
    if let Some(frame) = out[FRAME_SPAN] {
        let named: f64 = out[..FRAME_SPAN].iter().flatten().sum();
        out[REST_SPAN] = Some((frame - named).max(0.0));
    }
    out
}

/// Which ring of queries a frame writes, and when its results may be
/// read. See the module header for why the depth is frames in flight
/// and not swapchain images.
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    rings: u32,
    delay: u32,
}

impl Schedule {
    /// `in_flight` is how many frames the CPU may have submitted and
    /// not yet waited for. Zero is meaningless and reads as one.
    pub fn new(in_flight: u32) -> Self {
        let delay = in_flight.max(1);
        Schedule { rings: delay + 1, delay }
    }

    /// Rings of queries held. Always one more than the delay, so the
    /// frame being read is never the frame being reset.
    pub fn rings(&self) -> u32 {
        self.rings
    }

    /// Frames between writing a timestamp and reading it.
    pub fn delay(&self) -> u32 {
        self.delay
    }

    /// Total queries the pool needs.
    pub fn query_count(&self) -> u32 {
        self.rings() * SLOTS as u32
    }

    /// Which ring this frame writes into.
    pub fn slot(&self, frame: u64) -> usize {
        (frame % self.rings as u64) as usize
    }

    /// First query of a ring.
    pub fn first_query(&self, frame: u64) -> u32 {
        (self.slot(frame) * SLOTS) as u32
    }

    /// Whether a frame submitted as `submitted` is safe to read now
    /// that `current` frames have been submitted in total.
    pub fn ready(&self, submitted: u64, current: u64) -> bool {
        current >= submitted + self.delay as u64
    }
}

/// The last N samples of one measure, and the three numbers worth
/// printing about them. The worst is kept as well as the average
/// because a compositor is judged by its worst frame, not its mean
/// one.
#[derive(Clone, Debug)]
pub struct MeasureWindow {
    cap: usize,
    buf: VecDeque<f64>,
}

impl MeasureWindow {
    pub fn new(cap: usize) -> Self {
        MeasureWindow { cap: cap.max(1), buf: VecDeque::new() }
    }

    pub fn push(&mut self, ms: f64) {
        if self.buf.len() == self.cap {
            self.buf.pop_front();
        }
        self.buf.push_back(ms);
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn clear(&mut self) {
        self.buf.clear();
    }

    pub fn mean(&self) -> f64 {
        if self.buf.is_empty() {
            return 0.0;
        }
        self.buf.iter().sum::<f64>() / self.buf.len() as f64
    }

    /// The middle sample, or the average of the two middle ones when
    /// the count is even.
    pub fn median(&self) -> f64 {
        if self.buf.is_empty() {
            return 0.0;
        }
        let mut v: Vec<f64> = self.buf.iter().copied().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        }
    }

    pub fn worst(&self) -> f64 {
        self.buf.iter().copied().fold(0.0f64, f64::max)
    }
}

/// A frame whose queries are recorded but not yet read.
#[derive(Clone, Copy, Debug)]
struct Pending {
    frame: u64,
    first_query: u32,
    mask: u16,
}

/// The live instrument: a pool of timestamp queries, the schedule that
/// says when to read them, and the windows the report is drawn from.
pub struct GpuTiming {
    pool: vk::QueryPool,
    schedule: Schedule,
    period_ns: f32,
    valid_bits: u32,
    /// Frames between reports.
    interval: u32,
    /// Frames submitted so far; also the index of the next one.
    frame: u64,
    /// The frame being recorded right now.
    live_query: u32,
    live_mask: u16,
    pending: VecDeque<Pending>,
    windows: Vec<MeasureWindow>,
    collected: u32,
    /// Set when the driver answers a read with something other than
    /// success or "not ready": measuring stops, rendering does not.
    broken: bool,
}

impl GpuTiming {
    /// Reads [`ENV_VAR`] and builds the instrument, or returns `None`
    /// having allocated nothing. `in_flight` is the renderer's own
    /// frame depth — see the module header.
    ///
    /// # Safety
    /// `device` must be the live logical device the frames are
    /// recorded on, and it must outlive the returned value.
    pub unsafe fn from_env(
        device: &ash::Device,
        period_ns: f32,
        valid_bits: u32,
        compute_and_graphics: bool,
        in_flight: u32,
    ) -> Option<Self> {
        let raw = std::env::var_os(ENV_VAR)?;
        let interval = interval_from_env(raw.to_str())?;
        Self::with_interval(device, period_ns, valid_bits, compute_and_graphics, in_flight, interval)
    }

    /// The half of [`GpuTiming::from_env`] that has already decided to
    /// measure. Split out so a caller with its own switch — a test, a
    /// future settings page — does not have to go through the
    /// environment.
    ///
    /// # Safety
    /// As [`GpuTiming::from_env`].
    pub unsafe fn with_interval(
        device: &ash::Device,
        period_ns: f32,
        valid_bits: u32,
        compute_and_graphics: bool,
        in_flight: u32,
        interval: u32,
    ) -> Option<Self> {
        if let Err(why) = caps_verdict(period_ns, valid_bits) {
            eprintln!("nacelle-gpu: {ENV_VAR} is set, but {why} — no timing on this device");
            return None;
        }
        let schedule = Schedule::new(in_flight);
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(schedule.query_count());
        let pool = match device.create_query_pool(&info, None) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("nacelle-gpu: cannot create the timestamp pool ({e:?}) — no timing");
                return None;
            }
        };
        let interval = interval.max(1);
        eprintln!(
            "nacelle-gpu: timing on — {period_ns} ns/tick, {valid_bits} valid bits, \
             timestampComputeAndGraphics {compute_and_graphics}, \
             read {} frame(s) late, report every {interval} frames",
            schedule.delay()
        );
        Some(GpuTiming {
            pool,
            schedule,
            period_ns,
            valid_bits,
            interval,
            frame: 0,
            live_query: 0,
            live_mask: 0,
            pending: VecDeque::new(),
            windows: (0..MEASURES).map(|_| MeasureWindow::new(interval as usize)).collect(),
            collected: 0,
            broken: false,
        })
    }

    /// Opens a frame: claims a ring, resets its queries and writes the
    /// first timestamp. `has_glass` decides which of the six the frame
    /// will write, and the mask it produces is the single source of
    /// truth from here on — [`GpuTiming::mark`] refuses any slot the
    /// mask does not name, so a recording site and its bookkeeping
    /// cannot drift apart.
    ///
    /// Must be called after `begin_command_buffer` and outside any
    /// render pass, because `vkCmdResetQueryPool` may not appear
    /// inside one.
    ///
    /// # Safety
    /// `cmd` must be recording, on `device`.
    pub unsafe fn begin_frame(&mut self, device: &ash::Device, cmd: vk::CommandBuffer, has_glass: bool) {
        if self.broken {
            return;
        }
        self.live_query = self.schedule.first_query(self.frame);
        self.live_mask = frame_mask(has_glass);
        device.cmd_reset_query_pool(cmd, self.pool, self.live_query, SLOTS as u32);
        self.mark(device, cmd, SLOT_FRAME_BEGIN);
    }

    /// Writes one of the frame's timestamps. The frame's first is
    /// taken at `TOP_OF_PIPE` (when the GPU reaches the command
    /// buffer); every other at `BOTTOM_OF_PIPE` (when everything
    /// recorded before it has finished), which is what makes the
    /// differences add up to the frame.
    ///
    /// # Safety
    /// `cmd` must be recording, on `device`, inside the frame opened
    /// by [`GpuTiming::begin_frame`].
    pub unsafe fn mark(&self, device: &ash::Device, cmd: vk::CommandBuffer, slot: usize) {
        if self.broken {
            return;
        }
        debug_assert!(slot < SLOTS, "timestamp slot {slot} is out of range");
        if (self.live_mask >> slot) & 1 == 0 {
            return;
        }
        let stage = if slot == SLOT_FRAME_BEGIN {
            vk::PipelineStageFlags::TOP_OF_PIPE
        } else {
            vk::PipelineStageFlags::BOTTOM_OF_PIPE
        };
        device.cmd_write_timestamp(cmd, stage, self.pool, self.live_query + slot as u32);
    }

    /// Closes a frame once it has been submitted: its queries are now
    /// in flight, and the frame counter moves on.
    pub fn end_frame(&mut self) {
        if self.broken {
            return;
        }
        self.pending.push_back(Pending {
            frame: self.frame,
            first_query: self.live_query,
            mask: self.live_mask,
        });
        self.frame += 1;
        self.live_mask = 0;
    }

    /// Reads every frame old enough to be safe, feeds the windows, and
    /// prints when the interval is up. Call it at the top of a frame,
    /// after the fence wait and before the next `begin_frame`: that is
    /// the one moment at which the previous frame is provably complete
    /// and its queries have not yet been reset.
    ///
    /// # Safety
    /// `device` must be the device the pool was created on.
    pub unsafe fn collect(&mut self, device: &ash::Device, w: u32, h: u32) {
        if self.broken {
            return;
        }
        while let Some(&p) = self.pending.front() {
            if !self.schedule.ready(p.frame, self.frame) {
                break;
            }
            self.pending.pop_front();
            match self.read_frame(device, p) {
                Ok(vals) => {
                    let ms = spans_ms(&vals, self.valid_bits, self.period_ns);
                    for (win, v) in self.windows.iter_mut().zip(ms.iter()) {
                        if let Some(v) = v {
                            win.push(*v);
                        }
                    }
                    self.collected += 1;
                }
                Err(e) => {
                    eprintln!("nacelle-gpu: reading timestamps failed ({e:?}) — timing off");
                    self.broken = true;
                    return;
                }
            }
        }
        if self.collected >= self.interval {
            self.report(w, h);
            self.collected = 0;
            for win in &mut self.windows {
                win.clear();
            }
        }
    }

    /// One frame's six timestamps, `None` where the frame did not
    /// write the query or the GPU has not published it. Never waits:
    /// `VK_NOT_READY` costs a sample, never a stall.
    unsafe fn read_frame(
        &self,
        device: &ash::Device,
        p: Pending,
    ) -> Result<[Option<u64>; SLOTS], vk::Result> {
        // Two words per query: the value, then the availability the
        // WITH_AVAILABILITY flag appends. Zeroed first, because the
        // driver leaves the value of an unavailable query untouched.
        let mut raw = [[0u64; 2]; SLOTS];
        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY;
        match device.get_query_pool_results(self.pool, p.first_query, &mut raw, flags) {
            Ok(()) => {}
            // Expected, not exceptional: a query this frame never wrote
            // is never available, and a mixed answer is exactly what a
            // frame without glass produces.
            Err(vk::Result::NOT_READY) => {}
            Err(e) => return Err(e),
        }
        let mut vals = [None; SLOTS];
        for (slot, entry) in raw.iter().enumerate() {
            let recorded = (p.mask >> slot) & 1 == 1;
            if recorded && entry[1] != 0 {
                vals[slot] = Some(entry[0]);
            }
        }
        Ok(vals)
    }

    /// Prints the window to stderr. The surface size is in the header
    /// because a multi-monitor session runs one renderer per screen,
    /// and two anonymous tables would be worse than none.
    pub fn report(&self, w: u32, h: u32) {
        let frames = self.windows[FRAME_SPAN].len();
        if frames == 0 {
            return;
        }
        eprintln!(
            "nacelle-gpu {w}x{h}: {frames} frames, milliseconds  {:>7} {:>8} {:>8}",
            "avg", "median", "worst"
        );
        for (i, win) in self.windows.iter().enumerate() {
            if win.is_empty() {
                continue;
            }
            let name = if i == REST_SPAN { "rest" } else { SPANS[i].name };
            eprintln!(
                "  {name:<10} n={:<5} {:>7.3} {:>8.3} {:>8.3}",
                win.len(),
                win.mean(),
                win.median(),
                win.worst()
            );
        }
    }

    /// The last word on the way out, so a short run — a screenshot, a
    /// crash reproduction — still prints something.
    pub fn report_final(&self, w: u32, h: u32) {
        if !self.broken {
            self.report(w, h);
        }
    }

    /// # Safety
    /// The device must be idle and must still be the one the pool was
    /// created on.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_query_pool(self.pool, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The switch has to be readable by someone who has not read the
    /// source: absent and the off-words mean off, everything else
    /// means on, and a plain number is also the window.
    #[test]
    fn the_switch_reads_the_way_it_is_documented() {
        assert_eq!(interval_from_env(None), None);
        for off in ["", "0", "off", "OFF", "no", "false", "None", " 0 "] {
            assert_eq!(interval_from_env(Some(off)), None, "{off:?} must be off");
        }
        assert_eq!(interval_from_env(Some("1")), Some(DEFAULT_INTERVAL));
        assert_eq!(interval_from_env(Some("on")), Some(DEFAULT_INTERVAL));
        assert_eq!(interval_from_env(Some("yes")), Some(DEFAULT_INTERVAL));
        assert_eq!(interval_from_env(Some("600")), Some(600));
        assert_eq!(interval_from_env(Some(" 30 ")), Some(30));
        // A typo measures with the default rather than measuring
        // nothing: a switch that is set must never be silently ignored.
        assert_eq!(interval_from_env(Some("tak")), Some(DEFAULT_INTERVAL));
    }

    /// A device that cannot time says so, and the reason is a sentence
    /// the log can print.
    #[test]
    fn an_untimeable_device_is_refused_with_a_reason() {
        assert!(caps_verdict(1.0, 64).is_ok());
        assert!(caps_verdict(38.4615, 36).is_ok());
        let no_bits = caps_verdict(1.0, 0).unwrap_err();
        assert!(no_bits.contains("no timestamp bits"), "{no_bits}");
        assert!(caps_verdict(0.0, 64).is_err());
        assert!(caps_verdict(-1.0, 64).is_err());
        assert!(caps_verdict(f32::NAN, 64).is_err());
    }

    /// Everything above `timestampValidBits` is undefined, so the
    /// subtraction has to happen inside that width — and a counter
    /// that wrapped between two timestamps must still give the small
    /// true difference, not a number the size of the counter.
    #[test]
    fn a_wrapped_counter_still_gives_the_short_difference() {
        assert_eq!(mask_bits(64), u64::MAX);
        assert_eq!(mask_bits(32), 0xFFFF_FFFF);
        assert_eq!(mask_bits(36), (1u64 << 36) - 1);

        // 32-bit counter, five ticks before the wrap to five after.
        let begin = 0xFFFF_FFFBu64;
        let end = 0x0000_0004u64;
        assert_eq!(delta_ticks(begin, end, 32), 9);
        // The same pair read as full width is the enormous answer the
        // masking exists to prevent.
        assert!(delta_ticks(begin, end, 64) > 1 << 40);
        // Rubbish in the undefined bits changes nothing.
        assert_eq!(delta_ticks(begin | (1 << 40), end, 32), 9);
        // The ordinary case is ordinary.
        assert_eq!(delta_ticks(1_000, 1_700, 64), 700);
    }

    /// The period is the whole reason no constant would do: the same
    /// tick count is a different millisecond on each vendor.
    #[test]
    fn ticks_become_milliseconds_through_the_devices_period() {
        // NVIDIA: one nanosecond a tick.
        assert!((ticks_to_ms(1_000_000, 1.0) - 1.0).abs() < 1e-9);
        // An AMD part: 38.4615 ns a tick, so 26 000 ticks is a
        // millisecond.
        assert!((ticks_to_ms(26_000, 38.4615) - 1.0).abs() < 1e-3);
        assert_eq!(ticks_to_ms(0, 38.4615), 0.0);
    }

    /// The frame being written must never be the frame being read —
    /// that is the whole job of keeping one ring more than the delay.
    #[test]
    fn the_ring_read_is_never_the_ring_being_overwritten() {
        for in_flight in 1..=4u32 {
            let s = Schedule::new(in_flight);
            assert_eq!(s.rings(), s.delay() + 1);
            assert_eq!(s.query_count(), s.rings() * SLOTS as u32);
            for frame in (s.delay() as u64)..1_000 {
                let read = frame - s.delay() as u64;
                assert_ne!(
                    s.slot(frame),
                    s.slot(read),
                    "in_flight {in_flight}: frame {frame} would reset the ring it is reading"
                );
                assert_eq!(s.first_query(frame), (s.slot(frame) * SLOTS) as u32);
            }
        }
    }

    /// The read waits exactly the declared number of frames: not one
    /// fewer (that reads a frame the GPU may still be inside) and the
    /// depth of this renderer — one command buffer, one fence — is
    /// one.
    #[test]
    fn a_frame_becomes_readable_exactly_one_delay_later() {
        let s = Schedule::new(1);
        assert_eq!(s.delay(), 1);
        assert!(!s.ready(7, 7), "the frame just submitted is not readable");
        assert!(s.ready(7, 8), "the next frame's fence wait publishes it");
        assert!(s.ready(7, 40));

        let deep = Schedule::new(3);
        assert!(!deep.ready(10, 12));
        assert!(deep.ready(10, 13));

        // Zero is meaningless and must not become "read it now".
        assert_eq!(Schedule::new(0).delay(), 1);
    }

    /// The mask is the contract between the recording sites and the
    /// reader: a frame without glass opens no offscreen pass, so those
    /// two timestamps are not written and must not be claimed.
    #[test]
    fn the_mask_names_the_passes_a_frame_of_that_shape_runs() {
        let glass = frame_mask(true);
        for slot in 0..SLOTS {
            assert_eq!((glass >> slot) & 1, 1, "a glass frame writes slot {slot}");
        }
        let plain = frame_mask(false);
        for slot in [SLOT_FRAME_BEGIN, SLOT_UPLOADS_END, SLOT_MAIN_START, SLOT_FRAME_END] {
            assert_eq!((plain >> slot) & 1, 1, "every frame writes slot {slot}");
        }
        assert_eq!((plain >> SLOT_BASE_END) & 1, 0);
        assert_eq!((plain >> SLOT_PYRAMID_END) & 1, 0);
    }

    /// On a frame with glass every span is present and they add up to
    /// the frame exactly — the property that lets a reader trust the
    /// breakdown at all.
    #[test]
    fn a_glass_frame_breaks_down_into_parts_that_sum_to_the_whole() {
        // Ticks at 1 ns: uploads 0.1 ms, base 4 ms, pyramid 1.5 ms,
        // composite 0.4 ms, main 2 ms.
        let vals = [
            Some(1_000_000u64),
            Some(1_100_000),
            Some(5_100_000),
            Some(6_600_000),
            Some(7_000_000),
            Some(9_000_000),
        ];
        let ms = spans_ms(&vals, 64, 1.0);
        let names: Vec<&str> = SPANS.iter().map(|s| s.name).collect();
        assert_eq!(names, ["uploads", "base", "pyramid", "composite", "main", "frame"]);
        let got: Vec<f64> = ms[..=FRAME_SPAN].iter().map(|v| v.unwrap()).collect();
        let want = [0.1, 4.0, 1.5, 0.4, 2.0, 8.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-9, "{got:?} != {want:?}");
        }
        let parts: f64 = ms[..FRAME_SPAN].iter().flatten().sum();
        assert!((parts - ms[FRAME_SPAN].unwrap()).abs() < 1e-9);
        assert!(ms[REST_SPAN].unwrap().abs() < 1e-9, "nothing is left over");
    }

    /// On a frame without glass the three glass spans are absent —
    /// absent, not zero — and what the swapchain pass spent on its own
    /// setup shows up as the leftover.
    #[test]
    fn a_plain_frame_reports_no_blur_at_all_rather_than_a_blur_of_zero() {
        let vals = [
            Some(0u64),
            Some(100_000),  // uploads 0.1 ms
            None,           // no base pass
            None,           // no pyramid
            Some(300_000),  // 0.2 ms of render-pass begin, unbracketed
            Some(2_300_000) // main 2.0 ms
        ];
        let ms = spans_ms(&vals, 64, 1.0);
        assert!(ms[1].is_none(), "base must be absent");
        assert!(ms[2].is_none(), "pyramid must be absent");
        assert!(ms[3].is_none(), "composite must be absent");
        assert!((ms[0].unwrap() - 0.1).abs() < 1e-9);
        assert!((ms[4].unwrap() - 2.0).abs() < 1e-9);
        assert!((ms[FRAME_SPAN].unwrap() - 2.3).abs() < 1e-9);
        assert!((ms[REST_SPAN].unwrap() - 0.2).abs() < 1e-9, "the gap is the leftover");
    }

    /// A frame the GPU has not finished publishing yields nothing
    /// rather than a difference against a zero.
    #[test]
    fn an_unpublished_endpoint_drops_its_span_and_takes_the_frame_with_it() {
        let vals = [Some(10u64), Some(20), None, None, Some(40), None];
        let ms = spans_ms(&vals, 64, 1.0);
        assert!(ms[0].is_some(), "uploads has both its endpoints");
        assert!(ms[4].is_none(), "main lost its end");
        assert!(ms[FRAME_SPAN].is_none(), "the frame lost its end");
        assert!(ms[REST_SPAN].is_none(), "no frame, no leftover");
    }

    /// Mean, median and worst, on an even count and an odd one.
    #[test]
    fn the_window_reports_the_three_numbers_correctly() {
        let mut w = MeasureWindow::new(8);
        assert!(w.is_empty());
        assert_eq!(w.mean(), 0.0);
        assert_eq!(w.median(), 0.0);
        assert_eq!(w.worst(), 0.0);
        for v in [3.0, 1.0, 2.0] {
            w.push(v);
        }
        assert_eq!(w.len(), 3);
        assert!((w.mean() - 2.0).abs() < 1e-9);
        assert!((w.median() - 2.0).abs() < 1e-9);
        assert!((w.worst() - 3.0).abs() < 1e-9);
        w.push(10.0);
        assert!((w.median() - 2.5).abs() < 1e-9, "even count averages the middle pair");
        assert!((w.mean() - 4.0).abs() < 1e-9);
        assert!((w.worst() - 10.0).abs() < 1e-9);
    }

    /// The window is the report's window: older samples fall out, so
    /// "worst of the last N" means the last N and not all time.
    #[test]
    fn the_window_forgets_everything_older_than_its_capacity() {
        let mut w = MeasureWindow::new(3);
        for v in [99.0, 1.0, 2.0, 3.0] {
            w.push(v);
        }
        assert_eq!(w.len(), 3);
        assert!((w.worst() - 3.0).abs() < 1e-9, "the spike aged out");
        assert!((w.mean() - 2.0).abs() < 1e-9);
        w.clear();
        assert!(w.is_empty());
        // A nonsense capacity must still hold something.
        let mut tiny = MeasureWindow::new(0);
        tiny.push(5.0);
        assert_eq!(tiny.len(), 1);
    }

    /// The mask and the span table have to agree: feed a frame
    /// exactly the timestamps its shape writes, and the measures that
    /// come out must be the ones that shape can honestly report.
    #[test]
    fn what_a_frame_writes_decides_what_it_can_report() {
        for has_glass in [true, false] {
            let mask = frame_mask(has_glass);
            // A tick per slot, ascending, only where the mask says the
            // frame wrote one.
            let mut vals = [None; SLOTS];
            for (slot, v) in vals.iter_mut().enumerate() {
                if (mask >> slot) & 1 == 1 {
                    *v = Some(1_000_000u64 * (slot as u64 + 1));
                }
            }
            let ms = spans_ms(&vals, 64, 1.0);
            let named: Vec<&str> = ms
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_some())
                .map(|(i, _)| if i == REST_SPAN { "rest" } else { SPANS[i].name })
                .collect();
            if has_glass {
                assert_eq!(
                    named,
                    ["uploads", "base", "pyramid", "composite", "main", "frame", "rest"]
                );
                assert!(ms[REST_SPAN].unwrap().abs() < 1e-9, "glass leaves nothing over");
            } else {
                assert_eq!(named, ["uploads", "main", "frame", "rest"]);
                // The frame is five milliseconds and the two named
                // spans are one each, so the three between slots 1
                // and 4 belong to nobody but the leftover.
                assert!((ms[REST_SPAN].unwrap() - 3.0).abs() < 1e-9, "{ms:?}");
            }
        }
    }

    /// The whole instrument against a real driver: a pool, a reset, six
    /// timestamps in one command buffer, a fence, and a read one frame
    /// later. Everything above this test is arithmetic that needs no
    /// GPU; this is the part that needs one, so it stays out of the
    /// gate and is run by hand:
    ///
    /// ```text
    ///   cargo test -- --ignored --nocapture
    /// ```
    ///
    /// It builds its own instance and device — no surface, no
    /// swapchain, nothing to do with [`crate::gfx`] — because the
    /// question it answers is only whether the queries come back.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn on_a_real_device_the_queries_come_back() {
        unsafe {
            let entry = ash::Entry::load().expect("no Vulkan loader");
            let app = std::ffi::CStr::from_bytes_with_nul(b"nacelle-gpu-probe\0").unwrap();
            let app_info = vk::ApplicationInfo::default()
                .application_name(app)
                .api_version(vk::make_api_version(0, 1, 0, 0));
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app_info), None)
                .expect("no Vulkan instance");
            let (pdevice, family) = instance
                .enumerate_physical_devices()
                .expect("no Vulkan devices")
                .iter()
                .find_map(|&pd| {
                    instance
                        .get_physical_device_queue_family_properties(pd)
                        .iter()
                        .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                        .map(|i| (pd, i as u32))
                })
                .expect("no graphics queue");
            let limits = instance.get_physical_device_properties(pdevice).limits;
            let valid_bits = instance.get_physical_device_queue_family_properties(pdevice)
                [family as usize]
                .timestamp_valid_bits;
            let priorities = [1.0f32];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities)];
            let device = instance
                .create_device(
                    pdevice,
                    &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
                    None,
                )
                .expect("no logical device");
            let queue = device.get_device_queue(family, 0);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let cmd = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[0];
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();

            // Something for the GPU to actually do between two of the
            // timestamps, so the probe can tell "the queries answer"
            // from "the queries answer with the truth": sixty-four
            // mebibytes of fill has to cost more than zero.
            const WORK: u64 = 64 * 1024 * 1024;
            let buf = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(WORK)
                        .usage(vk::BufferUsageFlags::TRANSFER_DST)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap();
            let req = device.get_buffer_memory_requirements(buf);
            let props = instance.get_physical_device_memory_properties(pdevice);
            let mem_type = (0..props.memory_type_count)
                .find(|&i| {
                    req.memory_type_bits & (1 << i) != 0
                        && props.memory_types[i as usize]
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .expect("no device-local memory");
            let mem = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(mem_type),
                    None,
                )
                .unwrap();
            device.bind_buffer_memory(buf, mem, 0).unwrap();

            let mut t = GpuTiming::with_interval(
                &device,
                limits.timestamp_period,
                valid_bits,
                limits.timestamp_compute_and_graphics == vk::TRUE,
                1,
                // A window wide enough that nothing is reported (and
                // so cleared) while the samples are being counted.
                1_000,
            )
            .expect("this device cannot be timed");

            // Two frames, because one frame proves nothing about a read
            // that is deliberately one frame late.
            for frame in 0..2 {
                device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
                device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .unwrap();
                t.begin_frame(&device, cmd, true);
                t.mark(&device, cmd, SLOT_UPLOADS_END);
                device.cmd_fill_buffer(cmd, buf, 0, WORK, 0xA5A5_A5A5);
                t.mark(&device, cmd, SLOT_BASE_END);
                for slot in [SLOT_PYRAMID_END, SLOT_MAIN_START, SLOT_FRAME_END] {
                    t.mark(&device, cmd, slot);
                }
                device.end_command_buffer(cmd).unwrap();
                let cmds = [cmd];
                device.reset_fences(&[fence]).unwrap();
                device
                    .queue_submit(
                        queue,
                        &[vk::SubmitInfo::default().command_buffers(&cmds)],
                        fence,
                    )
                    .unwrap();
                t.end_frame();
                device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
                // The renderer collects at the top of the next frame,
                // right after its fence wait; so does this.
                t.collect(&device, 1, frame + 1);
            }

            // A frame of nothing still has a frame span: the read came
            // back, the availability bits were set, the arithmetic ran.
            assert!(
                !t.windows[FRAME_SPAN].is_empty(),
                "not one frame's timestamps came back"
            );
            assert_eq!(t.collected as usize, t.windows[FRAME_SPAN].len());
            // And the number moves with the work: the span that holds
            // the fill is not zero, and the frame is at least as long
            // as it.
            let filled = t.windows[1].worst();
            assert!(filled > 0.0, "a 64 MiB fill measured as {filled} ms");
            assert!(t.windows[FRAME_SPAN].worst() >= filled);
            t.report(0, 0);

            let _ = device.device_wait_idle();
            t.destroy(&device);
            device.destroy_buffer(buf, None);
            device.free_memory(mem, None);
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// The span table has to stay a partition of the frame: adjacent,
    /// gapless, and covered by the whole. Everything the report says
    /// about leftovers rests on it.
    #[test]
    fn the_span_table_is_a_partition_of_the_frame() {
        let parts = &SPANS[..FRAME_SPAN];
        assert_eq!(parts[0].begin, SLOT_FRAME_BEGIN);
        assert_eq!(parts[parts.len() - 1].end, SLOT_FRAME_END);
        for pair in parts.windows(2) {
            assert_eq!(pair[0].end, pair[1].begin, "a gap between spans");
        }
        assert_eq!(SPANS[FRAME_SPAN].begin, SLOT_FRAME_BEGIN);
        assert_eq!(SPANS[FRAME_SPAN].end, SLOT_FRAME_END);
        assert_eq!(MEASURES, SPANS.len() + 1);
        assert_eq!(SLOTS, SLOT_FRAME_END + 1);
    }
}
