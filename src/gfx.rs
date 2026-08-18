//! Vulkan renderer (ash) — SEVEN pipelines over one vertex stream: the
//! atlas one (text and solid shapes, R8 coverage), the image one
//! (application-registered RGBA textures), the frosted-glass one (the
//! scene beneath, pre-rendered and blurred, sampled at one of three
//! pyramid ranks), the additive one (fs_main again under
//! SRC_ALPHA/ONE, so glow adds light instead of filming over it) and
//! the shape one (the vector core: an analytic distance field over
//! set 2's records, one quad per silhouette) with its FROSTED sibling
//! (fs_shape_glass — the one fragment that binds a pyramid target and
//! reads a record) and its ADDITIVE twin, which is the same fragment
//! under the same blend the atlas glow uses — because a glow adds
//! light and that is a property of the pipeline, not of the record.
//! The draw list's runs say which is which, in emission order, each
//! run under its own scissor.
//!
//! The count above said "six" until the frosted sibling was counted;
//! [`Pipe::N`] is the number that decides, and `the_advertised_pipeline_count_is_the_one_the_device_gets`
//! is what keeps the prose from drifting off it again.

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::collections::HashMap;
use std::ffi::CStr;

use nacelle::draw::{DrawRun, ImageId, Shape, Vertex, NO_SHAPE};
use nacelle::font::{ATLAS_H, ATLAS_W};

use crate::timing::{
    GpuTiming, SLOT_BASE_END, SLOT_FRAME_END, SLOT_MAIN_START, SLOT_PYRAMID_END, SLOT_UPLOADS_END,
};

const MAX_VERTS: usize = 400_000;
/// Shape records per frame: 16 384 × 80 B = 1.31 MB, host-visible and
/// persistently mapped like the vertex buffer. Overflow degrades the
/// MAX_VERTS way — records past the limit are clipped, the frame is
/// never lost (f3 2.5, R5).
///
/// The record was 64 B when §2.5 sized it and grew a `tint` with K3b
/// (§3.3). The allocation below has always read `size_of::<Shape>()`,
/// so nothing moved but this sentence; it is written out because a
/// budget nobody can check against the code is a budget that drifts.
const MAX_SHAPES: usize = 16_384;

/// How many frames the CPU may be ahead of the GPU: one command
/// buffer, one fence, one semaphore pair, and [`Gfx::render`] waits on
/// that fence before it touches any of them. Timing reads its
/// timestamps exactly this many frames late — see [`crate::timing`]
/// for why this number, and not the swapchain's image count, is the
/// one that matters. Grow the buffers and this constant grows with
/// them.
const FRAMES_IN_FLIGHT: u32 = 1;

/// One target of the frosted-glass chain: the base scene at full size,
/// then the shrinking pyramid whose repeated linear resampling IS the
/// blur. Each is drawable (a framebuffer) and samplable (a set).
struct BlurTarget {
    image: vk::Image,
    mem: vk::DeviceMemory,
    view: vk::ImageView,
    fb: vk::Framebuffer,
    desc_set: vk::DescriptorSet,
    w: u32,
    h: u32,
}

/// An application-registered RGBA image the shader can sample. Pixels
/// arrive through [`Gfx::update_texture`] and are copied to the GPU at
/// the start of the next frame.
struct Texture {
    image: vk::Image,
    mem: vk::DeviceMemory,
    view: vk::ImageView,
    desc_set: vk::DescriptorSet,
    w: u32,
    h: u32,
    /// Pixels waiting for the next frame's command buffer.
    pending: Option<Vec<u8>>,
    /// Whether the image has ever been filled — decides the first
    /// barrier's source layout.
    initialized: bool,
}

/// Why the renderer could not be built.
///
/// Every one of these is a MACHINE that cannot draw, not a program that
/// is wrong: no Vulkan library installed, no GPU that can both draw and
/// present to this window, a driver that refused an object. The answer
/// to all of them is the same — say so and leave — and a caller cannot
/// say anything at all about a panic that already unwound past it.
///
/// The line between this type and the panics still standing in this
/// file is exactly that: a panic is reserved for a program that asked
/// for something impossible, an error for a machine that cannot give
/// something possible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GfxError {
    /// The window handed out no raw handle to build a surface from.
    NoWindowHandle,
    /// The Vulkan loader could not open the library — the usual
    /// meaning is that no driver is installed.
    NoVulkanLibrary,
    /// This Vulkan has no surface extension for the display server the
    /// process is running under.
    NoSurfaceExtension,
    /// The instance itself was refused.
    NoInstance(vk::Result),
    /// A surface could not be made for this window.
    NoSurface(vk::Result),
    /// No physical device can both draw and present here.
    NoDevice,
    /// The logical device was refused.
    NoLogicalDevice(vk::Result),
    /// The surface answered nothing when asked for its formats.
    NoFormat(vk::Result),
    /// Either no memory type on this device satisfies what the named
    /// object needs, or the allocation was refused.
    NoMemory(&'static str),
}

impl std::fmt::Display for GfxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GfxError::NoWindowHandle => write!(f, "the window has no raw handle to draw into"),
            GfxError::NoVulkanLibrary => {
                write!(f, "no Vulkan library on this machine (is a driver installed?)")
            }
            GfxError::NoSurfaceExtension => {
                write!(f, "this Vulkan has no surface extension for this display server")
            }
            GfxError::NoInstance(e) => write!(f, "Vulkan refused an instance: {e:?}"),
            GfxError::NoSurface(e) => write!(f, "cannot create a surface for this window: {e:?}"),
            GfxError::NoDevice => write!(f, "no GPU here can both draw and present to this window"),
            GfxError::NoLogicalDevice(e) => write!(f, "the GPU refused a logical device: {e:?}"),
            GfxError::NoFormat(e) => write!(f, "the surface named no format: {e:?}"),
            GfxError::NoMemory(what) => write!(f, "no GPU memory for the {what}"),
        }
    }
}

impl std::error::Error for GfxError {}

/// What a swapchain call's error means for the next frame.
///
/// The whole point of naming these is that only the last one is worth
/// ending a process over. A swapchain that no longer matches its
/// surface is the ordinary consequence of a resize; a surface that is
/// GONE is the ordinary consequence of unplugging a monitor or
/// restarting a compositor — and neither is a reason for the desktop
/// to disappear along with it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fault {
    /// The swapchain no longer matches its surface: build another one
    /// and draw the frame again.
    Swapchain,
    /// The surface itself is gone: build a new surface for the same
    /// window, and a swapchain on top of that.
    Surface,
    /// Nothing this renderer knows how to repair.
    Fatal,
}

/// Reads a Vulkan result as a [`Fault`].
///
/// `SUBOPTIMAL_KHR` is here even though both call sites usually meet it
/// as a success value (`acquire_next_image` returns it in the tuple,
/// `queue_present` as `Ok(true)`): a driver is allowed to hand it back
/// as an error from either, and a renderer that then called it fatal
/// would die of a picture that was merely stretched.
fn fault_of(e: vk::Result) -> Fault {
    match e {
        vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => Fault::Swapchain,
        vk::Result::ERROR_SURFACE_LOST_KHR => Fault::Surface,
        _ => Fault::Fatal,
    }
}

impl Fault {
    /// Whether the surface has to be built anew before anything else.
    fn rebuilds_surface(self) -> bool {
        matches!(self, Fault::Surface)
    }

    /// Whether the swapchain has to be built anew. A lost surface says
    /// yes to BOTH questions, and in that order: the swapchain hangs
    /// off the surface, so one built against a dead surface is dead
    /// too.
    fn rebuilds_swapchain(self) -> bool {
        matches!(self, Fault::Surface | Fault::Swapchain)
    }

    /// Whether this renderer has any repair for it.
    fn is_fatal(self) -> bool {
        matches!(self, Fault::Fatal)
    }
}

/// Everything Vulkan keeps at the *process* level: the loaded library,
/// the one instance, and the instance-level surface entry points. One
/// per process, never one per screen.
struct Vk {
    entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
}

/// The one instance the whole process shares.
///
/// WHY this exists at all: every [`Gfx`] used to create its own
/// `VkInstance`, and multi-monitor means one `Gfx` per screen. Closing
/// the first screen ran that screen's `Drop`, and the second `Gfx` then
/// jumped through a zeroed dispatch table — a call through 0x0 inside
/// libnvidia-glcore.so, reached from `destroy_swapchain`, and a SIGSEGV
/// on every exit.
///
/// WHAT WAS MEASURED, and only this: under gdb, destroying exactly one
/// `Gfx` and leaking the rest ends the process cleanly; letting a second
/// one drop crashes. So the fault is in tearing down the SECOND set of
/// per-screen Vulkan state, not in teardown as such.
///
/// WHICH part of that teardown pulls the rug is NOT isolated. Two
/// candidates, and this static removes both, which is why it was not
/// worth separating them:
///   - `destroy_instance` on the first screen taking driver-global
///     state with it;
///   - `Entry` itself. In ash 0.38 an `Entry` owns an `Arc<Library>` and
///     every `Entry::load` is a fresh `dlopen` with its own count, so the
///     first `Gfx` to drop ran `dlclose` on the driver while the second
///     was still calling into it. That fits a jump to 0x0 more exactly
///     than a driver deinit hook does.
///
/// Sharing costs nothing: between construction and teardown neither the
/// entry nor the instance is touched. `surface_loader` and `pdevice` are
/// read when a swapchain is rebuilt (depth changes, resizes), not per
/// frame, and both are handles rather than the instance itself.
///
/// WHY it is never destroyed, deliberately: Rust does not drop statics
/// at process exit, and that is precisely the property wanted here. If
/// the instance died with the last screen, the last screen would again
/// be pulling the driver out from under any screen still finishing its
/// own teardown. Leaving it alive means every `destroy_device` and
/// `destroy_surface` runs against an instance that outlives it. The
/// memory is handed back by the kernel when the process ends, which is
/// the only moment the instance would have been freed anyway.
static VK: std::sync::OnceLock<Vk> = std::sync::OnceLock::new();

/// Loads the library and creates the process's instance on first call,
/// hands out the same one afterwards.
///
/// `display_handle` decides which surface extensions get enabled
/// (`VK_KHR_xlib_surface` versus `VK_KHR_wayland_surface` and so on),
/// so the first window to ask settles the list for every later one.
/// That is sound as long as the process has a single winit event loop
/// — winit enforces exactly that, so every window in a process reports
/// the same *kind* of display handle.
///
/// # Safety
/// Calls into the Vulkan loader; `display_handle` must be a live handle
/// from the window system this process is actually running under.
unsafe fn shared(display_handle: RawDisplayHandle) -> Result<&'static Vk, GfxError> {
    // The build is OUTSIDE `get_or_init`, because a machine with no
    // Vulkan at all has to be able to say so: the closure form can only
    // return a `Vk`, so its three failure points had nowhere to go but
    // a panic. Publishing afterwards keeps the "one instance, never
    // destroyed" property that [`VK`] exists for.
    if let Some(vk) = VK.get() {
        return Ok(vk);
    }
    let entry = ash::Entry::load().map_err(|_| GfxError::NoVulkanLibrary)?;

    let app_name = CStr::from_bytes_with_nul(b"nacelle-desktop\0").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .engine_name(app_name)
        .api_version(vk::make_api_version(0, 1, 0, 0));

    let ext_names = ash_window::enumerate_required_extensions(display_handle)
        .map_err(|_| GfxError::NoSurfaceExtension)?
        .to_vec();

    let instance_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&ext_names);
    let instance = entry
        .create_instance(&instance_info, None)
        .map_err(GfxError::NoInstance)?;

    let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

    // If some other thread published first, the one built here is
    // dropped — which costs an instance handle and nothing more, since
    // `Vk` has no `Drop` and ash never destroys an instance by itself.
    // That leak is the same deliberate one [`VK`] is documented for.
    Ok(VK.get_or_init(move || Vk {
        entry,
        instance,
        surface_loader,
    }))
}

pub struct Gfx {
    /// A clone of the process-wide surface loader: a handful of
    /// function pointers plus the instance handle, nothing owned. It is
    /// held per `Gfx` because the *surface* below is per window, and
    /// destroying it in `Drop` needs the loader right here.
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    /// The handles the surface was built from, kept because a surface
    /// can be LOST — a monitor unplugged, a compositor restarted — and
    /// the only way back is another surface for the same window.
    ///
    /// Keeping them is sound for the same reason the window can be
    /// borrowed at all: raw handles are plain `Copy` values that borrow
    /// nothing, and they stay meaningful exactly as long as the
    /// caller's window does. `Gfx` never outlives that window — it is
    /// the thing drawing into it.
    display_handle: RawDisplayHandle,
    window_handle: RawWindowHandle,
    pdevice: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline_layout: vk::PipelineLayout,
    /// The graphics pipelines, indexed by [`Pipe`] — an ARRAY and not
    /// six fields, so that choosing one is `pipes[pipe_of(kind)]` and
    /// the choosing is a pure function a test can call. Nothing here
    /// can be checked without a device; what can be checked is which
    /// NAME a run asks for, and that is where the choice now lives.
    pipes: [vk::Pipeline; Pipe::N],
    /// Offscreen pass for the frosted-glass chain: same format as the
    /// swapchain (which is what lets the ordinary pipelines draw into
    /// it), ending in a shader-readable layout.
    blur_pass: vk::RenderPass,
    /// scene (full size), half, quarter, eighth — in that order.
    blur_targets: Vec<BlurTarget>,
    /// How deep the pyramid goes (1..=3): half, quarter, or eighth
    /// with a smoothing step back up. Set from the blur radius.
    blur_depth: u32,
    /// The theme's render.text_gamma: exponent on glyph coverage. 1.0 is the
    /// identity lens; a kind-fallback 0.0 from a themeless run is treated as
    /// identity too, because "no lens" is optics, not a design choice.
    text_gamma: f32,
    desc_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    desc_set: vk::DescriptorSet,
    /// Descriptors for registered images, freed one by one.
    tex_pool: vk::DescriptorPool,
    textures: HashMap<u32, Texture>,
    #[allow(dead_code)]
    next_tex: u32,
    /// Staging buffers whose copies were submitted last frame; the
    /// fence wait at the top of a frame is what makes them safe to
    /// free.
    retired: Vec<(vk::Buffer, vk::DeviceMemory)>,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    /// Preferred swapchain bit depth (8, 10, 12 or 16); what the
    /// surface actually offers decides, falling back down.
    depth_pref: u32,
    lut_layout: vk::DescriptorSetLayout,
    lut_set: vk::DescriptorSet,
    lut_image: vk::Image,
    lut_mem: vk::DeviceMemory,
    lut_view: vk::ImageView,
    /// Edge size of the loaded LUT; 0 = none, the shader passes
    /// colours through.
    lut_size: u32,
    /// Voxels (RGBA f32) waiting for the next frame's command buffer,
    /// with their edge size.
    lut_pending: Option<(u32, Vec<f32>)>,
    lut_initialized: bool,
    sampler: vk::Sampler,
    atlas_image: vk::Image,
    atlas_mem: vk::DeviceMemory,
    atlas_view: vk::ImageView,
    atlas_initialized: bool,
    /// Atlas rows staged but not yet copied to the GPU, as a merged
    /// `(y0, y1)` span. A frame that bails early (zero extent, an
    /// out-of-date swapchain) keeps the span here so the rows the font
    /// system already drained still reach the texture on the next frame
    /// that records commands.
    pending_atlas_rows: Option<(u32, u32)>,
    staging_buf: vk::Buffer,
    staging_mem: vk::DeviceMemory,
    staging_ptr: *mut u8,
    vertex_buf: vk::Buffer,
    vertex_mem: vk::DeviceMemory,
    vertex_ptr: *mut u8,
    /// Set 2: the frame's shape records as one storage buffer. Its own
    /// layout and its own little pool — set 0 is repinned per run and
    /// the texture pool frees sets one by one; the shape set is one,
    /// written once, bound per pass (f3 D4).
    shapes_layout: vk::DescriptorSetLayout,
    shapes_pool: vk::DescriptorPool,
    shapes_set: vk::DescriptorSet,
    shapes_buf: vk::Buffer,
    shapes_mem: vk::DeviceMemory,
    shapes_ptr: *mut u8,
    cmd_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
    sem_image: vk::Semaphore,
    sem_render: vk::Semaphore,
    fence: vk::Fence,
    needs_recreate: bool,
    /// The surface itself is gone and has to be built again before the
    /// swapchain can be. Separate from `needs_recreate` because the two
    /// repairs are ordered: every lost surface also costs a swapchain,
    /// but no resize costs a surface.
    needs_surface: bool,
    /// GPU timestamps, per pass, when `NACELLE_GPU_TIMING` asks for
    /// them. `None` — the default — is not a disabled instrument but
    /// no instrument at all: no query pool, no windows, no branch that
    /// costs more than reading this field.
    timing: Option<GpuTiming>,
}

impl Gfx {
    /// Builds the renderer for a surface described by raw handles.
    /// `width`/`height` are the surface's pixel size right now; the
    /// swapchain follows [`Gfx::render`]'s sizes from then on.
    ///
    /// Fails instead of panicking on everything the MACHINE decides:
    /// no Vulkan library, no GPU that can present here, a driver that
    /// refuses the device or the memory. A caller that gets an
    /// [`GfxError`] can print one sentence and exit; a caller that got
    /// the old `expect` could only watch a stack trace go by.
    ///
    /// A caller may also NOT exit — a desktop that cannot build one of
    /// its screens goes on drawing the others — and that is why every
    /// failing path here gives back what it had already taken. See
    /// [`Gfx::furnish`] for the half of the construction that fails
    /// into a `Drop` instead of by name.
    pub fn new(
        handle: &(impl HasDisplayHandle + HasWindowHandle),
        width: u32,
        height: u32,
    ) -> Result<Self, GfxError> {
        unsafe {
            let display_handle = handle
                .display_handle()
                .map_err(|_| GfxError::NoWindowHandle)?
                .as_raw();
            let window_handle = handle
                .window_handle()
                .map_err(|_| GfxError::NoWindowHandle)?
                .as_raw();

            // The library, the instance and the surface loader belong to
            // the process, not to this screen — see [`VK`] for the crash
            // that taught us so.
            let vk = shared(display_handle)?;

            let surface = ash_window::create_surface(
                &vk.entry,
                &vk.instance,
                display_handle,
                window_handle,
                None,
            )
            .map_err(GfxError::NoSurface)?;
            let surface_loader = vk.surface_loader.clone();

            // GPU + queue family selection (graphics + present).
            //
            // The three exits in this stretch destroy the surface BY
            // NAME, and they are the only ones that have to: until the
            // logical device exists there is no `Gfx` to hang the
            // handle on, so there is no `Drop` to run. Everything from
            // the device onwards belongs to [`Gfx::furnish`], which
            // fails into that `Drop` instead.
            let pdevices = match vk.instance.enumerate_physical_devices() {
                Ok(p) => p,
                Err(_) => {
                    surface_loader.destroy_surface(surface, None);
                    return Err(GfxError::NoDevice);
                }
            };
            let picked = pdevices.iter().find_map(|&pd| {
                vk.instance
                    .get_physical_device_queue_family_properties(pd)
                    .iter()
                    .enumerate()
                    .find_map(|(i, props)| {
                        let ok = props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                            && surface_loader
                                .get_physical_device_surface_support(pd, i as u32, surface)
                                .unwrap_or(false);
                        if ok { Some((pd, i as u32)) } else { None }
                    })
            });
            let Some((pdevice, queue_family)) = picked else {
                surface_loader.destroy_surface(surface, None);
                return Err(GfxError::NoDevice);
            };

            let priorities = [1.0f32];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            let dev_exts = [ash::khr::swapchain::NAME.as_ptr()];
            let device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&dev_exts);
            let device = match vk.instance.create_device(pdevice, &device_info, None) {
                Ok(d) => d,
                Err(e) => {
                    surface_loader.destroy_surface(surface, None);
                    return Err(GfxError::NoLogicalDevice(e));
                }
            };
            let queue = device.get_device_queue(queue_family, 0);
            // Unlike the surface loader, this one stays per `Gfx`: its
            // pointers come from `get_device_proc_addr`, so they belong
            // to this screen's device and to no other.
            let swapchain_loader = ash::khr::swapchain::Device::new(&vk.instance, &device);
            let mem_props = vk.instance.get_physical_device_memory_properties(pdevice);

            // The renderer exists from this line on, and it is EMPTY:
            // every handle null, every list empty, no format chosen.
            // Null is what makes that safe to drop — `vkDestroy*` and
            // `vkFree*` take a null handle and do nothing — and being
            // droppable is the whole point, because it is what the rest
            // of the construction fails INTO.
            let mut gfx = Gfx {
                surface_loader,
                surface,
                display_handle,
                window_handle,
                pdevice,
                device,
                queue,
                swapchain_loader,
                swapchain: vk::SwapchainKHR::null(),
                format: vk::SurfaceFormatKHR::default(),
                extent: vk::Extent2D { width: 0, height: 0 },
                images: vec![],
                views: vec![],
                render_pass: vk::RenderPass::null(),
                framebuffers: vec![],
                pipeline_layout: vk::PipelineLayout::null(),
                pipes: [vk::Pipeline::null(); Pipe::N],
                blur_pass: vk::RenderPass::null(),
                blur_targets: Vec::new(),
                blur_depth: 3,
                text_gamma: 1.0,
                desc_layout: vk::DescriptorSetLayout::null(),
                desc_pool: vk::DescriptorPool::null(),
                desc_set: vk::DescriptorSet::null(),
                tex_pool: vk::DescriptorPool::null(),
                textures: HashMap::new(),
                next_tex: 1,
                retired: Vec::new(),
                mem_props,
                depth_pref: 8,
                lut_layout: vk::DescriptorSetLayout::null(),
                lut_set: vk::DescriptorSet::null(),
                lut_image: vk::Image::null(),
                lut_mem: vk::DeviceMemory::null(),
                lut_view: vk::ImageView::null(),
                lut_size: 0,
                lut_pending: Some((2, identity_lut(2))),
                lut_initialized: false,
                sampler: vk::Sampler::null(),
                atlas_image: vk::Image::null(),
                atlas_mem: vk::DeviceMemory::null(),
                atlas_view: vk::ImageView::null(),
                atlas_initialized: false,
                pending_atlas_rows: None,
                staging_buf: vk::Buffer::null(),
                staging_mem: vk::DeviceMemory::null(),
                staging_ptr: std::ptr::null_mut(),
                vertex_buf: vk::Buffer::null(),
                vertex_mem: vk::DeviceMemory::null(),
                vertex_ptr: std::ptr::null_mut(),
                shapes_layout: vk::DescriptorSetLayout::null(),
                shapes_pool: vk::DescriptorPool::null(),
                shapes_set: vk::DescriptorSet::null(),
                shapes_buf: vk::Buffer::null(),
                shapes_mem: vk::DeviceMemory::null(),
                shapes_ptr: std::ptr::null_mut(),
                cmd_pool: vk::CommandPool::null(),
                cmd_buf: vk::CommandBuffer::null(),
                sem_image: vk::Semaphore::null(),
                sem_render: vk::Semaphore::null(),
                fence: vk::Fence::null(),
                needs_recreate: false,
                needs_surface: false,
                timing: None,
            };

            // This `?` is the whole reason the construction is in two
            // halves: it drops `gfx`, and that drop destroys the device
            // and the surface together with whatever `furnish` had
            // already managed to build.
            gfx.furnish(vk, queue_family)?;
            gfx.recreate_swapchain(width, height);
            Ok(gfx)
        }
    }

    /// Fills the empty renderer in: the format, the render passes, the
    /// descriptor plumbing, the pipelines, the glyph atlas, the mapped
    /// buffers, the command pool and the sync objects.
    ///
    /// WHY THIS IS A SECOND STEP and not simply the rest of
    /// [`Gfx::new`]: six of the steps below can fail on a machine that
    /// has no memory for what they ask, and turning those panics into
    /// errors would have MOVED the fault rather than repaired it. A
    /// `?` out of the middle of a constructor drops nothing — the
    /// struct whose `Drop` destroys the device does not exist yet — so
    /// each of those exits would have leaked a `VkDevice` and a
    /// `VkSurfaceKHR`, and with them the render passes, pipelines,
    /// pools, sampler and atlas image made before the failing line.
    /// Under the panics that came before, that cost nothing: the
    /// process ended and the kernel swept up. It costs everything for a
    /// caller that goes ON — a desktop skipping the one screen it
    /// cannot draw on is exactly the caller this branch exists for,
    /// and it would have leaked a device per unusable monitor.
    ///
    /// Filling an already-built `Gfx` gives every one of those failures
    /// the same exit and the same cleanup: the caller's `?` drops the
    /// half-built renderer, and [`Gfx`]'s `Drop` destroys precisely the
    /// handles that got made. The rest are still null, which every
    /// `vkDestroy*` and `vkFree*` accepts and ignores.
    ///
    /// # Safety
    /// Vulkan calls throughout. `vk` must be the instance this `Gfx`'s
    /// device was created from, and `queue_family` the family its queue
    /// was taken from.
    unsafe fn furnish(&mut self, vk: &Vk, queue_family: u32) -> Result<(), GfxError> {
        // Surface format: the depth preference decides later, at
        // recreate; the start is the plain eight-bit UNORM path.
        let formats = self
            .surface_loader
            .get_physical_device_surface_formats(self.pdevice, self.surface)
            .map_err(GfxError::NoFormat)?;
        self.format = pick_format(&formats, 8);

        self.render_pass = create_render_pass(&self.device, self.format.format);
        self.blur_pass = create_blur_pass(&self.device, self.format.format);

        // Descriptors: atlas texture (binding 0) + sampler (binding 1),
        // because WGSL has no combined image sampler.
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        self.desc_layout = self
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .unwrap();
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(1),
        ];
        self.desc_pool = self
            .device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
            .unwrap();
        let layouts = [self.desc_layout];
        self.desc_set = self
            .device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.desc_pool)
                    .set_layouts(&layouts),
            )
            .unwrap()[0];

        // The grading LUT lives in its own set (group 1): the
        // atlas set and every image set share group 0's layout,
        // and the LUT must not have to be written into each.
        let lut_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        self.lut_layout = self
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&lut_bindings),
                None,
            )
            .unwrap();

        // Set 2: the shape records (f3 D4). Set 0 is repinned per
        // run — a texture and a pyramid target each own a set — so
        // the one frame-wide storage buffer gets its own group, its
        // own layout and its own one-set pool; the existing pools
        // hold only SAMPLED_IMAGE and SAMPLER counts (R5).
        let shapes_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        self.shapes_layout = self
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&shapes_bindings),
                None,
            )
            .unwrap();
        let shapes_pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)];
        self.shapes_pool = self
            .device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&shapes_pool_sizes),
                None,
            )
            .unwrap();
        let shapes_layouts = [self.shapes_layout];
        self.shapes_set = self
            .device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.shapes_pool)
                    .set_layouts(&shapes_layouts),
            )
            .unwrap()[0];

        let pipes = create_pipeline(
            &self.device,
            self.render_pass,
            self.desc_layout,
            self.lut_layout,
            self.shapes_layout,
        );
        self.pipeline_layout = pipes.layout;
        self.pipes = pipes.by_pipe;

        // Room for the application's images: browser frames,
        // previews, a wallpaper. Freed individually as images go.
        let tex_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(96),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(96),
        ];
        self.tex_pool = self
            .device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(96)
                    .pool_sizes(&tex_pool_sizes),
                None,
            )
            .unwrap();

        // The identity LUT: two voxels an edge, each the colour of
        // its own corner. Present from the first frame so the
        // pipeline layout never has an unbound set; replaced in
        // place when a .cube is chosen.
        let (lut_image, lut_mem, lut_view) = create_lut_image(&self.device, &self.mem_props, 2)
            .ok_or(GfxError::NoMemory("grading LUT"))?;
        self.lut_image = lut_image;
        self.lut_mem = lut_mem;
        self.lut_view = lut_view;
        let lut_set_layouts = [self.lut_layout];
        self.lut_set = self
            .device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.tex_pool)
                    .set_layouts(&lut_set_layouts),
            )
            .unwrap()[0];

        // Glyph atlas: R8, device-local.
        let atlas_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8_UNORM)
            .extent(vk::Extent3D {
                width: ATLAS_W as u32,
                height: ATLAS_H as u32,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        self.atlas_image = self.device.create_image(&atlas_info, None).unwrap();
        let req = self.device.get_image_memory_requirements(self.atlas_image);
        self.atlas_mem = alloc_memory(
            &self.device,
            &self.mem_props,
            req,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or(GfxError::NoMemory("glyph atlas"))?;
        self.device
            .bind_image_memory(self.atlas_image, self.atlas_mem, 0)
            .unwrap();
        self.atlas_view = self
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(self.atlas_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8_UNORM)
                    .subresource_range(color_range()),
                None,
            )
            .unwrap();

        self.sampler = self
            .device
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
            .unwrap();

        let tex_info = [vk::DescriptorImageInfo::default()
            .image_view(self.atlas_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let samp_info = [vk::DescriptorImageInfo::default().sampler(self.sampler)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&tex_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&samp_info),
        ];
        self.device.update_descriptor_sets(&writes, &[]);
        let lut_tex_info = [vk::DescriptorImageInfo::default()
            .image_view(self.lut_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let lut_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.lut_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&lut_tex_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.lut_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&samp_info),
        ];
        self.device.update_descriptor_sets(&lut_writes, &[]);

        // Staging buffer for atlas uploads.
        let (staging_buf, staging_mem, staging_ptr) = create_host_buffer(
            &self.device,
            &self.mem_props,
            (ATLAS_W * ATLAS_H) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .ok_or(GfxError::NoMemory("atlas staging buffer"))?;
        self.staging_buf = staging_buf;
        self.staging_mem = staging_mem;
        self.staging_ptr = staging_ptr;

        // Vertex buffer (host-visible, persistently mapped).
        let (vertex_buf, vertex_mem, vertex_ptr) = create_host_buffer(
            &self.device,
            &self.mem_props,
            (MAX_VERTS * std::mem::size_of::<Vertex>()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )
        .ok_or(GfxError::NoMemory("vertex buffer"))?;
        self.vertex_buf = vertex_buf;
        self.vertex_mem = vertex_mem;
        self.vertex_ptr = vertex_ptr;

        // Shape records: the same persistent mapping, written each
        // frame beside the vertices, read by fs_shape through set 2.
        let (shapes_buf, shapes_mem, shapes_ptr) = create_host_buffer(
            &self.device,
            &self.mem_props,
            (MAX_SHAPES * std::mem::size_of::<Shape>()) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .ok_or(GfxError::NoMemory("shape record buffer"))?;
        self.shapes_buf = shapes_buf;
        self.shapes_mem = shapes_mem;
        self.shapes_ptr = shapes_ptr;
        let shapes_buf_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.shapes_buf)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let shapes_write = [vk::WriteDescriptorSet::default()
            .dst_set(self.shapes_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&shapes_buf_info)];
        self.device.update_descriptor_sets(&shapes_write, &[]);

        self.cmd_pool = self
            .device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();
        self.cmd_buf = self
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap()[0];

        self.sem_image = self
            .device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        self.sem_render = self
            .device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        self.fence = self
            .device
            .create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
            .unwrap();

        // GPU timing, only if asked for. Two device facts decide
        // whether it can work at all: the length of a tick
        // (`timestampPeriod`, nanoseconds) and how many bits of a
        // timestamp this queue family actually fills
        // (`timestampValidBits`). The family is asked rather than
        // the device-wide `timestampComputeAndGraphics`, because
        // the family's answer is the exact one for the queue the
        // frames go to; the device-wide promise is only reported.
        let limits = vk.instance.get_physical_device_properties(self.pdevice).limits;
        let valid_bits = vk
            .instance
            .get_physical_device_queue_family_properties(self.pdevice)
            .get(queue_family as usize)
            .map(|q| q.timestamp_valid_bits)
            .unwrap_or(0);
        self.timing = GpuTiming::from_env(
            &self.device,
            limits.timestamp_period,
            valid_bits,
            limits.timestamp_compute_and_graphics == vk::TRUE,
            FRAMES_IN_FLIGHT,
        );

        Ok(())
    }

    pub fn resize(&mut self) {
        self.needs_recreate = true;
    }

    /// Asks for a swapchain bit depth. Takes effect at the next
    /// swapchain rebuild; pass through [`Gfx::resize`] to force one.
    pub fn set_color_depth(&mut self, bits: u32) {
        let bits = match bits {
            8 | 10 | 12 | 16 => bits,
            _ => 8,
        };
        if bits != self.depth_pref {
            self.depth_pref = bits;
            self.needs_recreate = true;
        }
    }

    /// The bit depth the swapchain ACTUALLY carries, which is not always
    /// the one that was asked for: [`Gfx::set_color_depth`] states a
    /// wish and the surface answers with whatever formats it has.
    ///
    /// Read rather than remembered, so it cannot drift from the picture:
    /// the number comes off `self.format`, the same field the images and
    /// the render pass were built from. Valid from the first frame — the
    /// constructor builds a swapchain before it hands the `Gfx` back —
    /// and it moves at the rebuild, not at the request, so a caller that
    /// asks in the same breath as it sets still gets the OLD answer.
    ///
    /// THE REBUILD IS INSIDE [`Gfx::render`], which is the whole of what
    /// a caller has to know: ask AFTER a frame has been drawn, never
    /// between setting the depth and drawing. The desktop's settings
    /// window reads it on the line following its `draw_screen`, and
    /// treats the gap between the request and that read as "not
    /// measured" rather than as an answer — a wish paired with the
    /// depth of the format being replaced reads as a shortfall the
    /// surface was never asked about.
    pub fn color_depth(&self) -> u32 {
        format_bits(self.format.format)
    }

    /// The theme's glyph-coverage exponent (render.text_gamma). Clamped to
    /// the token's own stated range; 0 (the engine's kind fallback when no
    /// theme declares it) means identity.
    pub fn set_text_gamma(&mut self, g: f32) {
        self.text_gamma = if g <= 0.0 { 1.0 } else { g.clamp(0.45, 2.2) };
    }

    /// Sets the frosted-glass radius as a percentage. It picks how
    /// deep the downsampling pyramid goes — a third each: the half,
    /// the quarter, or the eighth — which is also how many glass ranks
    /// the frame can serve (see [`Gfx::glass_ranks`]).
    pub fn set_blur_radius(&mut self, percent: u32) {
        self.blur_depth = match percent.min(100) {
            0..=33 => 1,
            34..=66 => 2,
            _ => 3,
        };
    }

    /// How many glass ranks this frame's pyramid actually writes. The
    /// theme resolver clamps every `elev.*.glass.rank` against it at
    /// bake; the renderer's own rank mapping never trusts the token.
    pub fn glass_ranks(&self) -> u8 {
        self.blur_depth.clamp(1, 3) as u8
    }

    /// Writes down what a [`Fault`] asks for. Never clears a flag —
    /// two faults in one frame are two repairs, not the later one.
    fn note(&mut self, f: Fault) {
        self.needs_surface |= f.rebuilds_surface();
        self.needs_recreate |= f.rebuilds_swapchain();
    }

    /// Builds a new surface for the same window after the old one was
    /// lost, and everything that hung off it.
    ///
    /// A surface dies for reasons that have nothing to do with this
    /// program: a monitor unplugged, a compositor restarted, a session
    /// switched. Vulkan's answer to all of them is `SURFACE_LOST_KHR`
    /// from whichever call happened to be next, and the only repair is
    /// another surface — the handle itself is dead, not stale.
    ///
    /// The two semaphores are made anew here on purpose. A frame that
    /// died at `queue_present` left `sem_render` signalled with nobody
    /// ever going to wait on it, and the next frame's submit would
    /// signal a signalled semaphore. The device is idle at this point,
    /// so replacing them is free and it is the only moment in the frame
    /// loop when it is.
    ///
    /// The outcome is read off the fields, not returned: a success
    /// clears `needs_surface` and sets `needs_recreate`, a failure
    /// leaves `needs_surface` standing and the extent at zero — which
    /// is what makes `render` skip the frame rather than draw into
    /// nothing, and what brings the next frame back here to try again.
    ///
    /// The teardown runs once per loss, not once per retry — AS FAR AS
    /// `first_try` reaches, and that is not as far as it sounds. The
    /// flag is `surface != null`, and the surface is only left null
    /// when `create_surface` itself FAILED. Under Wayland and X11 that
    /// call mostly just wraps the handles it is given
    /// (`wl_display` + `wl_surface`, or an xcb connection and window)
    /// and SUCCEEDS even for a window that is gone; the loss is then
    /// reported by the next call that touches the surface for real.
    /// So a window closed for good takes the other road: a surface is
    /// made, `recreate_swapchain` meets `SURFACE_LOST` again, sets
    /// `needs_surface` again, and the next frame tears the whole thing
    /// down once more — device wait, blur targets, framebuffers,
    /// views, swapchain, surface and both semaphores — sixty times a
    /// second, and silently, because the message above sits only in
    /// the branch where `create_surface` failed.
    ///
    /// NOT REPAIRED HERE, and named rather than left to be rediscovered:
    /// the honest fix is a backoff on the repair itself — a surface
    /// counts as proven only once a swapchain has been built on it, and
    /// an unproven one is not retried on every frame. That is a state
    /// machine, not a line, and this branch is a bugfix branch.
    unsafe fn rebuild_surface(&mut self) {
        let first_try = self.surface != vk::SurfaceKHR::null();
        if first_try {
            let _ = self.device.device_wait_idle();
            self.destroy_blur_targets();
            for fb in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(fb, None);
            }
            for v in self.views.drain(..) {
                self.device.destroy_image_view(v, None);
            }
            // The swapchain first: it names the surface, so it cannot
            // outlive it.
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader.destroy_swapchain(self.swapchain, None);
                self.swapchain = vk::SwapchainKHR::null();
            }
            self.images.clear();
            self.surface_loader.destroy_surface(self.surface, None);
            self.surface = vk::SurfaceKHR::null();
            self.extent = vk::Extent2D { width: 0, height: 0 };

            for s in [self.sem_image, self.sem_render] {
                self.device.destroy_semaphore(s, None);
            }
            self.sem_image = self
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap();
            self.sem_render = self
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap();
        }

        let Some(vk) = VK.get() else { return };
        match ash_window::create_surface(
            &vk.entry,
            &vk.instance,
            self.display_handle,
            self.window_handle,
            None,
        ) {
            Ok(s) => {
                self.surface = s;
                self.needs_surface = false;
                self.needs_recreate = true;
            }
            Err(e) if first_try => {
                eprintln!(
                    "nacelle-renderer: the surface was lost and will not come back yet ({e:?}); \
                     frames are skipped until it does"
                );
            }
            Err(_) => {}
        }
    }

    fn recreate_swapchain(&mut self, width: u32, height: u32) {
        unsafe {
            // A surface that is gone answers nothing; asking it for
            // capabilities below would only be the same loss again.
            if self.surface == vk::SurfaceKHR::null() {
                self.extent = vk::Extent2D { width: 0, height: 0 };
                return;
            }
            let _ = self.device.device_wait_idle();
            for fb in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(fb, None);
            }
            for v in self.views.drain(..) {
                self.device.destroy_image_view(v, None);
            }

            // The depth preference may have moved since the last
            // build. A new format means a new render pass and new
            // pipelines — they are compiled against it.
            //
            // This is the FIRST of the three questions this function
            // puts to the surface, and it is read like the other two.
            // It used to be the exception — `if let Ok(..)`, no else —
            // and a `SURFACE_LOST` from here was recorded nowhere: the
            // old format was kept and the function walked on to ask a
            // dead surface for its size. The loss was caught twenty
            // lines down, so nothing broke; but a rule that holds for
            // two of three neighbouring calls is a rule nobody will
            // read as a rule.
            let formats = match self
                .surface_loader
                .get_physical_device_surface_formats(self.pdevice, self.surface)
            {
                Ok(f) => f,
                Err(e) => {
                    let f = fault_of(e);
                    if f.is_fatal() {
                        eprintln!("nacelle-renderer: the surface will not say its formats: {e:?}");
                    }
                    self.extent = vk::Extent2D { width: 0, height: 0 };
                    self.needs_recreate = true;
                    self.needs_surface |= f.rebuilds_surface();
                    return;
                }
            };
            {
                let want = pick_format(&formats, self.depth_pref);
                if want.format != self.format.format
                    || want.color_space != self.format.color_space
                {
                    eprintln!(
                        "nacelle-desktop: swapchain format {:?} ({:?})",
                        want.format, want.color_space
                    );
                    self.format = want;
                    for p in self.pipes {
                        self.device.destroy_pipeline(p, None);
                    }
                    self.device
                        .destroy_pipeline_layout(self.pipeline_layout, None);
                    self.device.destroy_render_pass(self.render_pass, None);
                    self.device.destroy_render_pass(self.blur_pass, None);
                    self.render_pass =
                        create_render_pass(&self.device, self.format.format);
                    self.blur_pass =
                        create_blur_pass(&self.device, self.format.format);
                    let pipes = create_pipeline(
                        &self.device,
                        self.render_pass,
                        self.desc_layout,
                        self.lut_layout,
                        self.shapes_layout,
                    );
                    self.pipeline_layout = pipes.layout;
                    self.pipes = pipes.by_pipe;
                }
            }

            // THE call that ended the desktop when a monitor was
            // unplugged: a resize arrives, the surface is already gone,
            // and asking a dead surface for its size answered
            // SURFACE_LOST into an `unwrap`.
            let caps = match self
                .surface_loader
                .get_physical_device_surface_capabilities(self.pdevice, self.surface)
            {
                Ok(c) => c,
                Err(e) => {
                    let f = fault_of(e);
                    if f.is_fatal() {
                        eprintln!("nacelle-renderer: the surface will not say its size: {e:?}");
                    }
                    self.extent = vk::Extent2D { width: 0, height: 0 };
                    self.needs_recreate = true;
                    self.needs_surface |= f.rebuilds_surface();
                    return;
                }
            };
            let extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                    height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
                }
            };
            if extent.width == 0 || extent.height == 0 {
                // The surface has no pixels right now. Everything the
                // old size owned was already destroyed above, so the
                // extent must go to zero with it: that is what tells
                // `render` to skip the frame instead of indexing into
                // framebuffers this call never rebuilt. `needs_recreate`
                // stays set, so a real size still gets one.
                self.extent = extent;
                return;
            }
            let mut image_count = caps.min_image_count + 1;
            if caps.max_image_count > 0 {
                image_count = image_count.min(caps.max_image_count);
            }
            let old = self.swapchain;
            let info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(image_count)
                .image_format(self.format.format)
                .image_color_space(self.format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true)
                .old_swapchain(old);
            self.swapchain = match self.swapchain_loader.create_swapchain(&info, None) {
                Ok(s) => s,
                Err(e) => {
                    // The same reasoning as the capabilities call: a
                    // swapchain refused for a surface that died between
                    // the two calls is a repair, not an ending.
                    let f = fault_of(e);
                    if f.is_fatal() {
                        eprintln!("nacelle-renderer: cannot create a swapchain: {e:?}");
                    }
                    if old != vk::SwapchainKHR::null() {
                        self.swapchain_loader.destroy_swapchain(old, None);
                        self.swapchain = vk::SwapchainKHR::null();
                    }
                    self.extent = vk::Extent2D { width: 0, height: 0 };
                    self.needs_recreate = true;
                    self.needs_surface |= f.rebuilds_surface();
                    return;
                }
            };
            if old != vk::SwapchainKHR::null() {
                self.swapchain_loader.destroy_swapchain(old, None);
            }
            self.extent = extent;
            self.images = self
                .swapchain_loader
                .get_swapchain_images(self.swapchain)
                .unwrap();
            for &img in &self.images {
                let view = self
                    .device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(img)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(self.format.format)
                            .subresource_range(color_range()),
                        None,
                    )
                    .unwrap();
                self.views.push(view);
                let attachments = [view];
                let fb = self
                    .device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(self.render_pass)
                            .attachments(&attachments)
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                    .unwrap();
                self.framebuffers.push(fb);
            }
            self.rebuild_blur_targets();
            self.needs_recreate = false;
        }
    }

    /// (Re)builds the frosted-glass chain for the current extent and
    /// format: the base scene at full size and the half / quarter /
    /// eighth pyramid the blur is made of.
    ///
    /// A chain that cannot be built leaves the vector EMPTY rather than
    /// half-built, and that is already a state the frame path knows:
    /// the base-scene split asks for `blur_targets.len() == 4` before
    /// it fires and the rank lookup goes through `get`. So a device
    /// with no memory for the pyramid draws every frosted surface flat
    /// instead of ending the desktop.
    unsafe fn rebuild_blur_targets(&mut self) {
        self.destroy_blur_targets();
        let (w, h) = (self.extent.width, self.extent.height);
        for div in [1u32, 2, 4, 8] {
            let (tw, th) = ((w / div).max(1), (h / div).max(1));
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(self.format.format)
                .extent(vk::Extent3D { width: tw, height: th, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                )
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = self.device.create_image(&info, None).unwrap();
            let req = self.device.get_image_memory_requirements(image);
            let Some(mem) = alloc_memory(
                &self.device,
                &self.mem_props,
                req,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ) else {
                self.device.destroy_image(image, None);
                eprintln!("nacelle-renderer: no GPU memory for the blur chain; glass draws flat");
                self.destroy_blur_targets();
                return;
            };
            self.device.bind_image_memory(image, mem, 0).unwrap();
            let view = self
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.format.format)
                        .subresource_range(color_range()),
                    None,
                )
                .unwrap();
            let attachments = [view];
            let fb = self
                .device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.blur_pass)
                        .attachments(&attachments)
                        .width(tw)
                        .height(th)
                        .layers(1),
                    None,
                )
                .unwrap();
            let layouts = [self.desc_layout];
            let desc_set = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.tex_pool)
                        .set_layouts(&layouts),
                )
                .unwrap()[0];
            let tex_info = [vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let samp_info =
                [vk::DescriptorImageInfo::default().sampler(self.sampler)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&tex_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&samp_info),
            ];
            self.device.update_descriptor_sets(&writes, &[]);
            self.blur_targets.push(BlurTarget { image, mem, view, fb, desc_set, w: tw, h: th });
        }
    }

    unsafe fn destroy_blur_targets(&mut self) {
        for t in self.blur_targets.drain(..) {
            self.device.destroy_framebuffer(t.fb, None);
            self.device.destroy_image_view(t.view, None);
            self.device.destroy_image(t.image, None);
            self.device.free_memory(t.mem, None);
            let _ = self.device.free_descriptor_sets(self.tex_pool, &[t.desc_set]);
        }
    }

    /// Renders a frame into a surface currently `width` x `height`
    /// pixels. Pass `atlas` only when the atlas has changed. `runs`
    /// partitions `verts` by texture; empty means all atlas. `shapes`
    /// is the frame's shape records — what a run tagged SHAPE indexes
    /// through each vertex's `shape`; empty when the vector lane is
    /// dark, which is the shipping default.
    pub fn render(
        &mut self,
        width: u32,
        height: u32,
        verts: &[Vertex],
        runs: &[DrawRun],
        shapes: &[Shape],
        atlas: Option<(&[u8], u32, u32)>,
        clear: [f32; 4],
    ) {
        unsafe {
            // Surface before swapchain: the second hangs off the first,
            // so a swapchain built while the surface is still dead is
            // one more dead thing. `rebuild_surface` sets
            // `needs_recreate` itself when it succeeds.
            if self.needs_surface {
                self.rebuild_surface();
            }
            if self.needs_recreate {
                self.recreate_swapchain(width, height);
            }
            if self.extent.width == 0 || self.extent.height == 0 {
                // The caller's font system drained its dirty rows to hand
                // them to us; bailing here without staging them would lose
                // them forever — on the first frame that means the mask
                // band never reaches the GPU and every sprite glow stays
                // dark. The fence wait makes the staging buffer safe to
                // write (the previous frame's copy may still be reading it).
                if atlas.is_some() {
                    self.device
                        .wait_for_fences(&[self.fence], true, u64::MAX)
                        .unwrap();
                    self.stash_atlas_rows(atlas);
                }
                return;
            }

            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .unwrap();

            // The copies these carried were submitted last frame, and
            // the fence just said that frame is done.
            for (buf, mem) in self.retired.drain(..) {
                self.device.destroy_buffer(buf, None);
                self.device.free_memory(mem, None);
            }

            // The same fence is what makes the previous frame's
            // timestamps readable, and this is the one moment they are
            // both complete and not yet reset. Reading never waits on
            // the GPU: a query that is somehow not published costs one
            // sample and nothing else.
            let (tw, th) = (self.extent.width, self.extent.height);
            if let Some(t) = self.timing.as_mut() {
                t.collect(&self.device, tw, th);
            }

            let acquired = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.sem_image,
                vk::Fence::null(),
            );
            let image_index = match acquired {
                Ok((idx, _)) => idx,
                Err(e) => {
                    let f = fault_of(e);
                    if f.is_fatal() {
                        panic!("acquire_next_image: {e:?}");
                    }
                    // Fence already waited above: staging is ours to write.
                    self.stash_atlas_rows(atlas);
                    self.note(f);
                    return;
                }
            };

            self.device.reset_fences(&[self.fence]).unwrap();

            // Upload vertices (the buffer is persistently mapped).
            let n = verts.len().min(MAX_VERTS);
            std::ptr::copy_nonoverlapping(
                verts.as_ptr() as *const u8,
                self.vertex_ptr,
                n * std::mem::size_of::<Vertex>(),
            );

            // And the shape records beside them — same mapping, same
            // overflow rule as MAX_VERTS: clip, never lose the frame.
            // fs_shape clamps its index, so a vertex pointing past the
            // clip reads the last record instead of out of bounds.
            let m = shapes.len().min(MAX_SHAPES);
            std::ptr::copy_nonoverlapping(
                shapes.as_ptr() as *const u8,
                self.shapes_ptr,
                m * std::mem::size_of::<Shape>(),
            );

            // Frosted glass: everything before the first glass run —
            // any rank, the legacy BLUR_IMAGE included — is the BASE
            // SCENE: rendered offscreen, shrunk down the pyramid (which
            // is the blur), and brought back as one quad. Each glass
            // quad then samples the pyramid target its rank resolves
            // to, by screen position.
            let mut blur_base: Option<u32> = None;
            {
                let mut prev = 0u32;
                for run in runs {
                    if run.image.is_some_and(is_glass) {
                        blur_base = Some(prev.min(n as u32));
                        break;
                    }
                    prev = run.end;
                }
            }
            let blur_base = blur_base
                .filter(|_| self.blur_targets.len() == 4 && n + 6 <= MAX_VERTS);
            // The one helper quad every glass pass draws: a unit
            // square, which a push constant of screen = (1, 1) turns
            // into "the whole target", whatever its size.
            let aux = n as u32;
            if blur_base.is_some() {
                let corners: [([f32; 2], [f32; 2]); 6] = [
                    ([0.0, 0.0], [0.0, 0.0]),
                    ([1.0, 0.0], [1.0, 0.0]),
                    ([1.0, 1.0], [1.0, 1.0]),
                    ([0.0, 0.0], [0.0, 0.0]),
                    ([1.0, 1.0], [1.0, 1.0]),
                    ([0.0, 1.0], [0.0, 1.0]),
                ];
                let quad: Vec<Vertex> = corners
                    .iter()
                    .map(|&(pos, uv)| Vertex { pos, uv, color: [1.0; 4], shape: NO_SHAPE })
                    .collect();
                std::ptr::copy_nonoverlapping(
                    quad.as_ptr() as *const u8,
                    self.vertex_ptr.add(n * std::mem::size_of::<Vertex>()),
                    6 * std::mem::size_of::<Vertex>(),
                );
            }

            // The atlas travels by ROWS: (pixels, first row, row count). A
            // glyph-churn frame re-uploads a shelf, not four megabytes —
            // r1's mandatory rider on growing the atlas, without which the
            // 2048 square would cost a millisecond per typed character. The
            // staging buffer is written at the same row offset the copy
            // reads, so the region maths stays in one place.
            self.stash_atlas_rows(atlas);
            let upload_rows = self
                .pending_atlas_rows
                .take()
                .map(|(y0, y1)| (y0, y1 - y0));
            let upload_atlas = upload_rows.is_some() || !self.atlas_initialized;

            let cmd = self.cmd_buf;
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();
            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            // Opens the frame's timestamps. It has to be here: after
            // the command buffer is recording (the pool reset is a
            // command) and before any render pass (a reset may not be
            // inside one). `blur_base` already knows whether this
            // frame has glass, which is what decides how many of the
            // six timestamps it will write.
            if let Some(t) = self.timing.as_mut() {
                t.begin_frame(&self.device, cmd, blur_base.is_some());
            }

            if upload_atlas {
                // Before the first full upload the staging buffer may hold
                // only a fragment; initialisation always copies everything.
                let (y0, rows) = if self.atlas_initialized {
                    upload_rows.unwrap_or((0, ATLAS_H as u32))
                } else {
                    (0, ATLAS_H as u32)
                };
                self.record_atlas_upload(cmd, y0, rows);
                self.atlas_initialized = true;
            }
            self.record_texture_uploads(cmd);
            self.record_lut_upload(cmd);
            self.gpu_mark(cmd, SLOT_UPLOADS_END);

            // Bindings that hold for every pass this frame.
            self.device
                .cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buf], &[0]);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                1,
                &[self.lut_set],
                &[],
            );

            if let Some(base) = blur_base {
                // The base scene, full size, UNGRADED — the grade is
                // applied once, at composite.
                self.push_pc(cmd, self.extent.width as f32, self.extent.height as f32, 0.0);
                let t0 = (self.blur_targets[0].fb, self.blur_targets[0].w, self.blur_targets[0].h);
                self.begin_blur_pass(cmd, t0.0, t0.1, t0.2, clear);
                self.record_runs(cmd, 0, base, runs, (t0.1, t0.2), false);
                self.device.cmd_end_render_pass(cmd);
                self.gpu_mark(cmd, SLOT_BASE_END);
                // Down the pyramid: each level is the previous one
                // resampled linearly, and the shrinking IS the blur —
                // then back up (and, at full depth, down once more) so
                // every target a rank can sample is smooth, not blocky.
                self.push_pc(cmd, 1.0, 1.0, 0.0);
                let steps = pyramid_steps(self.blur_depth);
                for &(dst, src) in steps {
                    let (fb, w, h) = (
                        self.blur_targets[dst].fb,
                        self.blur_targets[dst].w,
                        self.blur_targets[dst].h,
                    );
                    self.begin_blur_pass(cmd, fb, w, h, [0.0; 4]);
                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipes[Pipe::Image as usize],
                    );
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[self.blur_targets[src].desc_set],
                        &[],
                    );
                    self.device.cmd_draw(cmd, 6, 1, aux, 0);
                    self.device.cmd_end_render_pass(cmd);
                }
                self.gpu_mark(cmd, SLOT_PYRAMID_END);
            }

            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue { float32: clear },
            }];
            let rp_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                })
                .clear_values(&clear_values);
            self.device
                .cmd_begin_render_pass(cmd, &rp_info, vk::SubpassContents::INLINE);
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.extent.width as f32,
                height: self.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                }],
            );
            match blur_base {
                Some(base) => {
                    // The base scene arrives as one quad, graded here
                    // and only here.
                    self.push_pc(cmd, 1.0, 1.0, self.lut_size as f32);
                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipes[Pipe::Image as usize],
                    );
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[self.blur_targets[0].desc_set],
                        &[],
                    );
                    self.device.cmd_draw(cmd, 6, 1, aux, 0);
                    // The composite ends here and the main pass
                    // begins; a timestamp inside a render pass is
                    // legal, and it is the only place the boundary
                    // exists, because both live in the same pass.
                    self.gpu_mark(cmd, SLOT_MAIN_START);
                    // Everything above the glass, glass included.
                    self.push_pc(
                        cmd,
                        self.extent.width as f32,
                        self.extent.height as f32,
                        self.lut_size as f32,
                    );
                    self.record_runs(
                        cmd,
                        base,
                        n as u32,
                        runs,
                        (self.extent.width, self.extent.height),
                        true,
                    );
                }
                None => {
                    // No glass, so no composite: the main pass starts
                    // with the runs themselves. What the swapchain
                    // pass spent getting here shows up as the
                    // report's leftover, not inside a named span.
                    self.gpu_mark(cmd, SLOT_MAIN_START);
                    self.push_pc(
                        cmd,
                        self.extent.width as f32,
                        self.extent.height as f32,
                        self.lut_size as f32,
                    );
                    self.record_runs(
                        cmd,
                        0,
                        n as u32,
                        runs,
                        (self.extent.width, self.extent.height),
                        false,
                    );
                }
            }
            self.device.cmd_end_render_pass(cmd);
            self.gpu_mark(cmd, SLOT_FRAME_END);
            self.device.end_command_buffer(cmd).unwrap();

            let wait_sems = [self.sem_image];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_sems = [self.sem_render];
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmds)
                .signal_semaphores(&signal_sems);
            self.device
                .queue_submit(self.queue, &[submit], self.fence)
                .unwrap();
            // The queries are in flight from here; the frame counter
            // moves on and the read of this frame is due once the
            // fence has been waited on.
            if let Some(t) = self.timing.as_mut() {
                t.end_frame();
            }

            let swapchains = [self.swapchain];
            let indices = [image_index];
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_sems)
                .swapchains(&swapchains)
                .image_indices(&indices);
            match self.swapchain_loader.queue_present(self.queue, &present) {
                Ok(suboptimal) => {
                    if suboptimal {
                        self.needs_recreate = true;
                    }
                }
                Err(e) => {
                    let f = fault_of(e);
                    if f.is_fatal() {
                        panic!("queue_present: {e:?}");
                    }
                    self.note(f);
                }
            }
        }
    }

    /// One of the frame's timestamps, if anyone asked for timing. The
    /// whole per-site cost when nobody did is this `Option` check —
    /// which is why the recording sites can sit in the frame path
    /// without a feature flag around each of them.
    #[inline]
    unsafe fn gpu_mark(&self, cmd: vk::CommandBuffer, slot: usize) {
        if let Some(t) = self.timing.as_ref() {
            t.mark(&self.device, cmd, slot);
        }
    }

    unsafe fn push_pc(&self, cmd: vk::CommandBuffer, w: f32, h: f32, lut: f32) {
        // 64 bytes (D0): screen/lut/text_gamma in the first 16, then the
        // homography's three columns at a 16-byte stride — WGSL lays a
        // mat3x3 out as three vec3s each padded to 16. Identity today:
        // the cube still projects on the CPU, and identity is proven
        // bit-for-bit neutral (p.z == 1.0, the manual divide is by 1.0).
        let mut push = [0.0f32; 16];
        push[0] = w;
        push[1] = h;
        push[2] = lut;
        push[3] = self.text_gamma;
        push[4] = 1.0; // column 0 . x
        push[9] = 1.0; // column 1 . y
        push[14] = 1.0; // column 2 . z
        self.device.cmd_push_constants(
            cmd,
            self.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            std::slice::from_raw_parts(push.as_ptr() as *const u8, 64),
        );
    }

    unsafe fn begin_blur_pass(
        &self,
        cmd: vk::CommandBuffer,
        fb: vk::Framebuffer,
        w: u32,
        h: u32,
        clear: [f32; 4],
    ) {
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue { float32: clear },
        }];
        let info = vk::RenderPassBeginInfo::default()
            .render_pass(self.blur_pass)
            .framebuffer(fb)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: w, height: h },
            })
            .clear_values(&clear_values);
        self.device
            .cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: w as f32,
            height: h as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        self.device.cmd_set_viewport(cmd, 0, &[viewport]);
        self.device.cmd_set_scissor(
            cmd,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: w, height: h },
            }],
        );
    }

    /// Replays the draw-list runs whose vertices fall in `[from, to)`,
    /// binding per run kind: the atlas (straight or additive), a
    /// registered image, or — for glass — the pyramid target the run's
    /// rank resolves to at this frame's blur depth. Each run's clip
    /// becomes the scissor, clamped into `target` (the render target's
    /// size: the offscreen scene target for the base pass, the surface
    /// for the composite); no clip restores the full target, which is
    /// what the pass began with. `glass_live` says whether the pyramid
    /// was written this frame — when it was not, glass runs are simply
    /// not there rather than sampling garbage.
    unsafe fn record_runs(
        &self,
        cmd: vk::CommandBuffer,
        from: u32,
        to: u32,
        runs: &[DrawRun],
        target: (u32, u32),
        glass_live: bool,
    ) {
        if to <= from {
            return;
        }
        let (tw, th) = target;
        // Set 2 — the frame's shape records — holds for the whole pass:
        // one set, written once, bound here rather than per run. Only
        // fs_shape reads it, but binding is state, not work.
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            2,
            &[self.shapes_set],
            &[],
        );
        // The pass has just set the full-target scissor, so that is
        // what `cur` starts as and what a clipless run restores.
        let mut cur = scissor_for(None, tw, th);
        // What set 0 currently holds; None = nothing bound yet.
        let mut bound: Option<RunKind> = None;
        let mut segment =
            |start: u32, end: u32, image: Option<ImageId>, clip: Option<[f32; 4]>| {
                let (start, end) = (start.max(from), end.min(to));
                if end <= start {
                    return;
                }
                let want = scissor_for(clip, tw, th);
                if want[2] <= 0 || want[3] <= 0 {
                    // The clip is empty: nothing of this run can land.
                    return;
                }
                let kind = run_kind(image, self.blur_depth);
                // Resolve before recording anything, so a skipped run
                // leaves no scissor or binding behind.
                // WHICH pipeline is `pipe_of`'s answer and nothing
                // else's — a pure function, tested for every kind. What
                // is left here is which DESCRIPTOR SET goes with it,
                // which needs the frame's own state and cannot be
                // decided anywhere but in the middle of recording.
                let pipeline = self.pipes[pipe_of(kind) as usize];
                let set = match kind {
                    // fs_shape samples nothing from set 0, but the
                    // shared layout demands one bound; the atlas set is
                    // always alive.
                    RunKind::Atlas | RunKind::Add | RunKind::Shape | RunKind::ShapeAdd => {
                        self.desc_set
                    }
                    // A frosted band reads the pyramid like glass, so it
                    // binds what glass binds — the pipeline it draws
                    // with is the one that ALSO reads set 2.
                    RunKind::ShapeGlass(t) | RunKind::Glass(t) => {
                        if !glass_live {
                            // No blurred scene this frame: the glass
                            // would sample garbage, so it is simply
                            // not there.
                            return;
                        }
                        match self.blur_targets.get(t) {
                            Some(bt) => bt.desc_set,
                            None => return,
                        }
                    }
                    RunKind::Image(id) => match self.textures.get(&id) {
                        Some(tex) => tex.desc_set,
                        // The image is gone — or the id is a reserved
                        // instruction this renderer does not know.
                        // Skipping the run beats sampling a stale
                        // descriptor.
                        None => return,
                    },
                };
                if want != cur {
                    self.device.cmd_set_scissor(
                        cmd,
                        0,
                        &[vk::Rect2D {
                            offset: vk::Offset2D { x: want[0], y: want[1] },
                            extent: vk::Extent2D {
                                width: want[2] as u32,
                                height: want[3] as u32,
                            },
                        }],
                    );
                    cur = want;
                }
                if bound != Some(kind) {
                    self.device
                        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[set],
                        &[],
                    );
                    bound = Some(kind);
                }
                self.device.cmd_draw(cmd, end - start, 1, start, 0);
            };
        if runs.is_empty() {
            segment(0, to, None, None);
        } else {
            let mut start = 0u32;
            for run in runs {
                segment(start, run.end, run.image, run.clip);
                start = run.end;
            }
        }
    }

    /// Loads a grading LUT parsed from a .cube file, or none. The
    /// voxels travel to the GPU at the next frame; until then the old
    /// grading holds.
    pub fn set_lut(&mut self, lut: Option<(u32, Vec<f32>)>) {
        match lut {
            Some((size, voxels)) if size >= 2 => {
                self.lut_pending = Some((size, voxels));
            }
            _ => {
                // Back to the identity: colours pass through, and the
                // shader is told so.
                self.lut_pending = Some((2, identity_lut(2)));
                self.lut_size = 0;
            }
        }
    }

    /// Records the LUT upload when new voxels wait. Rebuilding the
    /// image when the edge size changes needs an idle device — a LUT
    /// changes on a settings click, never mid-animation.
    unsafe fn record_lut_upload(&mut self, cmd: vk::CommandBuffer) {
        let Some((size, voxels)) = self.lut_pending.take() else { return };
        let is_identity = self.lut_size == 0 && size == 2;
        let bytes = std::slice::from_raw_parts(
            voxels.as_ptr() as *const u8,
            voxels.len() * 4,
        );
        // A different edge size means a different image.
        let current_edge = if self.lut_initialized { self.lut_edge() } else { 0 };
        if current_edge != size {
            let _ = self.device.device_wait_idle();
            // Built BEFORE the old one is let go. The other order —
            // which is what stood here — hands the fields three dead
            // handles the moment the new image cannot be had, and every
            // later frame samples them.
            let Some((img, mem, view)) = create_lut_image(&self.device, &self.mem_props, size)
            else {
                eprintln!("nacelle-renderer: no GPU memory for a {size}-edge grading LUT; keeping the old one");
                return;
            };
            self.device.destroy_image_view(self.lut_view, None);
            self.device.destroy_image(self.lut_image, None);
            self.device.free_memory(self.lut_mem, None);
            self.lut_image = img;
            self.lut_mem = mem;
            self.lut_view = view;
            self.lut_initialized = false;
            let info = [vk::DescriptorImageInfo::default()
                .image_view(self.lut_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(self.lut_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&info)];
            self.device.update_descriptor_sets(&writes, &[]);
        }
        let Some((buf, mem, ptr)) = create_host_buffer(
            &self.device,
            &self.mem_props,
            bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
        ) else {
            eprintln!("nacelle-renderer: no staging memory for the grading LUT; it stays as it was");
            return;
        };
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        let to_dst = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(if self.lut_initialized {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            } else {
                vk::ImageLayout::UNDEFINED
            })
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(self.lut_image)
            .subresource_range(color_range());
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_dst],
        );
        let region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D { width: size, height: size, depth: size });
        self.device.cmd_copy_buffer_to_image(
            cmd,
            buf,
            self.lut_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
        let to_read = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.lut_image)
            .subresource_range(color_range());
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_read],
        );
        self.lut_initialized = true;
        self.lut_size = if is_identity { 0 } else { size };
        self.retired.push((buf, mem));
    }

    /// Edge size of the image currently bound as the LUT.
    fn lut_edge(&self) -> u32 {
        if self.lut_size == 0 { 2 } else { self.lut_size }
    }

    /// Registers an RGBA image of the given size. The handle is what
    /// [`nacelle::draw::DrawList::image`] takes; pixels arrive through
    /// [`Gfx::update_texture`].
    ///
    /// IT HAS CALLERS, and the `allow` below says nothing to the
    /// contrary — `dead_code` only ever looked inside this crate. The
    /// live one is `nacelle-desktop`'s `install_plate` (screen.rs),
    /// which registers the two theme plates, "backdrop" and "overlay",
    /// and re-registers them whenever a theme swap or a resize changes
    /// their size. A branch that reads the `allow` as "no callers" and
    /// changes the signature breaks that file; this note is here
    /// because one already did.
    ///
    /// `None` when the device has no memory for the image. A caller
    /// that asked for a wallpaper and did not get one has a picture to
    /// leave out; it does not have a desktop to end.
    #[allow(dead_code)]
    pub fn create_texture(&mut self, w: u32, h: u32) -> Option<nacelle::draw::ImageId> {
        // The band at the top of the id space is renderer instructions
        // (glass ranks, ADD_ATLAS), never a registered texture — handing
        // one out would silently alias a sentinel (r1 §0.1).
        assert!(
            self.next_tex < nacelle::draw::RESERVED_IMAGE_MIN.0,
            "texture ids exhausted the unreserved band"
        );
        unsafe {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D { width: w.max(1), height: h.max(1), depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = self.device.create_image(&info, None).unwrap();
            let req = self.device.get_image_memory_requirements(image);
            let Some(mem) = alloc_memory(
                &self.device,
                &self.mem_props,
                req,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ) else {
                self.device.destroy_image(image, None);
                eprintln!("nacelle-renderer: no GPU memory for a {w}x{h} image");
                return None;
            };
            self.device.bind_image_memory(image, mem, 0).unwrap();
            let view = self
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .subresource_range(color_range()),
                    None,
                )
                .unwrap();
            let layouts = [self.desc_layout];
            let desc_set = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.tex_pool)
                        .set_layouts(&layouts),
                )
                .unwrap()[0];
            let tex_info = [vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let samp_info =
                [vk::DescriptorImageInfo::default().sampler(self.sampler)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&tex_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&samp_info),
            ];
            self.device.update_descriptor_sets(&writes, &[]);
            let id = self.next_tex;
            self.next_tex += 1;
            self.textures.insert(
                id,
                Texture {
                    image,
                    mem,
                    view,
                    desc_set,
                    w: w.max(1),
                    h: h.max(1),
                    pending: None,
                    initialized: false,
                },
            );
            Some(nacelle::draw::ImageId(id))
        }
    }

    /// Hands the image new pixels: tightly packed RGBA, exactly
    /// width * height * 4 bytes. They reach the GPU at the start of
    /// the next frame; until then the old picture stays.
    #[allow(dead_code)]
    pub fn update_texture(&mut self, id: nacelle::draw::ImageId, rgba: &[u8]) {
        if let Some(tex) = self.textures.get_mut(&id.0) {
            if rgba.len() == (tex.w * tex.h * 4) as usize {
                tex.pending = Some(rgba.to_vec());
            } else {
                eprintln!(
                    "nacelle-desktop: texture {} expects {} bytes, got {}",
                    id.0,
                    tex.w * tex.h * 4,
                    rgba.len()
                );
            }
        }
    }

    /// Lets an image go. Waits for the device: a texture dies rarely
    /// and never mid-animation, so the stall is the simple answer to
    /// "is the GPU still reading it".
    #[allow(dead_code)]
    pub fn destroy_texture(&mut self, id: nacelle::draw::ImageId) {
        if let Some(tex) = self.textures.remove(&id.0) {
            unsafe {
                let _ = self.device.device_wait_idle();
                let _ = self
                    .device
                    .free_descriptor_sets(self.tex_pool, &[tex.desc_set]);
                self.device.destroy_image_view(tex.view, None);
                self.device.destroy_image(tex.image, None);
                self.device.free_memory(tex.mem, None);
            }
        }
    }

    /// Records the copies for every texture with pixels waiting. Each
    /// upload rides its own staging buffer, retired after the frame's
    /// fence — the shared staging buffer belongs to the atlas and may
    /// be busy with it in the same frame.
    unsafe fn record_texture_uploads(&mut self, cmd: vk::CommandBuffer) {
        let ids: Vec<u32> = self
            .textures
            .iter()
            .filter(|(_, t)| t.pending.is_some())
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let (data, image, w, h, initialized) = {
                let tex = self.textures.get_mut(&id).unwrap();
                (
                    tex.pending.take().unwrap(),
                    tex.image,
                    tex.w,
                    tex.h,
                    tex.initialized,
                )
            };
            let Some((buf, mem, ptr)) = create_host_buffer(
                &self.device,
                &self.mem_props,
                data.len() as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
            ) else {
                // The pixels go back on the queue: the next frame is
                // another chance, and the image keeps the picture it
                // had until one of them takes.
                if let Some(tex) = self.textures.get_mut(&id) {
                    tex.pending = Some(data);
                }
                eprintln!("nacelle-renderer: no staging memory for image {id}; upload deferred");
                continue;
            };
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            let to_dst = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(if initialized {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                })
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image)
                .subresource_range(color_range());
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
            let region = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            self.device.cmd_copy_buffer_to_image(
                cmd,
                buf,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
            let to_read = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image)
                .subresource_range(color_range());
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );
            self.textures.get_mut(&id).unwrap().initialized = true;
            self.retired.push((buf, mem));
        }
    }

    /// Writes the caller's dirty atlas rows into the staging buffer and
    /// merges their span into [`Self::pending_atlas_rows`]. The atlas
    /// travels by ROWS: `(pixels, first row, row count)` — a glyph-churn
    /// frame re-uploads a shelf, not four megabytes. The staging buffer is
    /// written at the same row offset the copy reads, so the region maths
    /// stays in one place, and staging contents persist across frames,
    /// which is what lets a span survive a frame that never records its
    /// copy. Caller must have waited the frame fence: the previous
    /// frame's copy may still be reading the staging buffer until then.
    unsafe fn stash_atlas_rows(&mut self, atlas: Option<(&[u8], u32, u32)>) {
        let Some((data, y0, rows)) = atlas else { return };
        if rows == 0 {
            return;
        }
        let row_bytes = ATLAS_W;
        let y0 = (y0 as usize).min(ATLAS_H);
        let rows = (rows as usize).min(ATLAS_H - y0);
        if rows == 0 {
            return;
        }
        let off = y0 * row_bytes;
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(off),
            self.staging_ptr.add(off),
            rows * row_bytes,
        );
        let (a0, a1) = (y0 as u32, (y0 + rows) as u32);
        self.pending_atlas_rows = Some(match self.pending_atlas_rows {
            Some((p0, p1)) => (p0.min(a0), p1.max(a1)),
            None => (a0, a1),
        });
    }

    unsafe fn record_atlas_upload(&self, cmd: vk::CommandBuffer, y0: u32, rows: u32) {
        let to_dst = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(if self.atlas_initialized {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            } else {
                vk::ImageLayout::UNDEFINED
            })
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.atlas_image)
            .subresource_range(color_range());
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_dst],
        );
        let region = vk::BufferImageCopy::default()
            .buffer_offset((y0 as u64) * ATLAS_W as u64)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: y0 as i32, z: 0 })
            .image_extent(vk::Extent3D {
                width: ATLAS_W as u32,
                height: rows.min(ATLAS_H as u32 - y0),
                depth: 1,
            });
        self.device.cmd_copy_buffer_to_image(
            cmd,
            self.staging_buf,
            self.atlas_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
        let to_read = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.atlas_image)
            .subresource_range(color_range());
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_read],
        );
    }
}

impl Drop for Gfx {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            // The last word before the pool goes: a run too short to
            // reach a report still prints one.
            if let Some(t) = self.timing.take() {
                t.report_final(self.extent.width, self.extent.height);
                t.destroy(&self.device);
            }
            let d = &self.device;
            d.destroy_fence(self.fence, None);
            d.destroy_semaphore(self.sem_image, None);
            d.destroy_semaphore(self.sem_render, None);
            d.destroy_command_pool(self.cmd_pool, None);
            d.destroy_buffer(self.vertex_buf, None);
            d.free_memory(self.vertex_mem, None);
            d.destroy_buffer(self.shapes_buf, None);
            d.free_memory(self.shapes_mem, None);
            d.destroy_descriptor_pool(self.shapes_pool, None);
            d.destroy_descriptor_set_layout(self.shapes_layout, None);
            d.destroy_buffer(self.staging_buf, None);
            d.free_memory(self.staging_mem, None);
            d.destroy_sampler(self.sampler, None);
            d.destroy_image_view(self.atlas_view, None);
            d.destroy_image(self.atlas_image, None);
            d.free_memory(self.atlas_mem, None);
            for (buf, mem) in self.retired.drain(..) {
                d.destroy_buffer(buf, None);
                d.free_memory(mem, None);
            }
            for (_, tex) in self.textures.drain() {
                d.destroy_image_view(tex.view, None);
                d.destroy_image(tex.image, None);
                d.free_memory(tex.mem, None);
            }
            d.destroy_image_view(self.lut_view, None);
            d.destroy_image(self.lut_image, None);
            d.free_memory(self.lut_mem, None);
            d.destroy_descriptor_set_layout(self.lut_layout, None);
            for t in self.blur_targets.drain(..) {
                d.destroy_framebuffer(t.fb, None);
                d.destroy_image_view(t.view, None);
                d.destroy_image(t.image, None);
                d.free_memory(t.mem, None);
            }
            d.destroy_render_pass(self.blur_pass, None);
            d.destroy_descriptor_pool(self.tex_pool, None);
            d.destroy_descriptor_pool(self.desc_pool, None);
            d.destroy_descriptor_set_layout(self.desc_layout, None);
            for p in self.pipes {
                d.destroy_pipeline(p, None);
            }
            d.destroy_pipeline_layout(self.pipeline_layout, None);
            for fb in self.framebuffers.drain(..) {
                d.destroy_framebuffer(fb, None);
            }
            for v in self.views.drain(..) {
                d.destroy_image_view(v, None);
            }
            d.destroy_render_pass(self.render_pass, None);
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            }
            d.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            // No `destroy_instance` here, and that is the whole point.
            // The instance lives in the process-wide [`VK`] static and
            // is never destroyed, so the two calls above are safe even
            // when a second screen is still running: the device and the
            // surface are this screen's alone, while the instance
            // outlives every `Gfx`. See [`VK`] for why tearing it down
            // per screen crashed the process.
        }
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

// ---- The run mapping, as pure functions so it is testable without a
// device. The renderer decides everything against the frame's actual
// state (blur depth, live targets), never against a theme token.

/// Whether a run handle is one of the glass instructions: the three
/// tessellated ranks, the three SHAPE_GLASS lanes of the vector core,
/// or the legacy BLUR_IMAGE. All seven live in the reserved band, but
/// the band holds non-glass instructions too — ADD_ATLAS must never
/// trigger the base-scene split.
///
/// The vector lanes belong here for one reason and it is the whole
/// reason this predicate exists: they SAMPLE the pyramid, so everything
/// before the first of them is the base scene. A frosted surface drawn
/// small enough, or drawn during a ride, emits no tessellated core at
/// all (f3 §3.3) — the band is then the only glass in the frame, and a
/// classifier that did not know it would leave the pyramid unwritten
/// and the frost with nothing to read.
fn is_glass(id: ImageId) -> bool {
    id == nacelle::draw::BLUR_IMAGE
        || id == nacelle::draw::GLASS_RANK_1
        || id == nacelle::draw::GLASS_RANK_2
        || id == nacelle::draw::GLASS_RANK_3
        || is_shape_glass(id)
}

/// Whether a handle is one of the vector core's frosted lanes — the
/// band of a frosted surface, drawn through `fs_shape_glass` with both
/// the record and the blurred scene in hand.
fn is_shape_glass(id: ImageId) -> bool {
    id == nacelle::draw::SHAPE_GLASS_1
        || id == nacelle::draw::SHAPE_GLASS_2
        || id == nacelle::draw::SHAPE_GLASS_3
}

/// The rank a glass handle asks for, on either lane. BLUR_IMAGE aliases
/// rank 2: the composite used to pick target 2 whenever the depth
/// allowed it, and the legacy handle must keep producing exactly that
/// picture.
fn glass_rank(id: ImageId) -> u8 {
    if id == nacelle::draw::GLASS_RANK_1 || id == nacelle::draw::SHAPE_GLASS_1 {
        1
    } else if id == nacelle::draw::GLASS_RANK_3 || id == nacelle::draw::SHAPE_GLASS_3 {
        3
    } else {
        2
    }
}

/// Which pyramid target a rank samples, against the depth the frame
/// actually wrote. A rank the pyramid did not reach falls to the
/// deepest target written, so glass never samples an image left in
/// UNDEFINED layout (r1 R3).
fn glass_target(rank: u8, blur_depth: u32) -> usize {
    match (rank, blur_depth) {
        (1, _) => 1,
        (2, d) if d >= 2 => 2,
        (3, 3) => 3,
        (3, 2) => 2,
        _ => 1,
    }
}

/// The downsample schedule per depth. Every entry renders `dst` from
/// `src` by linear resampling; the shrinking is the blur. At full depth
/// the list ends (2,3),(3,2): the quarter is smoothed from the eighth,
/// then the eighth is RE-derived from that smoothed quarter — without
/// the final step rank 3 would stretch raw 8x8 blocks while rank 2
/// shows a smoothed picture (r1's correction to Appendix B R3). The
/// extra pass covers 1/64 of the screen: under 0.01 ms at 1440p.
fn pyramid_steps(blur_depth: u32) -> &'static [(usize, usize)] {
    match blur_depth {
        1 => &[(1, 0)],
        2 => &[(1, 0), (2, 1)],
        _ => &[(1, 0), (2, 1), (3, 2), (2, 3), (3, 2)],
    }
}

/// What set 0 must hold for a run. Reserved ids with no instruction
/// here classify as images and die on the missing-texture fail-safe —
/// `create_texture` keeps the reserved band out of the texture map.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RunKind {
    Atlas,
    /// The atlas under the additive pipeline.
    Add,
    /// The vector core: fs_shape over the set-2 records.
    Shape,
    /// The vector core's ADDITIVE lane: the same fragment, the same
    /// records, SRC_ALPHA/ONE. A glow (f3 §2.6) and nothing else.
    ShapeAdd,
    /// The vector core's FROSTED band (f3 §3.3): fs_shape_glass over
    /// the set-2 records AND the pyramid target the run's rank resolves
    /// to. The one kind that reads both.
    ShapeGlass(usize),
    /// Glass, resolved to a pyramid target index.
    Glass(usize),
    Image(u32),
}

fn run_kind(image: Option<ImageId>, blur_depth: u32) -> RunKind {
    match image {
        None => RunKind::Atlas,
        Some(id) if id == nacelle::draw::ADD_ATLAS => RunKind::Add,
        Some(id) if id == nacelle::draw::SHAPE => RunKind::Shape,
        Some(id) if id == nacelle::draw::SHAPE_ADD => RunKind::ShapeAdd,
        Some(id) if is_shape_glass(id) => {
            RunKind::ShapeGlass(glass_target(glass_rank(id), blur_depth))
        }
        Some(id) if is_glass(id) => RunKind::Glass(glass_target(glass_rank(id), blur_depth)),
        Some(id) => RunKind::Image(id.0),
    }
}

/// One graphics pipeline of this renderer, BY NAME — and the index of
/// its handle in [`Gfx::pipes`] and in the array `create_pipeline`
/// builds, which are the same order because this enum is what orders
/// both.
///
/// It exists so that "which pipeline does this run draw with" can be
/// answered by a function a test can call. A `vk::Pipeline` is an
/// opaque device handle: nothing outside a live GPU can tell one from
/// another, so a branch that picks the wrong one is invisible to every
/// test in this crate — which is exactly what happened to the frosted
/// band's branch when it lived inside `record_runs`.
///
/// The discriminants are the array indices, and `create_pipeline` fills
/// its create-infos BY NAME at the same indices, so the numbers below
/// are free to move: handle and lookup move together. What is left
/// unguarded is one pairing per line there — which stage array each
/// create-info takes, and so which fragment entry point it compiles.
/// That one only a screen can answer, and it is a whole pipeline
/// wrong rather than one lane of one surface.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
enum Pipe {
    /// fs_main, normal blend: the atlas lane every glyph and solid fill
    /// rides.
    Atlas = 0,
    /// fs_image: the texture IS the colour. Also the pyramid's own
    /// downsample and upsample passes.
    Image = 1,
    /// fs_blur: the blurred scene, sampled by screen position.
    Blur = 2,
    /// fs_main under SRC_ALPHA/ONE colour, ZERO/ONE alpha: runs tagged
    /// ADD_ATLAS compose with light. Destination alpha stays untouched,
    /// which is what a passthrough swapchain will need (r1 R1/R8).
    Add = 3,
    /// fs_shape, normal blend: the vector core's lane. Dead code by
    /// intent until libnacelle's `render.vector` arms the SHAPE runs —
    /// nothing in the shipped picture emits them (f3 K1).
    Shape = 4,
    /// fs_shape_glass, normal blend: the same lane for a FROSTED band,
    /// the one pipeline that binds a pyramid target AND reads the shape
    /// records (f3 §3.3, K3b). Dead behind the same switch.
    ShapeGlass = 5,
    /// fs_shape under SRC_ALPHA/ONE colour, ZERO/ONE alpha: the vector
    /// core's glow (f3 §2.6). Same fragment, same stage array and same
    /// set as `Shape` — the ONE thing that differs is the blend, which
    /// is why it has to be a pipeline of its own and could not have
    /// been a bit in the record: blend state is fixed before the first
    /// fragment of a run is shaded. Its blend is `Add`'s, to the
    /// factor, so a glow composes with light exactly as the atlas glow
    /// it replaces did.
    ShapeAdd = 6,
}

impl Pipe {
    /// How many there are — the length of every array indexed by one.
    const N: usize = 7;
}

/// Which pipeline draws a run of this kind.
///
/// The frosted band is the whole reason this is a function: it binds
/// what glass binds and draws with what shape draws, so it is the one
/// kind whose set and whose pipeline come from different places. Sent
/// down `Blur` it would sample the pyramid with a fragment that reads
/// no record — the surface would lose its silhouette and paint its
/// whole quad, corners, margin and all.
fn pipe_of(kind: RunKind) -> Pipe {
    match kind {
        RunKind::Atlas => Pipe::Atlas,
        RunKind::Add => Pipe::Add,
        RunKind::Shape => Pipe::Shape,
        RunKind::ShapeAdd => Pipe::ShapeAdd,
        RunKind::ShapeGlass(_) => Pipe::ShapeGlass,
        RunKind::Glass(_) => Pipe::Blur,
        RunKind::Image(_) => Pipe::Image,
    }
}

/// A run's clip as a scissor `[x, y, w, h]` inside a `tw` x `th`
/// target: outward rounding, because a fractional panel edge must not
/// shave pixels it owns; clamped, because a negative offset is invalid
/// usage and an offset past the render area is rendering outside it;
/// never negative in size. None means the whole target. A clip that
/// misses the target entirely comes back zero-area — the caller skips
/// the draw instead of recording it.
fn scissor_for(clip: Option<[f32; 4]>, tw: u32, th: u32) -> [i32; 4] {
    match clip {
        None => [0, 0, tw as i32, th as i32],
        Some([x, y, w, h]) => {
            let x0 = (x.floor().max(0.0) as i32).min(tw as i32);
            let y0 = (y.floor().max(0.0) as i32).min(th as i32);
            let x1 = ((x + w).ceil() as i32).min(tw as i32).max(x0);
            let y1 = ((y + h).ceil() as i32).min(th as i32).max(y0);
            [x0, y0, x1 - x0, y1 - y0]
        }
    }
}

fn create_render_pass(device: &ash::Device, format: vk::Format) -> vk::RenderPass {
    let attachments = [vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];
    let color_refs = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs)];
    let deps = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
    unsafe {
        device
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses)
                    .dependencies(&deps),
                None,
            )
            .unwrap()
    }
}

/// The offscreen pass of the frosted-glass chain. Same format as the
/// swapchain — which is exactly what lets the ordinary pipelines draw
/// into it (render-pass compatibility is formats and sample counts) —
/// but it ends shader-readable instead of presentable, and its
/// dependencies fence the sampling that follows.
fn create_blur_pass(device: &ash::Device, format: vk::Format) -> vk::RenderPass {
    let attachments = [vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let color_refs = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs)];
    let deps = [
        // Whoever sampled this image last (the previous frame's
        // composite) must be done before it is overwritten.
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
        // And whoever samples it next waits for the write.
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags::SHADER_READ),
    ];
    unsafe {
        device
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses)
                    .dependencies(&deps),
                None,
            )
            .unwrap()
    }
}

/// The pipelines over the one vertex stream, sharing one layout. All
/// are compiled against the main render pass and drawn in the blur
/// pass too — render-pass compatibility is formats and sample counts,
/// and the two passes match.
struct Pipelines {
    layout: vk::PipelineLayout,
    /// The seven handles indexed by [`Pipe`] — what each one draws is
    /// written at the enum, once, rather than beside every field that
    /// would otherwise hold it. The create-infos below are filled in at
    /// the index their own name resolves to, so this array and every
    /// lookup against it stay in step by construction.
    by_pipe: [vk::Pipeline; Pipe::N],
}

fn create_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    desc_layout: vk::DescriptorSetLayout,
    lut_layout: vk::DescriptorSetLayout,
    shapes_layout: vk::DescriptorSetLayout,
) -> Pipelines {
    unsafe {
        // One SPIR-V module, one vertex stage, four fragment entry
        // points; the two additive pipelines reuse fs_main and fs_shape
        // and differ from their twins only in blend state.
        let spv = crate::shaders::compile();
        let shader_mod = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
            .unwrap();

        let vs_entry = CStr::from_bytes_with_nul(b"vs_main\0").unwrap();
        let fs_entry = CStr::from_bytes_with_nul(b"fs_main\0").unwrap();
        let fs_image_entry = CStr::from_bytes_with_nul(b"fs_image\0").unwrap();
        let fs_blur_entry = CStr::from_bytes_with_nul(b"fs_blur\0").unwrap();
        let fs_shape_entry = CStr::from_bytes_with_nul(b"fs_shape\0").unwrap();
        let fs_shape_glass_entry = CStr::from_bytes_with_nul(b"fs_shape_glass\0").unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader_mod)
                .name(vs_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader_mod)
                .name(fs_entry),
        ];
        let stages_image = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader_mod)
                .name(vs_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader_mod)
                .name(fs_image_entry),
        ];
        let stages_blur = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader_mod)
                .name(vs_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader_mod)
                .name(fs_blur_entry),
        ];
        let stages_shape = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader_mod)
                .name(vs_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader_mod)
                .name(fs_shape_entry),
        ];
        let stages_shape_glass = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader_mod)
                .name(vs_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader_mod)
                .name(fs_shape_glass_entry),
        ];

        let bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attrs = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(8),
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .binding(0)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(16),
            // The shape-record index (f3 D3). The stride above grows by
            // itself: it is size_of::<Vertex>().
            vk::VertexInputAttributeDescription::default()
                .location(3)
                .binding(0)
                .format(vk::Format::R32_UINT)
                .offset(32),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attrs);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let blend_add_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ZERO)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend_add = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&blend_add_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // 64 B (D0): 16 of scalars plus the mat3x3's three padded
        // columns — half the 128 B Vulkan guarantees, hairline_floor
        // and friends still have room.
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(64)];
        let set_layouts = [desc_layout, lut_layout, shapes_layout];
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )
            .unwrap();

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        let info_image = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages_image)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        let info_blur = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages_blur)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        let info_add = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend_add)
            .dynamic_state(&dynamic)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        let info_shape_glass = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages_shape_glass)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        // The glow's pipeline: `stages_shape` again, `blend_add` again
        // — the two arrays that already exist, paired for the first
        // time. Nothing new is compiled and nothing new is written.
        let info_shape_add = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages_shape)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend_add)
            .dynamic_state(&dynamic)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        let info_shape = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages_shape)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        // The create-infos go in at the INDEX THEIR NAME RESOLVES TO,
        // not in the order they happen to be written: `Gfx` looks a
        // pipeline up by the same name, so a discriminant moved in the
        // enum moves the handle and the lookup together and the picture
        // does not change. Nothing in this crate can tell two
        // `vk::Pipeline`s apart — the only defence against binding the
        // wrong one is not needing to check.
        let mut infos = [vk::GraphicsPipelineCreateInfo::default(); Pipe::N];
        infos[Pipe::Atlas as usize] = info;
        infos[Pipe::Image as usize] = info_image;
        infos[Pipe::Blur as usize] = info_blur;
        infos[Pipe::Add as usize] = info_add;
        infos[Pipe::Shape as usize] = info_shape;
        infos[Pipe::ShapeGlass as usize] = info_shape_glass;
        infos[Pipe::ShapeAdd as usize] = info_shape_add;
        let pipelines = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &infos, None)
            .expect("cannot create pipelines");

        device.destroy_shader_module(shader_mod, None);
        Pipelines {
            layout,
            by_pipe: pipelines
                .try_into()
                .expect("the device returned a pipeline per create-info"),
        }
    }
}

/// A 3D image for the grading LUT: RGBA float voxels, `size` an edge.
/// `None` when the device has no memory for it — the image handle is
/// destroyed on that path, so nothing is left dangling.
fn create_lut_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u32,
) -> Option<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    unsafe {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .extent(vk::Extent3D { width: size, height: size, depth: size })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = device.create_image(&info, None).unwrap();
        let req = device.get_image_memory_requirements(image);
        let Some(mem) = alloc_memory(device, mem_props, req, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        else {
            device.destroy_image(image, None);
            return None;
        };
        device.bind_image_memory(image, mem, 0).unwrap();
        let view = device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_3D)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .subresource_range(color_range()),
                None,
            )
            .unwrap();
        Some((image, mem, view))
    }
}

/// Identity voxels for an edge of `size`: each voxel is its own
/// coordinate, so sampling gives back the input colour exactly.
fn identity_lut(size: u32) -> Vec<f32> {
    let n = size as usize;
    let mut v = Vec::with_capacity(n * n * n * 4);
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let d = (n - 1).max(1) as f32;
                v.extend_from_slice(&[r as f32 / d, g as f32 / d, b as f32 / d, 1.0]);
            }
        }
    }
    v
}

/// Parses a .cube 3D LUT (the Adobe/Resolve text format): LUT_3D_SIZE
/// N followed by N^3 "r g b" lines, red fastest — the same order the
/// 3D image wants. Returns the edge size and RGBA voxels.
pub fn parse_cube(text: &str) -> Option<(u32, Vec<f32>)> {
    let mut size = 0usize;
    let mut voxels: Vec<f32> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let first = it.next()?;
        match first {
            "LUT_3D_SIZE" => {
                size = it.next()?.parse().ok()?;
                if !(2..=129).contains(&size) {
                    return None;
                }
                voxels.reserve(size * size * size * 4);
            }
            "TITLE" | "DOMAIN_MIN" | "DOMAIN_MAX" | "LUT_1D_SIZE" => {
                // 1D LUTs are a different animal; a file leading with
                // one is refused rather than misread.
                if first == "LUT_1D_SIZE" {
                    return None;
                }
            }
            _ => {
                let r: f32 = first.parse().ok()?;
                let g: f32 = it.next()?.parse().ok()?;
                let b: f32 = it.next()?.parse().ok()?;
                voxels.extend_from_slice(&[r, g, b, 1.0]);
            }
        }
    }
    if size >= 2 && voxels.len() == size * size * size * 4 {
        Some((size as u32, voxels))
    } else {
        None
    }
}

/// Picks the surface format nearest the preferred bit depth, falling
/// back towards eight. Twelve maps to the sixteen-bit float formats —
/// Vulkan has no twelve-bit swapchain — and whether the wire then
/// carries 12 bpc is between the compositor and the display.
fn pick_format(formats: &[vk::SurfaceFormatKHR], depth: u32) -> vk::SurfaceFormatKHR {
    let tiers: &[&[vk::Format]] = match depth {
        16 | 12 => &[
            &[vk::Format::R16G16B16A16_SFLOAT],
            &[
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::Format::A2R10G10B10_UNORM_PACK32,
            ],
            &[vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM],
        ],
        10 => &[
            &[
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::Format::A2R10G10B10_UNORM_PACK32,
            ],
            &[vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM],
        ],
        _ => &[&[vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM]],
    };
    for tier in tiers {
        if let Some(f) = formats.iter().copied().find(|f| tier.contains(&f.format)) {
            return f;
        }
    }
    formats[0]
}

/// How many bits per colour channel a swapchain format really carries.
///
/// The counterpart of [`pick_format`], and the reason it exists: that
/// function ASKS, and a surface is free to answer with less. A settings
/// page which only ever showed the number that was asked for would say
/// "16" over a picture the driver is handing back in eight — and a user
/// looking for the difference between the two would find none, with
/// nothing anywhere to tell them why.
///
/// Sixteen-bit float is reported as sixteen even though the twelve the
/// page may have asked for rides in it; twelve is a wish about the wire
/// and this is a statement about the buffer.
fn format_bits(f: vk::Format) -> u32 {
    match f {
        vk::Format::R16G16B16A16_SFLOAT | vk::Format::R16G16B16A16_UNORM => 16,
        vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32 => 10,
        _ => 8,
    }
}

/// The first memory type an object may use that also has every flag
/// asked for, or `None` when this device offers no such type.
///
/// `None` and not a panic: the caller is always in a position to say
/// something better than "the process is over". At build time that is a
/// [`GfxError`] the desktop can print; at frame time it is one feature
/// switched off — a texture not registered, a blur chain not built —
/// while the picture goes on being drawn.
fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        type_bits & (1 << i) != 0
            && props.memory_types[i as usize].property_flags.contains(flags)
    })
}

/// Memory for one object, or `None` when the device has no type for it
/// or refuses the allocation.
fn alloc_memory(
    device: &ash::Device,
    props: &vk::PhysicalDeviceMemoryProperties,
    req: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
) -> Option<vk::DeviceMemory> {
    let ty = find_memory_type(props, req.memory_type_bits, flags)?;
    unsafe {
        device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty),
                None,
            )
            .ok()
    }
}

/// A host-visible, persistently mapped buffer, or `None` when the
/// device has no memory for it. The buffer handle is destroyed on the
/// way out of the failing path, so a `None` leaves nothing behind.
fn create_host_buffer(
    device: &ash::Device,
    props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Option<(vk::Buffer, vk::DeviceMemory, *mut u8)> {
    unsafe {
        let buf = device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .ok()?;
        let req = device.get_buffer_memory_requirements(buf);
        let Some(mem) = alloc_memory(
            device,
            props,
            req,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) else {
            device.destroy_buffer(buf, None);
            return None;
        };
        device.bind_buffer_memory(buf, mem, 0).unwrap();
        let ptr = device
            .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .unwrap() as *mut u8;
        Some((buf, mem, ptr))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        fault_of, find_memory_type, format_bits, glass_rank, glass_target, is_glass, parse_cube,
        pick_format, pipe_of, pyramid_steps, run_kind, scissor_for, vk, Fault, Pipe, RunKind,
    };
    use nacelle::draw::{
        ImageId, ADD_ATLAS, BLUR_IMAGE, GLASS_RANK_1, GLASS_RANK_2, GLASS_RANK_3, SHAPE,
        SHAPE_ADD, SHAPE_GLASS_1, SHAPE_GLASS_2, SHAPE_GLASS_3,
    };

    /// The base-scene split triggers on glass and only on glass: the
    /// three ranks and the legacy handle, never the additive
    /// instruction, never a free reserved slot, never a real texture.
    #[test]
    fn glass_is_the_ranks_and_the_legacy_handle_and_nothing_else() {
        for id in [BLUR_IMAGE, GLASS_RANK_1, GLASS_RANK_2, GLASS_RANK_3] {
            assert!(is_glass(id), "{id:?} must count as glass");
        }
        // The vector core's frosted lanes count too, and they have to:
        // a small frosted surface, or one drawn during a ride, emits no
        // tessellated core at all (f3 §3.3) — its band is then the only
        // glass in the frame, and a split that did not fire would leave
        // the pyramid unwritten under a fragment that samples it.
        for id in [SHAPE_GLASS_1, SHAPE_GLASS_2, SHAPE_GLASS_3] {
            assert!(is_glass(id), "{id:?} samples the pyramid and must split the scene");
        }
        assert!(!is_glass(ADD_ATLAS));
        assert!(!is_glass(SHAPE), "the vector lane covers, it is not glass");
        assert!(!is_glass(ImageId(u32::MAX - 9)));
        assert!(!is_glass(ImageId(0)));
        assert!(!is_glass(ImageId(7)));
    }

    /// **Which fragment draws a run.** This decision used to sit in the
    /// middle of `record_runs`, between a scissor and a bind, where a
    /// test could not reach it at all — a `vk::Pipeline` is an opaque
    /// device handle, so a branch that picked the wrong one was correct
    /// as far as everything in this crate could tell, and wrong on the
    /// screen. Now the decision is a name and this is the test of it.
    ///
    /// The frosted band is the case that made it worth moving. It binds
    /// what glass binds and reads what a shape reads, so it is the one
    /// kind whose descriptor set and whose pipeline come from different
    /// places — and sent down `Blur` it would sample the pyramid with a
    /// fragment that reads no record: the surface would lose its
    /// silhouette and paint its whole quad, corners, margin and all.
    #[test]
    fn every_run_kind_names_the_fragment_that_can_draw_it() {
        assert_eq!(pipe_of(RunKind::Atlas), Pipe::Atlas);
        assert_eq!(pipe_of(RunKind::Add), Pipe::Add);
        assert_eq!(pipe_of(RunKind::Shape), Pipe::Shape);
        assert_eq!(pipe_of(RunKind::ShapeAdd), Pipe::ShapeAdd);
        assert_eq!(pipe_of(RunKind::Image(7)), Pipe::Image);
        for t in 0..3 {
            assert_eq!(pipe_of(RunKind::Glass(t)), Pipe::Blur);
            assert_eq!(pipe_of(RunKind::ShapeGlass(t)), Pipe::ShapeGlass);
        }
        // …and from the HANDLE the toolkit actually writes, which is
        // the whole road: a rank's band ends at the frosted fragment
        // and its core at the plain blur, from the same frame.
        assert_eq!(pipe_of(run_kind(Some(SHAPE_GLASS_2), 3)), Pipe::ShapeGlass);
        assert_eq!(pipe_of(run_kind(Some(GLASS_RANK_2), 3)), Pipe::Blur);
        assert_eq!(pipe_of(run_kind(Some(SHAPE), 3)), Pipe::Shape);
        // The glow's lane, and the pairing that makes it one: the same
        // fragment as `Shape`, a different blend. Sent down `Shape` it
        // would draw a correct glow that COVERS instead of lighting —
        // a milky rectangle over whatever it was meant to brighten.
        assert_eq!(pipe_of(run_kind(Some(SHAPE_ADD), 3)), Pipe::ShapeAdd);
        assert_ne!(Pipe::ShapeAdd as usize, Pipe::Shape as usize);
        assert_eq!(pipe_of(run_kind(None, 3)), Pipe::Atlas);
        // The names index an array, so two of them sharing a slot would
        // bind one pipeline for two kinds — silently, and only on the
        // lane that lost.
        let all = [
            Pipe::Atlas,
            Pipe::Image,
            Pipe::Blur,
            Pipe::Add,
            Pipe::Shape,
            Pipe::ShapeGlass,
            Pipe::ShapeAdd,
        ];
        let mut slots: Vec<usize> = all.iter().map(|p| *p as usize).collect();
        slots.sort_unstable();
        assert_eq!(slots, (0..Pipe::N).collect::<Vec<_>>());
    }

    /// A frosted surface is drawn in two pieces — the core through
    /// `GLASS_RANK_n`, the band through `SHAPE_GLASS_n` — and they must
    /// land on the SAME pyramid target at every depth. A mismatch would
    /// not crash and would not fail to draw: it would put one blur
    /// inside the surface and another around its rim, which is the
    /// stair-step defect K3b removed, wearing a different hat.
    #[test]
    fn the_band_and_the_core_of_one_rank_sample_one_target() {
        for depth in 1..=3u32 {
            for (tess, field) in [
                (GLASS_RANK_1, SHAPE_GLASS_1),
                (GLASS_RANK_2, SHAPE_GLASS_2),
                (GLASS_RANK_3, SHAPE_GLASS_3),
            ] {
                assert_eq!(glass_rank(tess), glass_rank(field), "{tess:?} vs {field:?}");
                assert_eq!(
                    run_kind(Some(field), depth),
                    RunKind::ShapeGlass(glass_target(glass_rank(tess), depth)),
                    "depth {depth}: the band left its core behind"
                );
            }
        }
    }

    /// rank 1 -> target 1 always; rank 2 -> 2 when written, else 1;
    /// rank 3 -> the deepest target the pyramid wrote.
    #[test]
    fn the_rank_fallback_table_is_exact() {
        assert_eq!(glass_target(1, 1), 1);
        assert_eq!(glass_target(1, 2), 1);
        assert_eq!(glass_target(1, 3), 1);
        assert_eq!(glass_target(2, 1), 1);
        assert_eq!(glass_target(2, 2), 2);
        assert_eq!(glass_target(2, 3), 2);
        assert_eq!(glass_target(3, 1), 1);
        assert_eq!(glass_target(3, 2), 2);
        assert_eq!(glass_target(3, 3), 3);
    }

    /// Whatever the mapping says a rank samples, the schedule must have
    /// written — sampling an unwritten target is sampling UNDEFINED
    /// layout, the exact bug the fallback table exists to prevent.
    #[test]
    fn every_rank_samples_a_target_the_pyramid_wrote() {
        for depth in 1..=3u32 {
            let written: Vec<usize> =
                pyramid_steps(depth).iter().map(|&(dst, _)| dst).collect();
            for rank in 1..=3u8 {
                let t = glass_target(rank, depth);
                assert!(
                    written.contains(&t),
                    "rank {rank} at depth {depth} maps to target {t}, never written"
                );
            }
        }
    }

    /// The composite used to pick target 1 at depth 1 and target 2
    /// otherwise; the legacy BLUR_IMAGE must keep doing precisely that.
    #[test]
    fn blur_image_still_means_todays_picture() {
        assert_eq!(glass_rank(BLUR_IMAGE), 2);
        for depth in 1..=3u32 {
            let legacy = if depth == 1 { 1 } else { 2 };
            assert_eq!(glass_target(glass_rank(BLUR_IMAGE), depth), legacy);
        }
    }

    /// At full depth the eighth is re-derived from the SMOOTHED quarter:
    /// the last write to target 3 comes after the (2,3) up-step and
    /// reads target 2 — otherwise rank 3 is blockier than rank 2.
    #[test]
    fn at_depth_three_the_eighth_is_rewritten_from_the_smoothed_quarter() {
        let steps = pyramid_steps(3);
        assert_eq!(steps, &[(1, 0), (2, 1), (3, 2), (2, 3), (3, 2)][..]);
        let last_w2 = steps.iter().rposition(|&(d, _)| d == 2).unwrap();
        let last_w3 = steps.iter().rposition(|&(d, _)| d == 3).unwrap();
        assert!(last_w3 > last_w2);
        assert_eq!(steps[last_w3], (3, 2));
        // The shallow depths are untouched — exactly yesterday's lists.
        assert_eq!(pyramid_steps(1), &[(1, 0)][..]);
        assert_eq!(pyramid_steps(2), &[(1, 0), (2, 1)][..]);
    }

    /// One classifier decides what set 0 holds. A reserved id with no
    /// instruction here classifies as an image and dies on the
    /// missing-texture fail-safe (the create_texture guard keeps the
    /// band out of the texture map), instead of vanishing silently
    /// somewhere else.
    #[test]
    fn runs_bind_by_kind() {
        assert_eq!(run_kind(None, 3), RunKind::Atlas);
        assert_eq!(run_kind(Some(ADD_ATLAS), 3), RunKind::Add);
        assert_eq!(run_kind(Some(SHAPE), 3), RunKind::Shape);
        assert_eq!(run_kind(Some(SHAPE), 1), RunKind::Shape);
        assert_eq!(run_kind(Some(SHAPE_ADD), 3), RunKind::ShapeAdd);
        assert_eq!(run_kind(Some(BLUR_IMAGE), 3), RunKind::Glass(2));
        assert_eq!(run_kind(Some(BLUR_IMAGE), 1), RunKind::Glass(1));
        assert_eq!(run_kind(Some(GLASS_RANK_1), 3), RunKind::Glass(1));
        assert_eq!(run_kind(Some(GLASS_RANK_2), 3), RunKind::Glass(2));
        assert_eq!(run_kind(Some(GLASS_RANK_3), 3), RunKind::Glass(3));
        assert_eq!(run_kind(Some(GLASS_RANK_3), 1), RunKind::Glass(1));
        // The frosted band is its OWN kind: it binds what glass binds
        // and draws with the pipeline that also reads set 2. Classified
        // as plain Glass it would draw through `fs_blur`, which reads
        // no record — the surface would lose its silhouette and paint
        // its whole quad, corners and all.
        assert_eq!(run_kind(Some(SHAPE_GLASS_1), 3), RunKind::ShapeGlass(1));
        assert_eq!(run_kind(Some(SHAPE_GLASS_2), 3), RunKind::ShapeGlass(2));
        assert_eq!(run_kind(Some(SHAPE_GLASS_3), 3), RunKind::ShapeGlass(3));
        assert_eq!(run_kind(Some(SHAPE_GLASS_3), 1), RunKind::ShapeGlass(1));
        assert_eq!(run_kind(Some(ImageId(7)), 3), RunKind::Image(7));
        // The reserved band still holds unclaimed handles, and one of
        // them still has to classify as an image and die on the
        // missing-texture fail-safe. `u32::MAX - 9` used to be the
        // spare; the glow took it, so the assertion moves down one.
        assert_eq!(
            run_kind(Some(ImageId(u32::MAX - 10)), 3),
            RunKind::Image(u32::MAX - 10)
        );
    }

    /// The scissor a clip maps to stays inside its target, rounds
    /// outward, and degenerates to zero area instead of going negative
    /// or leaking offsets past the framebuffer.
    #[test]
    fn the_scissor_stays_inside_its_target() {
        // No clip is the whole target.
        assert_eq!(scissor_for(None, 800, 600), [0, 0, 800, 600]);
        // Fractional rects round OUTWARD — a clipped panel must not
        // shave its own edge pixels.
        assert_eq!(
            scissor_for(Some([10.4, 10.6, 20.2, 20.2]), 800, 600),
            [10, 10, 21, 21]
        );
        // A negative origin clamps to the target's edge.
        assert_eq!(
            scissor_for(Some([-5.0, -7.0, 20.0, 20.0]), 800, 600),
            [0, 0, 15, 13]
        );
        // Past the far edge clamps to the target.
        assert_eq!(
            scissor_for(Some([790.0, 590.0, 50.0, 50.0]), 800, 600),
            [790, 590, 10, 10]
        );
        // Entirely off-target: zero area, offset still inside.
        assert_eq!(
            scissor_for(Some([900.0, 700.0, 40.0, 40.0]), 800, 600),
            [800, 600, 0, 0]
        );
        // An empty clip (a fully clipped subtree) is zero area too.
        let empty = scissor_for(Some([100.0, 100.0, 0.0, 0.0]), 800, 600);
        assert_eq!((empty[2], empty[3]), (0, 0));
    }

    // No unit test guards the shared instance, and one was removed rather
    // than kept: it asserted `OnceLock<Vk>: Sync`, which the declaration of
    // `static VK` already requires, so it could not fail on its own and its
    // comment claimed otherwise. A test that cannot fail is worse than no
    // test — it reports a proof that was never performed.
    //
    // The bug is a SIGSEGV on process exit with two screens open. Catching
    // it needs two real windows on a real GPU; it was verified by running
    // `nacelle-desktop --desktop` on a two-monitor desktop, three times,
    // exit status 0 each time, against the same build that crashed before.

    #[test]
    fn a_cube_file_parses_and_a_broken_one_does_not() {
        let ok = "# comment\nTITLE \"t\"\nLUT_3D_SIZE 2\n\
                  0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let (size, voxels) = parse_cube(ok).expect("a well-formed cube");
        assert_eq!(size, 2);
        assert_eq!(voxels.len(), 2 * 2 * 2 * 4);
        // Red fastest: the second voxel is pure red.
        assert_eq!(&voxels[4..8], &[1.0, 0.0, 0.0, 1.0]);

        // Too few rows for the declared size.
        assert!(parse_cube("LUT_3D_SIZE 2\n0 0 0\n").is_none());
        // A 1D LUT is refused, not misread.
        assert!(parse_cube("LUT_1D_SIZE 2\n0 0 0\n1 1 1\n").is_none());
        // Rubbish is rubbish.
        assert!(parse_cube("not a lut at all").is_none());
    }

    /// A depth chosen in the settings window asks for a DIFFERENT
    /// surface format, and what comes back is reported as the depth it
    /// really is.
    ///
    /// Two halves of one honesty. The first is the request: a swapchain
    /// rebuild only happens because `pick_format` answered something
    /// other than the format in force, so a preference that mapped to
    /// the same format for every number would be a control with nothing
    /// behind it. The second is the answer: a surface offering nothing
    /// but eight bits gives eight whatever was asked, and the number the
    /// page shows has to be the one in the buffer — otherwise the window
    /// says "16" over a picture that is not sixteen and the user hunts
    /// for a difference that was never rendered.
    ///
    /// What this CANNOT check without a device: that the driver then
    /// hands those bits to the wire. Twelve rides in a sixteen-bit float
    /// buffer and what the cable carries is between the compositor and
    /// the display; nothing in this process can see it.
    #[test]
    fn a_depth_asks_for_a_format_and_the_answer_is_reported_as_it_is() {
        let srgb = vk::ColorSpaceKHR::SRGB_NONLINEAR;
        let sf = |format| vk::SurfaceFormatKHR { format, color_space: srgb };

        // A surface with the whole ladder on it: every step is reachable
        // and every step is a different format, which is what makes the
        // rebuild fire.
        let rich = [
            sf(vk::Format::B8G8R8A8_UNORM),
            sf(vk::Format::A2B10G10R10_UNORM_PACK32),
            sf(vk::Format::R16G16B16A16_SFLOAT),
        ];
        assert_eq!(format_bits(pick_format(&rich, 8).format), 8);
        assert_eq!(format_bits(pick_format(&rich, 10).format), 10);
        assert_eq!(format_bits(pick_format(&rich, 16).format), 16);
        // Twelve has no swapchain of its own and rides the float one.
        assert_eq!(format_bits(pick_format(&rich, 12).format), 16);
        assert_ne!(
            pick_format(&rich, 8).format,
            pick_format(&rich, 10).format,
            "two depths that picked one format would never rebuild anything"
        );

        // And the honest half: a surface that offers eight and nothing
        // else answers eight to every wish.
        let poor = [sf(vk::Format::B8G8R8A8_UNORM)];
        for asked in [8, 10, 12, 16] {
            assert_eq!(
                format_bits(pick_format(&poor, asked).format),
                8,
                "asked for {asked} bits over an eight-bit surface and the \
                 number reported was not the one in the buffer"
            );
        }
    }

    /// **Unplugging a monitor must not end the desktop.**
    ///
    /// A surface dies for reasons that are nobody's fault and nobody's
    /// bug: a cable pulled, a compositor restarted, a session switched.
    /// Vulkan says so with `ERROR_SURFACE_LOST_KHR` out of whichever
    /// call came next — `acquire_next_image`, `queue_present`, or the
    /// capabilities query inside the swapchain rebuild — and every one
    /// of those three sites used to hand it to a `panic!` or an
    /// `unwrap`. One unplugged screen, no desktop.
    ///
    /// The two recoverable answers are not interchangeable, and the
    /// asymmetry is the point of the second half of this test: a
    /// swapchain can be rebuilt without touching the surface, but a
    /// surface cannot be rebuilt without also rebuilding the swapchain
    /// that names it. A `Fault::Surface` that asked for only the first
    /// repair would leave a live swapchain pointing at a dead surface
    /// — and the frame after would fail in exactly the same way, for
    /// ever.
    #[test]
    fn a_lost_surface_is_a_repair_and_only_the_unknown_is_fatal() {
        assert_eq!(fault_of(vk::Result::ERROR_SURFACE_LOST_KHR), Fault::Surface);
        assert_eq!(fault_of(vk::Result::ERROR_OUT_OF_DATE_KHR), Fault::Swapchain);
        // A driver may hand SUBOPTIMAL back as an error rather than in
        // the success value; a renderer that died of a merely stretched
        // picture would be the same bug in a smaller coat.
        assert_eq!(fault_of(vk::Result::SUBOPTIMAL_KHR), Fault::Swapchain);

        // Nothing else is guessed at. A lost device or an exhausted
        // heap is not a thing this renderer knows how to repair, and
        // pretending otherwise would spin the frame loop for ever.
        for e in [
            vk::Result::ERROR_DEVICE_LOST,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            vk::Result::ERROR_INITIALIZATION_FAILED,
        ] {
            assert_eq!(fault_of(e), Fault::Fatal, "{e:?} was given a repair it has none of");
        }

        // The ordering, stated as the two questions the frame asks.
        assert!(Fault::Surface.rebuilds_surface());
        assert!(
            Fault::Surface.rebuilds_swapchain(),
            "a swapchain over a surface that was just replaced is a dead swapchain"
        );
        assert!(!Fault::Swapchain.rebuilds_surface(), "a resize costs no surface");
        assert!(Fault::Swapchain.rebuilds_swapchain());
        assert!(!Fault::Surface.is_fatal());
        assert!(!Fault::Swapchain.is_fatal());
        assert!(Fault::Fatal.is_fatal());
        assert!(
            !Fault::Fatal.rebuilds_surface() && !Fault::Fatal.rebuilds_swapchain(),
            "a fault with no repair must not also ask for one"
        );
    }

    /// **A memory type that does not exist is an answer, not an
    /// ending.**
    ///
    /// The search used to end the process on its own authority, from
    /// the bottom of a helper that has no idea what it was allocating
    /// for. Now it says `None` and the caller decides: a [`GfxError`]
    /// before the first frame, one feature switched off after it.
    ///
    /// The positive halves matter as much. `type_bits` is a MASK from
    /// `vkGetImageMemoryRequirements` — an allocator that ignored it
    /// would hand back a type the object may not legally use — and the
    /// flags are a SUBSET test, not equality: a heap with more
    /// properties than were asked for is still a heap that will do.
    #[test]
    fn a_missing_memory_type_answers_none_instead_of_ending_the_process() {
        use vk::MemoryPropertyFlags as F;
        let ty = |flags| vk::MemoryType { property_flags: flags, heap_index: 0 };
        let mut props = vk::PhysicalDeviceMemoryProperties::default();
        props.memory_type_count = 3;
        props.memory_types[0] = ty(F::DEVICE_LOCAL);
        props.memory_types[1] = ty(F::HOST_VISIBLE);
        props.memory_types[2] = ty(F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED);

        // The case that used to be a panic: this device has no
        // host-coherent type the object is allowed to use.
        assert_eq!(find_memory_type(&props, 0b011, F::HOST_COHERENT), None);
        // …and the same when the object may use nothing at all.
        assert_eq!(find_memory_type(&props, 0, F::DEVICE_LOCAL), None);
        // …and when the device reports no types whatsoever, which is
        // what a software rasteriser stub looks like from here.
        let empty = vk::PhysicalDeviceMemoryProperties::default();
        assert_eq!(find_memory_type(&empty, u32::MAX, F::DEVICE_LOCAL), None);

        // The mask is obeyed: type 0 fits the flags and is forbidden.
        assert_eq!(find_memory_type(&props, 0b110, F::DEVICE_LOCAL), None);
        assert_eq!(find_memory_type(&props, 0b111, F::DEVICE_LOCAL), Some(0));
        // Extra properties do not disqualify a type.
        assert_eq!(
            find_memory_type(&props, 0b111, F::HOST_VISIBLE | F::HOST_COHERENT),
            Some(2)
        );
        // The FIRST match wins, and drivers order these best-first.
        assert_eq!(find_memory_type(&props, 0b111, F::HOST_VISIBLE), Some(1));
        // Nothing past `memory_type_count` is read, whatever the array
        // happens to hold — the array is 32 long on every device.
        let mut short = props;
        short.memory_type_count = 1;
        assert_eq!(find_memory_type(&short, 0b111, F::HOST_VISIBLE), None);
    }

    /// **A construction that fails after the device exists must have
    /// something to give the device back to.**
    ///
    /// Turning the constructor's panics into errors moved the fault
    /// instead of repairing it. A `?` out of the middle of a
    /// constructor drops NOTHING — the struct whose `Drop` destroys
    /// the device has not been built yet — so every one of those exits
    /// walked out on a `VkDevice`, a `VkSurfaceKHR` and everything
    /// made in between, for the life of the process. Panics had hidden
    /// that: the process ended and the kernel swept up. The whole
    /// point of the change is a caller that goes ON, and a desktop that
    /// skips one unusable monitor per plug event would have leaked a
    /// device per event.
    ///
    /// The repair is structural, so the guard is too: the fallible
    /// steps live in `furnish`, which fills an ALREADY-BUILT `Gfx`, and
    /// the only exits left in `new` itself are the three before the
    /// device exists — each handing the surface back by name, because
    /// at that moment nothing else can.
    ///
    /// This reads the source rather than running the constructor
    /// because running it needs a GPU, and the property is about which
    /// LINE a failure leaves from. Checked red against the code it
    /// replaces: on the old `new`, the seven `?` exits between the
    /// device and the struct trip the first assertion and the absent
    /// named exits trip the second.
    #[test]
    fn no_construction_step_can_fail_before_there_is_a_drop_to_catch_it() {
        let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gfx.rs"))
            .expect("this file");
        let start = src.find("    pub fn new(").expect("Gfx::new");
        let struct_at = src[start..]
            .find("let mut gfx = Gfx {")
            .expect("Gfx::new builds a Gfx")
            + start;
        let device_at = src[start..]
            .find("let queue = device.get_device_queue")
            .expect("Gfx::new creates a logical device")
            + start;
        assert!(
            device_at < struct_at,
            "the device is made after the struct, and this test no longer knows what it is reading"
        );

        // Nothing between the device and the struct may fail by `?`:
        // there is no `Drop` yet, so such an exit is a leak by
        // construction.
        let between = &src[device_at..struct_at];
        for leaky in [".map_err(GfxError::", ".ok_or(GfxError::", "?;"] {
            assert!(
                !between.contains(leaky),
                "`{leaky}` sits between the logical device and the struct whose Drop frees it; \
                 that exit leaks a VkDevice and a VkSurfaceKHR. Move the step into `furnish`."
            );
        }

        // And the exits that come BEFORE the struct hand the surface
        // back themselves, because at that point nothing else will.
        let head: Vec<&str> = src[start..struct_at].lines().collect();
        let mut named = 0;
        for (i, line) in head.iter().enumerate() {
            if line.contains("return Err(GfxError::") {
                named += 1;
                assert!(
                    head[..i]
                        .iter()
                        .rev()
                        .take(3)
                        .any(|l| l.contains("destroy_surface")),
                    "an exit from Gfx::new walks out on the surface: {}",
                    line.trim()
                );
            }
        }
        assert!(
            named > 0,
            "Gfx::new has no named exit at all — either the surface is created later than this \
             test believes, or the guard is reading nothing and passing"
        );
    }

    /// **The number in the package description is the number of
    /// pipelines the device is actually asked for.**
    ///
    /// Both the README and the manifest said "two pipelines" long after
    /// there were seven, and the module header said "six" — it had
    /// counted every lane except the frosted band. Prose about a
    /// number drifts silently, so the number gets a reader: [`Pipe::N`]
    /// is what `create_graphics_pipelines` is handed, and this is what
    /// makes the sentence answer to it.
    #[test]
    fn the_advertised_pipeline_count_is_the_one_the_device_gets() {
        const NAMES: [&str; 11] = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        ];
        let word = NAMES[Pipe::N];

        let manifest = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
            .expect("the crate's own manifest");
        let desc = manifest
            .lines()
            .find(|l| l.trim_start().starts_with("description"))
            .expect("the manifest describes the crate");
        assert!(
            desc.contains(&format!("{word} pipelines")),
            "the manifest advertises a pipeline count that is not {word}: {desc}"
        );

        let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))
            .expect("the crate's own README");
        assert!(
            readme.contains(&format!("{word} pipelines")),
            "the README advertises a pipeline count that is not {word}"
        );

        // The THIRD sentence about the same number, and the one that
        // had actually drifted: this file's own header. Leaving it out
        // of the guard was the flaw in the first version of this test —
        // it tied down the two sentences that were wrong and left free
        // the one that had already gone wrong once. Case-folded because
        // the header shouts the number.
        let header = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gfx.rs"))
            .expect("this file")
            .lines()
            .take_while(|l| l.starts_with("//!"))
            .collect::<Vec<_>>()
            .join("\n")
            .to_lowercase();
        assert!(
            !header.is_empty(),
            "the module header is gone, and with it the sentence this guards"
        );
        assert!(
            header.contains(&format!("{word} pipelines")),
            "the header of gfx.rs advertises a pipeline count that is not {word}"
        );

        // The sibling list is part of the same drift. `nacelle-widgets`
        // was renamed to `nacelle-addons` and the link went on pointing
        // at a repository that does not exist.
        assert!(
            readme.contains("nacelle-addons"),
            "the README does not name the addons repository"
        );
        assert!(
            !readme.contains("nacelle-widgets"),
            "the README still links a repository that was renamed away"
        );
    }
}
