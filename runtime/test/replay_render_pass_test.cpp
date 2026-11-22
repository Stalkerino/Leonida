// Synthetic BeginRenderPass/EndRenderPass replay check: no Vulkan loader, GPU, shaders or game data. The mock
// captures the VkRenderingInfo handed to vkCmdBeginRendering and checks load/store ops, clear values, raster-bound
// colour attachment order, aspect handling, the pass-snapshot draw path, scope gates and the
// unresolved-resource gate (docs/d3d12_vulkan.md section 4.3).
#include "../src/vulkan_replay.h"
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
struct Captured {
    int begins = 0, ends = 0, draws = 0, barriers = 0, dispatches = 0;
    int barriers_inside = 0, dispatches_inside = 0;
    bool scope_open = false;
    std::uint32_t colors = 0;
    VkAttachmentLoadOp color_load[8]{}, depth_load{}, stencil_load{};
    VkAttachmentStoreOp color_store[8]{}, depth_store{}, stencil_store{};
    VkClearColorValue color_clear[8]{};
    VkImageView color_view[8]{};
    bool depth = false, stencil = false;
    VkClearDepthStencilValue depth_clear{}, stencil_clear{};
    VkImageView depth_view = VK_NULL_HANDLE, stencil_view = VK_NULL_HANDLE;
    VkExtent2D area{};
    std::uint32_t layers = 0;
    VkPipeline bound = VK_NULL_HANDLE;
} seen;

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
    std::uint32_t, const VkImageMemoryBarrier*) {
    ++seen.barriers;
    if (seen.scope_open) ++seen.barriers_inside;
}

VKAPI_ATTR void VKAPI_CALL begin(VkCommandBuffer, const VkRenderingInfo* info) {
    ++seen.begins;
    seen.scope_open = true;
    seen.colors = info->colorAttachmentCount;
    for (std::uint32_t i = 0; i < seen.colors && i < 8; ++i) {
        seen.color_load[i] = info->pColorAttachments[i].loadOp;
        seen.color_store[i] = info->pColorAttachments[i].storeOp;
        seen.color_clear[i] = info->pColorAttachments[i].clearValue.color;
        seen.color_view[i] = info->pColorAttachments[i].imageView;
    }
    if (info->pDepthAttachment) {
        seen.depth = true;
        seen.depth_load = info->pDepthAttachment->loadOp;
        seen.depth_store = info->pDepthAttachment->storeOp;
        seen.depth_clear = info->pDepthAttachment->clearValue.depthStencil;
        seen.depth_view = info->pDepthAttachment->imageView;
    }
    if (info->pStencilAttachment) {
        seen.stencil = true;
        seen.stencil_load = info->pStencilAttachment->loadOp;
        seen.stencil_store = info->pStencilAttachment->storeOp;
        seen.stencil_clear = info->pStencilAttachment->clearValue.depthStencil;
        seen.stencil_view = info->pStencilAttachment->imageView;
    }
    seen.area = info->renderArea.extent;
    seen.layers = info->layerCount;
}
VKAPI_ATTR void VKAPI_CALL end(VkCommandBuffer) {
    ++seen.ends;
    seen.scope_open = false;
}

VKAPI_ATTR void VKAPI_CALL bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline pipeline) {
    seen.bound = pipeline;
}
VKAPI_ATTR void VKAPI_CALL push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t,
    std::uint32_t, const void*) {}
VKAPI_ATTR void VKAPI_CALL set_viewport(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) {}
VKAPI_ATTR void VKAPI_CALL set_scissor(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) {}
VKAPI_ATTR void VKAPI_CALL set_topology(VkCommandBuffer, VkPrimitiveTopology) {}
VKAPI_ATTR void VKAPI_CALL set_stencil_ref(VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL set_blend(VkCommandBuffer, const float*) {}
VKAPI_ATTR void VKAPI_CALL bind_vb(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*,
    const VkDeviceSize*, const VkDeviceSize*, const VkDeviceSize*) {}
VKAPI_ATTR void VKAPI_CALL bind_ib(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) {}
VKAPI_ATTR void VKAPI_CALL draw(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {
    ++seen.draws;
}
VKAPI_ATTR void VKAPI_CALL draw_indexed(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t,
    std::int32_t, std::uint32_t) {
    ++seen.draws;
}
VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {
    ++seen.dispatches;
    if (seen.scope_open) ++seen.dispatches_inside;
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

leonida::ResourceBinding binding(VkFormat format, VkImageAspectFlags aspects) {
    leonida::ResourceBinding b{};
    b.image = reinterpret_cast<VkImage>(std::uintptr_t{1});
    b.format = format;
    b.aspects = aspects;
    b.width = b.height = 8;
    b.depth = b.mip_levels = b.array_layers = b.samples = 1;
    b.color_attachment = aspects == VK_IMAGE_ASPECT_COLOR_BIT;
    return b;
}

const leonida::ResourceBinding color = binding(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
const leonida::ResourceBinding color2 = [] {
    leonida::ResourceBinding b = binding(VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    b.image = reinterpret_cast<VkImage>(std::uintptr_t{2});  // distinct handle: a distinct captured view
    return b;
}();
const leonida::ResourceBinding depth_stencil =
    binding(VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
const leonida::ResourceBinding depth_only = binding(VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
const leonida::ResourceBinding d24s8 =
    binding(VK_FORMAT_D24_UNORM_S8_UINT, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
// The graphics pipeline used by the draw-in-a-pass tests. Its colour formats match the captured RTVs.
leonida::GraphicsPipeline graphics;
leonida::GraphicsPipeline* current_pipeline = nullptr;

const leonida::ResourceBinding* resolve(leonida::ResourceId id) {
    if (id == 1) return &color;
    if (id == 2) return &depth_stencil;
    if (id == 3) return &depth_only;
    if (id == 4) return &color2;
    if (id == 5) return &d24s8;
    return nullptr;
}

// The pipeline variant a pass draw requests: its colour vector is the pipeline's VkRenderingCreateInfo count and
// formats (VUID 06179), and depth/stencil are the two plane formats directly.
std::vector<VkFormat> variant_colors;
VkFormat variant_depth = VK_FORMAT_UNDEFINED, variant_stencil = VK_FORMAT_UNDEFINED;
int variant_calls = 0;
void attach_variant(leonida::GraphicsPipeline& g, VkPipeline result) {
    g.variant_attachments = [result](const std::vector<VkFormat>& colors, VkFormat depth, VkFormat stencil) {
        variant_colors = colors;
        variant_depth = depth;
        variant_stencil = stencil;
        ++variant_calls;
        return result;
    };
}

leonida::ReplayFunctions functions() {
    leonida::ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.begin_rendering = begin;
    vk.end_rendering = end;
    vk.image_view = [](const auto& b, const auto&) {
        const bool second = b.image == reinterpret_cast<VkImage>(std::uintptr_t{2});
        return reinterpret_cast<VkImageView>(static_cast<std::uintptr_t>(second ? 8 : 7));
    };
    vk.bind_pipeline = bind_pipeline;
    vk.push_constants = push_constants;
    vk.set_viewport = set_viewport;
    vk.set_scissor = set_scissor;
    vk.set_primitive_topology = set_topology;
    vk.set_primitive_restart_enable = [](VkCommandBuffer, VkBool32) {};  // required by prepare_draw
    vk.set_stencil_reference = set_stencil_ref;
    vk.set_blend_constants = set_blend;
    vk.bind_vertex_buffers2 = bind_vb;
    vk.bind_index_buffer = bind_ib;
    vk.draw = draw;
    vk.draw_indexed = draw_indexed;
    vk.dispatch = dispatch;
    vk.buffer_at = [](std::uint64_t, std::uint64_t&, VkBuffer&, VkDeviceSize&) { return false; };
    vk.graphics_pipeline = [](std::uintptr_t) -> const leonida::GraphicsPipeline* { return current_pipeline; };
    return vk;
}

namespace B = leonida::d3d12_render_beginning;
namespace E = leonida::d3d12_render_ending;

leonida::BeginRenderPass pass(std::uint32_t begin = B::preserve, std::uint32_t finish = E::preserve) {
    leonida::BeginRenderPass p{};
    leonida::RenderPassColor c{};
    c.view = leonida::TargetView{1, 0, 0, 1};
    c.beginning.type = begin;
    // 0.2f/0.4f/0.6f are 51/102/153 over 255: exact on the 8-bit UNORM target, so the colour survives the D3D-style
    // quantisation of render_target_clear_color (#589) unchanged and the expectation below stays exact either way.
    c.beginning.color = {0.2f, 0.4f, 0.6f, 1.0f};
    c.ending.type = finish;
    p.colors.push_back(c);
    return p;
}

leonida::ReplayResult run(const leonida::CommandList& list, const leonida::ReplayFunctions& vk) {
    seen = {};
    return leonida::replay(list, VK_NULL_HANDLE, resolve, vk);
}

} // namespace

int main() {
    using leonida::ReplayResult;
    current_pipeline = &graphics;
    graphics.pipeline = reinterpret_cast<VkPipeline>(std::uintptr_t{9});
    graphics.layout = reinterpret_cast<VkPipelineLayout>(std::uintptr_t{11});
    graphics.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
    graphics.variant = [](std::uint32_t) { return reinterpret_cast<VkPipeline>(std::uintptr_t{12}); };
    attach_variant(graphics, reinterpret_cast<VkPipeline>(std::uintptr_t{12}));
    const auto vk = functions();

    // PRESERVE at both ends is a LOAD/STORE scope over the captured RTV.
    leonida::CommandList list;
    expect(list.begin_render_pass(pass()) && list.end_render_pass() && list.close(), "record a balanced pass");
    expect(run(list, vk) == ReplayResult::success, "replay a balanced pass");
    expect(seen.begins == 1 && seen.ends == 1, "one rendering scope");
    expect(seen.colors == 1 && seen.color_load[0] == VK_ATTACHMENT_LOAD_OP_LOAD &&
               seen.color_store[0] == VK_ATTACHMENT_STORE_OP_STORE,
           "PRESERVE maps to LOAD/STORE");
    expect(seen.area.width == 8 && seen.area.height == 8 && seen.layers == 1, "area and layers from the RTV");
    expect(seen.barriers > 0 && seen.barriers_inside == 0, "layout transitions precede BeginRendering");

    // A CLEAR beginning writes the captured colour through the load op, not with a separate clear command.
    leonida::CommandList clear;
    expect(clear.begin_render_pass(pass(B::clear, E::discard)) && clear.end_render_pass() && clear.close(),
           "record a CLEAR/DISCARD pass");
    expect(run(clear, vk) == ReplayResult::success, "replay a CLEAR/DISCARD pass");
    expect(seen.color_load[0] == VK_ATTACHMENT_LOAD_OP_CLEAR, "CLEAR maps to LOAD_OP_CLEAR");
    expect(seen.color_clear[0].float32[0] == 0.2f && seen.color_clear[0].float32[1] == 0.4f &&
               seen.color_clear[0].float32[2] == 0.6f && seen.color_clear[0].float32[3] == 1.0f,
           "the recorded clear colour is used");
    expect(seen.color_store[0] == VK_ATTACHMENT_STORE_OP_NONE, "DISCARD stores nothing");

    // A NO_ACCESS beginning skips the RTV regardless of its ending access. The CLEAR attachment becomes target 0.
    leonida::BeginRenderPass mixed = pass(B::clear, E::preserve);
    leonida::RenderPassColor passive{};
    passive.view = leonida::TargetView{999, 0, 0, 1};  // would be unresolved if it were touched
    passive.beginning.type = B::no_access;
    passive.ending.type = E::no_access;
    mixed.colors.insert(mixed.colors.begin(), passive);
    leonida::RenderPassColor no_load = passive;
    no_load.ending.type = E::preserve;
    mixed.colors.push_back(no_load);
    leonida::CommandList holed;
    expect(holed.begin_render_pass(mixed) && holed.end_render_pass() && holed.close(),
           "record a pass with passive NO_ACCESS slots");
    expect(run(holed, vk) == ReplayResult::success, "passive slots are not resolved");
    expect(seen.colors == 1 && seen.barriers == 1, "only the raster-bound RTV is attached and transitioned");
    expect(seen.color_view[0] == reinterpret_cast<VkImageView>(std::uintptr_t{7}) &&
               seen.color_load[0] == VK_ATTACHMENT_LOAD_OP_CLEAR && seen.color_store[0] == VK_ATTACHMENT_STORE_OP_STORE,
           "target 0 gets the CLEAR/PRESERVE view");

    // An interior NO_ACCESS RTV is skipped. The distinct views show that the raster RTVs keep descriptor order.
    leonida::BeginRenderPass spaced = pass(B::preserve, E::preserve);
    leonida::RenderPassColor gap{};
    gap.view = leonida::TargetView{999, 0, 0, 1};
    gap.beginning.type = B::no_access;
    gap.ending.type = E::no_access;
    spaced.colors.push_back(gap);
    leonida::RenderPassColor last{};
    last.view = leonida::TargetView{4, 0, 0, 1};
    last.beginning.type = B::discard;
    last.ending.type = E::no_access;
    spaced.colors.push_back(last);
    leonida::CommandList spaced_list;
    expect(spaced_list.begin_render_pass(spaced) && spaced_list.end_render_pass() && spaced_list.close(),
           "record a pass with an interior NO_ACCESS slot");
    expect(run(spaced_list, vk) == ReplayResult::success, "the interior passive slot is not resolved");
    expect(seen.colors == 2, "colorAttachmentCount is the number of raster-bound RTVs");
    expect(seen.color_view[0] == reinterpret_cast<VkImageView>(std::uintptr_t{7}), "slot 0 keeps its view");
    expect(seen.color_view[1] == reinterpret_cast<VkImageView>(std::uintptr_t{8}) &&
               seen.color_load[1] == VK_ATTACHMENT_LOAD_OP_DONT_CARE && seen.color_store[1] == VK_ATTACHMENT_STORE_OP_NONE,
           "target 1 gets the DISCARD/NO_ACCESS view from descriptor 2");

    // Both DSV planes active: one image view (VUID 06085), independent ops and clear values.
    leonida::BeginRenderPass ds = pass(B::clear, E::preserve);
    leonida::RenderPassDepthStencil d{};
    d.view = leonida::TargetView{2, 0, 0, 1};
    d.depth_beginning.type = B::clear;
    d.depth_beginning.depth = 0.375f;
    d.depth_beginning.stencil = 0xa5;
    d.stencil_beginning.type = B::clear;
    d.stencil_beginning.stencil = 0x5a;
    d.depth_ending.type = E::preserve;
    d.stencil_ending.type = E::preserve;
    ds.depth_stencil = d;
    leonida::CommandList depth;
    expect(depth.begin_render_pass(ds) && depth.end_render_pass() && depth.close(), "record a DSV pass");
    expect(run(depth, vk) == ReplayResult::success, "replay a DSV pass");
    expect(seen.depth && seen.stencil, "both active planes are attached");
    expect(seen.depth_view == seen.stencil_view, "one DSV view for both aspects (VUID 06085)");
    expect(seen.depth_load == VK_ATTACHMENT_LOAD_OP_CLEAR && seen.stencil_load == VK_ATTACHMENT_LOAD_OP_CLEAR,
           "both planes clear independently");
    expect(seen.depth_store == VK_ATTACHMENT_STORE_OP_STORE && seen.stencil_store == VK_ATTACHMENT_STORE_OP_STORE,
           "both PRESERVE endings store");
    expect(seen.depth_clear.depth == 0.375f && seen.stencil_clear.stencil == 0x5a,
           "depth and stencil clear values are independent");

    // A passive stencil plane is not attached; only the depth plane is. A DISCARD beginning is active (DONT_CARE),
    // never represented by a dropped NO_ACCESS attachment.
    leonida::BeginRenderPass ds_passive = ds;
    ds_passive.depth_stencil->stencil_beginning.type = B::no_access;
    ds_passive.depth_stencil->stencil_ending.type = E::discard;
    ds_passive.depth_stencil->depth_beginning.type = B::discard;
    leonida::CommandList one_plane;
    expect(one_plane.begin_render_pass(ds_passive) && one_plane.end_render_pass() && one_plane.close(),
           "record a depth-only-active DSV pass");
    expect(run(one_plane, vk) == ReplayResult::success, "replay a depth-only-active DSV pass");
    expect(seen.depth && !seen.stencil, "the passive stencil plane is not attached");
    expect(seen.depth_load == VK_ATTACHMENT_LOAD_OP_DONT_CARE, "DISCARD beginning maps to DONT_CARE");

    // A depth-only format never gets a stencil attachment even when the stencil accesses are active.
    leonida::BeginRenderPass ds_single = ds;
    ds_single.depth_stencil->view = leonida::TargetView{3, 0, 0, 1};
    leonida::CommandList single;
    expect(single.begin_render_pass(ds_single) && single.end_render_pass() && single.close(),
           "record a depth-only-format DSV pass");
    expect(run(single, vk) == ReplayResult::success, "replay a depth-only-format DSV pass");
    expect(seen.depth && !seen.stencil, "no stencil attachment without the stencil aspect");

    // A draw inside the pass uses the pass snapshot even when the OM render targets recorded with it are stale and
    // unresolvable; no second rendering scope and no barrier are emitted inside the Vulkan scope.
    leonida::Draw draw{};
    draw.pipeline = 5;
    auto stale_views = std::make_shared<leonida::DrawViews>();
    stale_views->render_targets[0] = leonida::TargetView{999, 0, 0, 1};
    stale_views->render_target_count = 1;
    draw.views = stale_views;
    draw.depth = leonida::TargetView{999, 0, 0, 1};
    draw.viewport = {0.0f, 0.0f, 8.0f, 8.0f, 0.0f, 1.0f};
    draw.scissor = {0, 0, 8, 8};
    draw.count = 3;
    draw.instances = 1;
    leonida::CommandList with_draw;
    expect(with_draw.begin_render_pass(pass(B::clear, E::preserve)) && with_draw.draw(draw) &&
               with_draw.end_render_pass() && with_draw.close(),
           "record a draw inside a pass with stale OM targets");
    expect(run(with_draw, vk) == ReplayResult::success, "the draw resolves from the pass snapshot");
    expect(seen.draws == 1 && seen.begins == 1 && seen.ends == 1, "one scope and one draw");
    expect(seen.bound == graphics.pipeline, "the PSO matches the captured pass formats");
    expect(seen.barriers_inside == 0 && seen.dispatches_inside == 0, "no barrier or dispatch inside the scope");

    // A one-RTV PSO draws to target 0 in [NO_ACCESS, CLEAR], using the base pipeline.
    mixed.colors.pop_back();
    leonida::CommandList leading_gap;
    expect(leading_gap.begin_render_pass(mixed) && leading_gap.draw(draw) && leading_gap.end_render_pass() &&
               leading_gap.close(),
           "record a one-RTV draw after a leading NO_ACCESS descriptor");
    variant_calls = 0;
    expect(run(leading_gap, vk) == ReplayResult::success && seen.colors == 1 &&
               seen.color_view[0] == reinterpret_cast<VkImageView>(std::uintptr_t{7}) &&
               seen.draws == 1 && seen.bound == graphics.pipeline && variant_calls == 0,
           "target 0 reaches the first raster-bound RTV with the one-RTV PSO");

    // PSO formats match the compacted slot order. A variant trims the extra PSO format to the scope's count.
    leonida::GraphicsPipeline three = graphics;
    three.color_formats = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    const VkPipeline three_variant = reinterpret_cast<VkPipeline>(std::uintptr_t{13});
    attach_variant(three, three_variant);
    leonida::CommandList spaced_draw;
    expect(spaced_draw.begin_render_pass(spaced) && spaced_draw.draw(draw) && spaced_draw.end_render_pass() &&
               spaced_draw.close(),
           "record a draw in a pass with an interior passive slot");
    current_pipeline = &three;
    variant_calls = 0;
    const ReplayResult spaced_result = run(spaced_draw, vk);
    current_pipeline = &graphics;
    expect(spaced_result == ReplayResult::success, "pass-draw format matching uses the raster slots");
    expect(variant_calls == 1 && variant_colors.size() == 2 && variant_colors[0] == VK_FORMAT_R8G8B8A8_UNORM &&
               variant_colors[1] == VK_FORMAT_R16G16B16A16_UNORM,
           "the variant formats match the two raster-bound views in order");
    expect(seen.draws == 1 && seen.bound == three_variant, "the draw binds the count-matched variant");

    // Retain the replay fallback for a PSO with fewer formats than raster-bound RTVs. Its variant takes the
    // extra attachment format from the pass to satisfy VUID 06179.
    leonida::GraphicsPipeline one_rtv = graphics;
    one_rtv.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
    const VkPipeline one_variant = reinterpret_cast<VkPipeline>(std::uintptr_t{14});
    attach_variant(one_rtv, one_variant);
    leonida::CommandList one_in_two;
    expect(one_in_two.begin_render_pass(spaced) && one_in_two.draw(draw) && one_in_two.end_render_pass() &&
               one_in_two.close(),
           "record a 1-RTV PSO draw with two raster-bound RTVs");
    current_pipeline = &one_rtv;
    variant_calls = 0;
    const ReplayResult one_result = run(one_in_two, vk);
    current_pipeline = &graphics;
    expect(one_result == ReplayResult::success, "the one-RTV fallback replays with two raster-bound RTVs");
    expect(variant_calls == 1 && variant_colors.size() == 2 && variant_colors[0] == VK_FORMAT_R8G8B8A8_UNORM &&
               variant_colors[1] == VK_FORMAT_R16G16B16A16_UNORM,
           "the pipeline declares two colour attachments, matching the pass count (VUID 06179)");
    expect(seen.bound == one_variant, "the draw binds the count-matched variant");

    // PSO 3 RTVs inside a 1-slot pass: the pipeline declares one colour attachment.
    leonida::GraphicsPipeline three_in_one = graphics;
    three_in_one.color_formats = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_UNORM};
    const VkPipeline truncated = reinterpret_cast<VkPipeline>(std::uintptr_t{15});
    attach_variant(three_in_one, truncated);
    leonida::CommandList three_list;
    expect(three_list.begin_render_pass(pass()) && three_list.draw(draw) && three_list.end_render_pass() &&
               three_list.close(),
           "record a 3-RTV PSO draw in a 1-slot pass");
    current_pipeline = &three_in_one;
    variant_calls = 0;
    variant_colors.clear();
    const ReplayResult three_result = run(three_list, vk);
    current_pipeline = &graphics;
    expect(three_result == ReplayResult::success, "a 3-RTV PSO in a 1-slot pass is replayed, not refused");
    expect(variant_calls == 1 && variant_colors.size() == 1 && variant_colors[0] == VK_FORMAT_R8G8B8A8_UNORM,
           "the pipeline declares 1 colour attachment, matching the pass count (VUID 06179)");
    expect(seen.bound == truncated, "the draw binds the truncated variant");

    // Capture the blend array produced by the helper used by vkCreateGraphicsPipelines. VUID 00605 compares
    // the whole state even for undefined-format slots, which VUID 06062 skips.
    VkPipelineColorBlendAttachmentState blended{};
    blended.blendEnable = VK_TRUE;
    blended.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blended.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blended.colorWriteMask = 0xf;
    const std::vector<VkPipelineColorBlendAttachmentState> base{blended};
    const auto supports_blend = [](VkFormat format) {
        return format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R16G16B16A16_UNORM;
    };
    const auto same_state = [](const auto& a, const auto& b) { return std::memcmp(&a, &b, sizeof(a)) == 0; };
    const auto padded = leonida::variant_blend_states(
        base, {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED, VK_FORMAT_R16G16B16A16_UNORM},
        {VK_FORMAT_R8G8B8A8_UNORM}, false, supports_blend);
    expect(padded && padded->size() == 3 && same_state((*padded)[0], blended) &&
               same_state((*padded)[1], blended) && same_state((*padded)[2], blended),
           "both undefined and blend-capable padding keep the common blend state");

    const std::vector<VkFormat> integer_rtv{VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT};
    const std::vector<VkFormat> declared{VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED};
    expect(!leonida::variant_blend_states({blended, blended}, integer_rtv, declared, false, supports_blend),
           "a newly used non-blendable format is refused without independentBlend");
    const auto reinterpreted =
        leonida::variant_blend_states({blended, blended}, integer_rtv, declared, true, supports_blend);
    auto disabled = blended;
    disabled.blendEnable = VK_FALSE;
    expect(reinterpreted && reinterpreted->size() == 2 && same_state((*reinterpreted)[0], blended) &&
               same_state((*reinterpreted)[1], disabled),
           "independentBlend permits disabling blending only for the non-blendable real format");

    // Outside a pass a variant unbinds a slot without changing the PSO's blend array.
    const auto unbound = leonida::variant_blend_states(
        {blended, blended}, declared, {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_UNORM}, false, supports_blend);
    expect(unbound && unbound->size() == 2 && same_state((*unbound)[0], blended) && same_state((*unbound)[1], blended),
           "unbinding an OM RTV preserves both original blend states");

    // A genuine format clash at an active slot (both real but different) is still refused before emit.
    leonida::GraphicsPipeline wrong = graphics;
    wrong.color_formats = {VK_FORMAT_R16G16B16A16_UNORM};  // pass slot 0 is RGBA8
    leonida::CommandList mismatch;
    expect(mismatch.begin_render_pass(pass()) && mismatch.draw(draw) && mismatch.end_render_pass() && mismatch.close(),
           "record a draw whose PSO format clashes with the pass");
    current_pipeline = &wrong;
    variant_calls = 0;
    const ReplayResult mismatch_result = run(mismatch, vk);
    current_pipeline = &graphics;
    expect(mismatch_result == ReplayResult::unsupported && seen.begins == 0 && variant_calls == 0,
           "a real active-slot format mismatch is refused before any command is emitted");

    // RISK-1: a D24S8 DSV with a NO_ACCESS stencil plane and a two-plane D24S8 PSO is the canonical depth-only pass.
    // The pass declares only DEPTH; the variant keeps D24S8 on depth and makes stencil UNDEFINED.
    leonida::GraphicsPipeline dsv = graphics;
    dsv.depth_format = VK_FORMAT_D24_UNORM_S8_UINT;
    dsv.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    const VkPipeline dsv_variant = reinterpret_cast<VkPipeline>(std::uintptr_t{16});
    attach_variant(dsv, dsv_variant);
    leonida::BeginRenderPass depth_only_pass = pass(B::clear, E::preserve);
    leonida::RenderPassDepthStencil only_depth{};
    only_depth.view = leonida::TargetView{5, 0, 0, 1};
    only_depth.depth_beginning.type = B::clear;
    only_depth.depth_beginning.depth = 0.5f;
    only_depth.depth_ending.type = E::preserve;
    only_depth.stencil_beginning.type = B::no_access;
    only_depth.stencil_ending.type = E::no_access;
    depth_only_pass.depth_stencil = only_depth;
    leonida::CommandList dsv_list;
    expect(dsv_list.begin_render_pass(depth_only_pass) && dsv_list.draw(draw) && dsv_list.end_render_pass() &&
               dsv_list.close(),
           "record a D24S8 depth-only pass with a two-plane PSO");
    current_pipeline = &dsv;
    variant_calls = 0;
    const ReplayResult dsv_result = run(dsv_list, vk);
    current_pipeline = &graphics;
    expect(dsv_result == ReplayResult::success, "the canonical D24S8 + NO_ACCESS stencil pattern is supported");
    expect(variant_calls == 1 && variant_depth == VK_FORMAT_D24_UNORM_S8_UINT &&
               variant_stencil == VK_FORMAT_UNDEFINED,
           "the variant keeps D24S8 depth and drops the inactive stencil plane");
    expect(seen.depth && !seen.stencil && seen.bound == dsv_variant,
           "only the depth plane is attached and the variant is bound");

    // Commands Vulkan forbids inside a rendering scope are refused in prepare, before any command buffer is touched.
    leonida::CommandList bad_dispatch;
    leonida::Dispatch dispatch{};
    dispatch.pipeline = 5;
    dispatch.x = dispatch.y = dispatch.z = 1;
    expect(bad_dispatch.begin_render_pass(pass()) && bad_dispatch.dispatch(dispatch) &&
               bad_dispatch.end_render_pass() && bad_dispatch.close(),
           "record a dispatch inside a pass");
    expect(run(bad_dispatch, vk) == ReplayResult::invalid_render_pass && seen.begins == 0,
           "a dispatch inside a pass is refused before emit");

    // A submitted list never resolves a resource it did not capture at record time.
    leonida::BeginRenderPass missing = pass();
    missing.colors[0].view.resource = 999;
    leonida::CommandList unresolved;
    expect(unresolved.begin_render_pass(missing) && unresolved.end_render_pass() && unresolved.close(),
           "record a pass with an unknown RTV");
    expect(run(unresolved, vk) == ReplayResult::unresolved_resource && seen.begins == 0,
           "an unresolved RTV fails before anything is written");

    // Nesting is refused while recording (command_list_test.cpp); prepare also rejects a pass left open at Close.
    leonida::CommandList unbalanced;
    expect(unbalanced.begin_render_pass(pass()) && unbalanced.close(), "record an unclosed pass");
    // Close marks the IR list closed (so Reset is legal and can report the same E_INVALIDARG) but leaves the pass
    // open; prepare/execute then reject the whole batch before any command buffer is touched.
    expect(unbalanced.closed() && unbalanced.render_pass_open(), "Close keeps the unbalanced pass visible");
    expect(run(unbalanced, vk) == ReplayResult::invalid_render_pass && seen.begins == 0,
           "an unclosed pass is refused before batch emit");

    // Without dynamic rendering the pass cannot be replayed at all.
    auto no_dynamic = functions();
    no_dynamic.begin_rendering = nullptr;
    expect(run(list, no_dynamic) == ReplayResult::unsupported && seen.begins == 0,
           "a pass needs vkCmdBeginRendering");

    if (failures) return 1;
    std::printf("replay_render_pass_test: raster attachment order, blend states, pass-snapshot draw, DSV aspects, "
                "scope gates PASS\n");
    return 0;
}
