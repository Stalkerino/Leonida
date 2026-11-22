// Synthetic BeginRenderPass/EndRenderPass replay check: no Vulkan loader, GPU, shaders or game data. The mock
// captures the VkRenderingInfo handed to vkCmdBeginRendering and checks load/store ops, clear values, original-slot
// (no compaction) colour attachments, aspect handling, the pass-snapshot draw path, scope gates and the
// unresolved-resource gate (docs/d3d12_vulkan.md section 4.3).
#include "../src/vulkan_replay.h"
#include <cstdint>
#include <cstdio>

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
// The graphics pipeline used by the draw-in-a-pass tests. Its colour formats match the captured RTVs.
leonida::GraphicsPipeline graphics;
leonida::GraphicsPipeline* current_pipeline = nullptr;

const leonida::ResourceBinding* resolve(leonida::ResourceId id) {
    if (id == 1) return &color;
    if (id == 2) return &depth_stencil;
    if (id == 3) return &depth_only;
    if (id == 4) return &color2;
    return nullptr;
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
    c.beginning.color = {0.25f, 0.5f, 0.75f, 1.0f};
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
    expect(seen.color_clear[0].float32[0] == 0.25f && seen.color_clear[0].float32[1] == 0.5f &&
               seen.color_clear[0].float32[2] == 0.75f && seen.color_clear[0].float32[3] == 1.0f,
           "the recorded clear colour is used");
    expect(seen.color_store[0] == VK_ATTACHMENT_STORE_OP_NONE, "DISCARD stores nothing");

    // A passive NO_ACCESS slot keeps its original index with a null image view; the active slots keep theirs. An
    // active NO_ACCESS beginning is DONT_CARE (never LOAD/CLEAR) while an ending of NO_ACCESS stores nothing.
    leonida::BeginRenderPass mixed = pass(B::clear, E::preserve);
    leonida::RenderPassColor passive{};
    passive.view = leonida::TargetView{999, 0, 0, 1};  // would be unresolved if it were touched
    passive.beginning.type = B::no_access;
    passive.ending.type = E::no_access;
    mixed.colors.insert(mixed.colors.begin(), passive);
    leonida::RenderPassColor no_load = passive;
    no_load.view = leonida::TargetView{1, 0, 0, 1};
    no_load.ending.type = E::preserve;
    mixed.colors.push_back(no_load);
    leonida::CommandList holed;
    expect(holed.begin_render_pass(mixed) && holed.end_render_pass() && holed.close(),
           "record a pass with passive NO_ACCESS slots");
    expect(run(holed, vk) == ReplayResult::success, "passive slots are not resolved");
    expect(seen.colors == 3, "the scope covers through the highest active slot");
    expect(seen.color_view[0] == VK_NULL_HANDLE, "the passive slot keeps its index with a null view");
    expect(seen.color_view[1] != VK_NULL_HANDLE && seen.color_load[1] == VK_ATTACHMENT_LOAD_OP_CLEAR &&
               seen.color_store[1] == VK_ATTACHMENT_STORE_OP_STORE,
           "slot 1 keeps CLEAR/PRESERVE and its view");
    expect(seen.color_view[2] != VK_NULL_HANDLE && seen.color_load[2] == VK_ATTACHMENT_LOAD_OP_DONT_CARE &&
               seen.color_store[2] == VK_ATTACHMENT_STORE_OP_STORE,
           "slot 2 keeps DONT_CARE/PRESERVE and its view");

    // Interior passive slot with a distinct RTV at slot 2: no compaction, so shader output location 2 still lands
    // on that RTV. The view handle differs from slot 0's, so a shifted mapping would be visible here.
    leonida::BeginRenderPass spaced = pass(B::preserve, E::preserve);
    leonida::RenderPassColor gap{};
    gap.view = leonida::TargetView{999, 0, 0, 1};
    gap.beginning.type = B::no_access;
    gap.ending.type = E::no_access;
    spaced.colors.push_back(gap);
    leonida::RenderPassColor last{};
    last.view = leonida::TargetView{4, 0, 0, 1};
    last.beginning.type = B::preserve;
    last.ending.type = E::preserve;
    spaced.colors.push_back(last);
    leonida::CommandList spaced_list;
    expect(spaced_list.begin_render_pass(spaced) && spaced_list.end_render_pass() && spaced_list.close(),
           "record a pass with an interior NO_ACCESS slot");
    expect(run(spaced_list, vk) == ReplayResult::success, "the interior passive slot is not resolved");
    expect(seen.colors == 3, "colorAttachmentCount is the highest active slot plus one");
    expect(seen.color_view[0] == reinterpret_cast<VkImageView>(std::uintptr_t{7}), "slot 0 keeps its view");
    expect(seen.color_view[1] == VK_NULL_HANDLE, "slot 1 (NO_ACCESS) has a null image view");
    expect(seen.color_view[2] == reinterpret_cast<VkImageView>(std::uintptr_t{8}),
           "slot 2 keeps its own view at index 2");

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

    // A draw in a pass with an interior passive slot matches PSO formats by ORIGINAL slot index. The passive slot's
    // pipeline format is turned UNDEFINED through `variant`; the two active slots keep theirs. Compaction would
    // report an unsupported format mismatch here (slot 1 would compare RGBA16F against RGBA8).
    leonida::GraphicsPipeline three = graphics;
    three.color_formats = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_UNORM};
    std::uint32_t variant_mask = 0;
    three.variant = [&variant_mask](std::uint32_t m) {
        variant_mask = m;
        return reinterpret_cast<VkPipeline>(std::uintptr_t{13});
    };
    leonida::CommandList spaced_draw;
    expect(spaced_draw.begin_render_pass(spaced) && spaced_draw.draw(draw) && spaced_draw.end_render_pass() &&
               spaced_draw.close(),
           "record a draw in a pass with an interior passive slot");
    current_pipeline = &three;
    const ReplayResult spaced_result = run(spaced_draw, vk);
    current_pipeline = &graphics;
    expect(spaced_result == ReplayResult::success, "pass-draw format matching keeps original slot indices");
    expect(variant_mask == (1u << 1), "only the passive slot gets an UNDEFINED pipeline format");
    expect(seen.draws == 1 && seen.bound == reinterpret_cast<VkPipeline>(std::uintptr_t{13}),
           "the draw binds the slot-preserving variant");

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
    std::printf("replay_render_pass_test: slot-preserving attachments, pass-snapshot draw, DSV aspects, scope gates PASS\n");
    return 0;
}
