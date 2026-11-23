// Xbox physical pages (src/xbox_physical.h): CPU halves on a pagefile section mapped at title VAs over placeholders,
// write-fault dirty tracking, and coherence with the GPU halves (xbox_paging.h) through the sparse arena.
//
// Checks (FAIL = exit 1): one physical page mapped at two VAs shows the same bytes; a write marks exactly that page
// dirty, and taking the dirty list write-protects it again; CPU writes reach the GPU after an upload, read through
// the arena; a GPU write reaches the CPU half after a readback; on a page both sides write, each side's words stand
// (the CPU's where both wrote one); a failed upload stays dirty; a new GPU half starts as its CPU half; ReadFile into
// a clean page works after before_kernel_write; freed pages come back zeroed; no validation errors
// (LEONIDA_VK_VALIDATION=1).
// Measurements (INFO): cost of a first write to a clean page (the fault); upload cost, whole and changed words.
// No device, no sparse residency for buffers, or less than 64 GiB of sparse address space: SKIP, exit 77. Windows
// only.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"
#include "../src/xbox_paging.h"
#include "../src/xbox_physical.h"
#ifdef LEONIDA_XMEM_PAGE_MARKERS
#include "shims.h"
#include "xmem_page_markers.h"
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <vector>

using namespace leonida;
using namespace leonida::xbox;

namespace {

int g_failures = 0;
#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

// GPU work outside VulkanQueue::execute(), ordered on the queue's timeline (see vk_xbox_paging_test.cpp).
struct Gpu {
    vktest::Device& d;
    VulkanContext& context;
    VulkanQueue& queue;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
    std::uint64_t value = 0;
    std::vector<std::pair<VkCommandBuffer, std::uint64_t>> retired;

    Gpu(vktest::Device& device, VulkanContext& ctx, VulkanQueue& q) : d(device), context(ctx), queue(q) {
        VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex = d.family;
        d.fn<PFN_vkCreateCommandPool>("vkCreateCommandPool")(d.device, &info, nullptr, &pool);
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        d.fn<PFN_vkCreateSemaphore>("vkCreateSemaphore")(d.device, &semaphore, nullptr, &timeline);
    }
    ~Gpu() {
        d.fn<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(d.device);
        if (pool) d.fn<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(d.device, pool, nullptr);
        if (timeline) d.fn<PFN_vkDestroySemaphore>("vkDestroySemaphore")(d.device, timeline, nullptr);
    }

    template<typename Body>
    std::uint64_t submit(Body&& body) {
        VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc.commandPool = pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        d.fn<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(d.device, &alloc, &cb);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        d.fn<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(cb, &begin);
        body(cb);
        d.fn<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(cb);
        const std::uint64_t after = queue.submitted_ticket();
        const VkSemaphore wait_semaphore = queue.timeline();
        const std::uint64_t signal = ++value;
        VkTimelineSemaphoreSubmitInfo values{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        values.waitSemaphoreValueCount = after ? 1 : 0;
        values.pWaitSemaphoreValues = &after;
        values.signalSemaphoreValueCount = 1;
        values.pSignalSemaphoreValues = &signal;
        const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO, &values};
        info.waitSemaphoreCount = after ? 1 : 0;
        info.pWaitSemaphores = &wait_semaphore;
        info.pWaitDstStageMask = &stage;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &cb;
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &timeline;
        {
            std::lock_guard lock(context.submit_mutex());
            EXPECT(d.fn<PFN_vkQueueSubmit>("vkQueueSubmit")(d.queue, 1, &info, VK_NULL_HANDLE) == VK_SUCCESS);
        }
        queue.wait(timeline, signal);
        retired.push_back({cb, signal});
        return signal;
    }

    void wait(std::uint64_t v) {
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &timeline;
        info.pValues = &v;
        EXPECT(d.fn<PFN_vkWaitSemaphores>("vkWaitSemaphores")(d.device, &info, 10'000'000'000ull) == VK_SUCCESS);
        for (auto it = retired.begin(); it != retired.end();) {
            if (it->second > v) { ++it; continue; }
            d.fn<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(d.device, pool, 1, &it->first);
            it = retired.erase(it);
        }
    }
};

bool all_bytes(const void* p, std::uint8_t value, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i)
        if (b[i] != value) return false;
    return true;
}

}  // namespace

#ifdef LEONIDA_XMEM_PAGE_MARKERS
// Matches the measured 174 MiB, 2,784 distinct frames and three contiguous guest sub-windows. Window lengths
// and physical ordering are synthetic. Reverse/interleave the three backing sections to catch an accidental
// assumption that adjacent guest pages share an allocation or increasing file offsets.
int check_retail_backing() {
    constexpr std::uint32_t pages_per_window = 928, total = 3 * pages_per_window;
    constexpr VkDeviceSize bytes = total * kPageSize;
    constexpr std::uint32_t chunk = 32;
    static_assert(total % chunk == 0 && bytes == 182452224);
    using Allocate = BOOL(WINAPI*)(UINT64, UINT64*, UINT64*);
    using Map = void*(WINAPI*)(void*, UINT64, const UINT64*);
    using Reserve = void*(WINAPI*)(void*, SIZE_T, DWORD, UINT64, DWORD);
    using Free = BOOL(WINAPI*)(UINT64, const UINT64*);
    auto allocate = reinterpret_cast<Allocate>(rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages"));
    auto map = reinterpret_cast<Map>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages"));
    auto reserve = reinterpret_cast<Reserve>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    auto release = reinterpret_cast<Free>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages"));
    if (!allocate || !map || !reserve || !release) return 1;
    vktest::Device d;
    d.request_sparse = true;
    d.request_buffer_device_address = true;
    if (!d.create() || !d.sparse || !d.buffer_device_address ||
        (d.families[d.family].queueFlags & (VK_QUEUE_SPARSE_BINDING_BIT | VK_QUEUE_COMPUTE_BIT)) !=
        (VK_QUEUE_SPARSE_BINDING_BIT | VK_QUEUE_COMPUTE_BIT)) return 77;
    std::vector<UINT64> frames(total), order(total);
    std::uintptr_t base = 0x50000000;
    for (unsigned w = 0; w < 3; ++w) {
        UINT64 count = pages_per_window;
        if (!allocate(0x280, &count, frames.data() + w * pages_per_window) || count != pages_per_window) return 1;
    }
    for (unsigned i = 0; i < total; ++i)
        order[i] = frames[(i % 3) * pages_per_window + (pages_per_window - 1 - i / 3)];
    for (unsigned w = 0; w < 3; ++w) {
        auto* target = reinterpret_cast<void*>(base + w * pages_per_window * kPageSize);
        if (reserve(target, pages_per_window * kPageSize, 0x20402000, 0x23, 0x2004) != target ||
            map(target, pages_per_window, order.data() + w * pages_per_window) != target) return 1;
    }
    VkPhysicalDeviceMemoryProperties memory{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &memory);
    VulkanContext context(d.device_proc, d.device, true);
    VulkanQueue queue(context, d.queue, d.family);
    if (!context.loaded() || !queue.ok()) return 1;
    std::optional<PagePool> pool_storage;
    PagedRegion region;
    if (!region.create(d.device, d.device_proc, bytes)) return 1;
    auto& pool = pool_storage.emplace(d.device, d.device_proc, pick_page_memory_type(memory, region.memory_type_bits()));
    // All CPU writers are on this thread. There are no remaps until the last GPU fence/readback.
    struct GuestPages {
        std::map<PhysicalPage, std::uint8_t*> mapped;
        bool allocated(PhysicalPage id) const { return mapped.contains(id); }
        std::uint8_t* data(PhysicalPage id) const { return mapped.at(id); }
        void mark_dirty(PhysicalPage) {}  // The fixture explicitly uploads every page in the working set.
        bool exclusive(std::span<const PhysicalPage> ids, const std::function<void(std::size_t, std::uint8_t*)>& merge) {
            for (auto id : ids) if (!allocated(id)) return false;
            for (std::size_t i = 0; i < ids.size(); ++i) merge(i, data(ids[i]));
            return true;
        }
    } store;
    PhysicalGpu physical(d.device, d.device_proc, pool, store, memory);
    if (!pool.ok() || !physical.ok()) return 1;
    PageMapper mapper(region, pool);
    Gpu gpu(d, context, queue);
    std::vector<PhysicalPage> pinned(total);
    std::vector<PageId> backing(total);
    auto cpu = [=](unsigned index) { return reinterpret_cast<std::uint32_t*>(base + index * kPageSize); };
    auto marker = [](unsigned page) { return 0xbec10409u + page * 2654435761u; };
    for (unsigned i = 0; i < total; ++i) {
        auto* words = cpu(i);
        words[0] = marker(i);
        words[1] = 0;
        words[kPageSize / 4 - 1] = ~marker(i);
        if (order[i] > UINT32_MAX) return 1;
        pinned[i] = static_cast<PhysicalPage>(order[i]);
        if (!pinned[i] || !store.mapped.emplace(pinned[i], reinterpret_cast<std::uint8_t*>(words)).second) return 1;
        if (!(backing[i] = physical.gpu_page(pinned[i]))) return 1;
    }
    if (std::set<PhysicalPage>(pinned.begin(), pinned.end()).size() != total) return 1;
    const auto handle = mapper.register_page_pool(std::move(backing));
    const PageMappingRange all{PageRangeType::incrementing, total, 0};
    const PageMappingBatch bind{0, &all, 1};
    const BindQueue submit{queue.timeline(), [&](VkBuffer buffer, std::span<const VkSparseMemoryBind> binds,
                                                 BindQueue::Submitted submitted) {
        return queue.bind_sparse_ordered(buffer, binds, std::move(submitted));
    }};
    if (!mapper.copy_page_mappings_batch(handle, {&bind, 1}, submit)) return 1;
    for (unsigned i = 0; i < total; i += chunk) {
        const auto part = std::span<const PhysicalPage>(pinned).subspan(i, chunk);
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(physical.record_upload(cb, part)); }));
        if (g_failures) return 1;
    }
    VkShaderModule module{};
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader.pCode = xmem_page_markers;
    shader.codeSize = sizeof(xmem_page_markers);
    if (d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &shader, nullptr, &module) != VK_SUCCESS) return 1;
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    VkPipelineLayout layout{};
    if (d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(d.device, &layout_info, nullptr, &layout) != VK_SUCCESS) return 1;
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.layout = layout;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    VkPipeline pipeline{};
    if (d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(d.device, {}, 1, &info, nullptr, &pipeline) != VK_SUCCESS) return 1;
    gpu.wait(gpu.submit([&](VkCommandBuffer cb) {
        const auto address = region.address();
        d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        d.fn<PFN_vkCmdPushConstants>("vkCmdPushConstants")(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(address), &address);
        d.fn<PFN_vkCmdDispatch>("vkCmdDispatch")(cb, total / 32, 1, 1);
    }));
    d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(d.device, pipeline, nullptr);
    d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(d.device, layout, nullptr);
    d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(d.device, module, nullptr);
    for (unsigned i = 0; i < total; i += chunk) {
        const auto part = std::span<const PhysicalPage>(pinned).subspan(i, chunk);
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(physical.record_readback(cb, part)); }));
        if (g_failures || !physical.finish_readback(part)) return 1;
    }
    unsigned mismatches = 0;
    for (unsigned i = 0; i < total; ++i) {
        const auto* words = cpu(i);
        if (words[0] != marker(i) || words[1] != (marker(i) ^ 0x5fa9142bu) ||
            words[kPageSize / 4 - 1] != ~marker(i)) ++mismatches;
    }
    std::printf("span=%llu windows=3 mapped=2784 unique=2784 physical_order=interleaved-reversed "
                "stage_batch=2097152 compute_dispatches=1 invocations=2784 shader_outputs_checked=2784 "
                "input_and_neighbor_guards=5568 mismatches=%u\n", static_cast<unsigned long long>(bytes), mismatches);
    for (unsigned w = 0; w < 3; ++w) {
        auto* target = reinterpret_cast<void*>(base + w * pages_per_window * kPageSize);
        if (map(target, pages_per_window, nullptr) != target ||
            !release(pages_per_window, frames.data() + w * pages_per_window)) return 1;
        // Per-page unmap preserves each placeholder. Rejoin the window before releasing the reservation.
        if (!VirtualFree(target, pages_per_window * kPageSize, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) ||
            !VirtualFree(target, 0, MEM_RELEASE)) return 1;
    }
    return mismatches || d.validation_errors() ? 1 : 0;
}
#endif

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--xmem-retail-backing") == 0) {
#ifdef LEONIDA_XMEM_PAGE_MARKERS
        return check_retail_backing();
#else
        return 77;
#endif
    }
    vktest::Device d;
    d.request_sparse = true;
    d.request_buffer_device_address = true;
    if (!d.create()) {
        std::printf("SKIP: no usable Vulkan device\n");
        return 77;
    }
    if (!d.sparse || !(d.families[d.family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)) {
        std::printf("SKIP: no sparse residency for buffers on the graphics queue\n");
        return 77;
    }
    VkPhysicalDeviceProperties dev_props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &dev_props);
    constexpr VkDeviceSize kArena = 64ull << 30;
    if (dev_props.limits.sparseAddressSpaceSize < kArena) {
        std::printf("SKIP: %llu bytes of sparse address space, the test needs 64 GiB\n",
                    static_cast<unsigned long long>(dev_props.limits.sparseAddressSpaceSize));
        return 77;
    }
    VkPhysicalDeviceMemoryProperties mem_props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical,
                                                                                                  &mem_props);
    VulkanContext context(d.device_proc, d.device, true);
    VulkanQueue queue(context, d.queue, d.family);
    EXPECT(context.loaded() && queue.ok());

    // The pool is made after the arena (it needs the arena's memory types) but must outlive it.
    std::optional<PagePool> pool_storage;
    PagedRegion region;
    EXPECT(region.create(d.device, d.device_proc, kArena));
    PagePool& pool =
        pool_storage.emplace(d.device, d.device_proc, pick_page_memory_type(mem_props, region.memory_type_bits()));
    PageMapper mapper(region, pool);
    const BindQueue submit{queue.timeline(), [&](VkBuffer buffer, std::span<const VkSparseMemoryBind> binds,
                                                 BindQueue::Submitted submitted) {
        return queue.bind_sparse_ordered(buffer, binds, std::move(submitted));
    }};

    PhysicalPageStore store(1024);
    if (!store.ok()) {
        std::printf("FAIL the physical page store (section + placeholder APIs) could not be created\n");
        return 1;
    }
    PhysicalGpu gpu_half(d.device, d.device_proc, pool, store, mem_props);
    EXPECT(gpu_half.ok());
    Gpu gpu(d, context, queue);

    // Pages the store does not have are refused, and the GPU half allocates nothing for them.
    {
        EXPECT(store.data(1024) == nullptr && !store.allocated(1024) && !store.allocated(1000));
        store.mark_dirty(1024);
        const auto in_use = pool.pages_in_use();
        const PhysicalPage absent[]{1000, 5000};
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) {
            EXPECT(!gpu_half.record_upload(cb, absent));
            EXPECT(!gpu_half.record_readback(cb, absent));
        }));
        EXPECT(gpu_half.gpu_page(1000) == kNoPage);
        EXPECT(pool.pages_in_use() == in_use && gpu_half.pages_uploaded() == 0);
        EXPECT(!gpu_half.finish_readback(absent));
    }

    // --- Double mapping: one physical page at two VAs ----------------------------------------------------------
    std::vector<PhysicalPage> phys;
    EXPECT(store.allocate(4, phys));
    auto* r1 = static_cast<std::uint8_t*>(store.reserve(8));
    auto* r2 = static_cast<std::uint8_t*>(store.reserve(8));
    EXPECT(r1 && r2);
    EXPECT(store.map(r1, phys));                               // r1 pages 0..3 = P0..P3
    const PhysicalPage one[1]{phys[1]};
    EXPECT(store.map(r2 + 2 * kPageSize, one));                // r2 page 2 = P1 as well
    // A VA inside a page is refused, as MapViewOfFile3 refuses it, and r1 page 0 still shows P0.
    const PhysicalPage other[1]{phys[3]};
    store.data(phys[0])[0] = 0xA5;
    for (std::size_t offset : {std::size_t(1), std::size_t(4096), std::size_t(kPageSize - 1)})
        EXPECT(!store.map(r1 + offset, other));
    EXPECT(r1[0] == 0xA5);
    store.data(phys[0])[0] = 0;
    EXPECT(store.take_dirty().empty());                        // fresh pages start clean
    const auto faults0 = store.write_faults();
    std::memset(r1 + 1 * kPageSize, 0x11, kPageSize);          // write P1 through r1
    EXPECT(all_bytes(r2 + 2 * kPageSize, 0x11, kPageSize));    // ... and read it through r2
    EXPECT(store.write_faults() - faults0 == 1);               // one fault for the page, then plain writes
    {
        const auto dirty = store.take_dirty();
        EXPECT(dirty.size() == 1 && dirty[0] == phys[1]);
        std::printf("INFO double mapping: P%u written via VA %p, read via VA %p; dirty list {P%u}\n", phys[1],
                    static_cast<void*>(r1 + kPageSize), static_cast<void*>(r2 + 2 * kPageSize),
                    dirty.empty() ? 0u : dirty[0]);
    }
    r2[2 * kPageSize + 5] = 0x22;                              // a write through the other view is tracked too
    EXPECT(r1[kPageSize + 5] == 0x22);
    {
        const auto dirty = store.take_dirty();
        EXPECT(dirty.size() == 1 && dirty[0] == phys[1]);
    }

    // --- CPU write -> GPU read through the arena -----------------------------------------------------------------
    for (std::uint32_t i = 0; i < 4; ++i) std::memset(r1 + i * kPageSize, int(0x40 + i), kPageSize);
    const auto dirty = store.take_dirty();
    EXPECT(dirty.size() == 4);
    gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(gpu_half.record_upload(cb, dirty)); }));
    std::vector<PageId> gpu_pages;
    for (PhysicalPage p : phys) gpu_pages.push_back(gpu_half.gpu_page(p));
    const std::uint32_t pool_handle = mapper.register_page_pool(gpu_pages);
    constexpr std::uint64_t kVa = 12ull << 30;
    const PageMappingRange all{PageRangeType::incrementing, 4, 0};
    const PageMappingBatch batch{kVa, &all, 1};
    EXPECT(mapper.copy_page_mappings_batch(pool_handle, {&batch, 1}, submit));
    void* readback_map = nullptr;
    VkBuffer readback = d.buffer(4 * kPageSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback_map);
    gpu.wait(gpu.submit([&](VkCommandBuffer cb) {
        const VkBufferCopy copy{kVa, 0, 4 * kPageSize};
        d.fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(cb, region.buffer(), readback, 1, &copy);
    }));
    bool cpu_to_gpu = true;
    for (std::uint32_t i = 0; i < 4; ++i)
        cpu_to_gpu = cpu_to_gpu && all_bytes(static_cast<std::uint8_t*>(readback_map) + i * kPageSize,
                                             std::uint8_t(0x40 + i), kPageSize);
    EXPECT(cpu_to_gpu);
    std::printf("INFO CPU writes to P%u..P%u read back through arena VA 0x%llx after one upload: %s\n", phys[0],
                phys[3], (unsigned long long)kVa, cpu_to_gpu ? "match" : "MISMATCH");

    // --- GPU write -> CPU read -----------------------------------------------------------------------------------
    const PhysicalPage p2[1]{phys[2]};
    gpu.wait(gpu.submit([&](VkCommandBuffer cb) {
        d.fn<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(cb, region.buffer(), kVa + 2 * kPageSize, kPageSize, 0x5A5A5A5Au);
        EXPECT(gpu_half.record_readback(cb, p2));
    }));
    const PhysicalPage p3[1]{phys[3]};
    EXPECT(!gpu_half.finish_readback(p3));  // not the pages that were read back
    EXPECT(gpu_half.finish_readback(p2));
    EXPECT(!gpu_half.finish_readback(p2));  // finished once
    EXPECT(all_bytes(r1 + 2 * kPageSize, 0x5A, kPageSize));
    EXPECT(store.take_dirty().empty());  // a readback updates the CPU half without making it dirty
    std::printf("INFO GPU fill of arena page -> readback -> CPU view of P%u reads 0x%02x\n", phys[2],
                r1[2 * kPageSize]);

    // --- Pages both sides write ----------------------------------------------------------------------------------
    // P2 is at r1 + 2 pages on the CPU and at kVa + 2 pages in the arena; P3 one page on.
    constexpr VkDeviceSize kHalf = kPageSize / 2;
    std::uint8_t* const p2_cpu = r1 + 2 * kPageSize;
    const auto gpu_fill = [&](VkDeviceSize va, VkDeviceSize size, std::uint32_t word) {
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) {
            d.fn<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(cb, region.buffer(), va, size, word);
        }));
    };
    // Whether the arena page at `va` holds `value` in [offset, offset + size).
    const auto gpu_holds = [&](VkDeviceSize va, VkDeviceSize offset, VkDeviceSize size, std::uint8_t value) {
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) {
            const VkBufferCopy copy{va, 0, kPageSize};
            d.fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(cb, region.buffer(), readback, 1, &copy);
        }));
        return all_bytes(static_cast<std::uint8_t*>(readback_map) + offset, value, std::size_t(size));
    };
    const auto upload_dirty = [&] {
        const auto pages = store.take_dirty();
        bool uploaded = false;
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { uploaded = gpu_half.record_upload(cb, pages); }));
        return uploaded;
    };
    const auto read_back = [&](std::span<const PhysicalPage> pages) {
        bool recorded = false;
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { recorded = gpu_half.record_readback(cb, pages); }));
        return recorded && gpu_half.finish_readback(pages);
    };
    {
        const VkDeviceSize p2_gpu = kVa + 2 * kPageSize;
        // The GPU writes the first half of P2 and the CPU the second: the upload keeps the GPU's half...
        gpu_fill(p2_gpu, kHalf, 0x77777777u);
        std::memset(p2_cpu + kHalf, 0x66, kHalf);
        EXPECT(upload_dirty());
        EXPECT(gpu_holds(p2_gpu, 0, kHalf, 0x77) && gpu_holds(p2_gpu, kHalf, kHalf, 0x66));
        // ... and a readback gives the CPU the GPU's half.
        EXPECT(read_back(p2));
        EXPECT(all_bytes(p2_cpu, 0x77, kHalf) && all_bytes(p2_cpu + kHalf, 0x66, kHalf));
        // The same with the CPU's write not uploaded yet: the readback keeps it, and the next upload sends it.
        gpu_fill(p2_gpu, kHalf, 0x55555555u);
        std::memset(p2_cpu + kHalf, 0x44, kHalf);
        EXPECT(read_back(p2));
        EXPECT(all_bytes(p2_cpu, 0x55, kHalf) && all_bytes(p2_cpu + kHalf, 0x44, kHalf));
        EXPECT(upload_dirty());
        EXPECT(gpu_holds(p2_gpu, 0, kHalf, 0x55) && gpu_holds(p2_gpu, kHalf, kHalf, 0x44));
        // Both write the same word: the CPU's write stands, on both sides.
        p2_cpu[0] = 0x12;
        gpu_fill(p2_gpu, kPageSize, 0x34343434u);
        EXPECT(read_back(p2));
        EXPECT(p2_cpu[0] == 0x12 && all_bytes(p2_cpu + 4, 0x34, kPageSize - 4));
        EXPECT(upload_dirty());
        EXPECT(gpu_holds(p2_gpu, 0, 1, 0x12) && gpu_holds(p2_gpu, 4, kPageSize - 4, 0x34));
        // A CPU write between recording a readback and finishing it stands too.
        gpu_fill(p2_gpu, kPageSize, 0x21212121u);
        bool recorded = false;
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { recorded = gpu_half.record_readback(cb, p2); }));
        p2_cpu[8] = 0x99;
        EXPECT(recorded && gpu_half.finish_readback(p2));
        EXPECT(p2_cpu[8] == 0x99 && p2_cpu[12] == 0x21);
        EXPECT(upload_dirty());
        EXPECT(gpu_holds(p2_gpu, 8, 1, 0x99) && gpu_holds(p2_gpu, 12, 4, 0x21));
    }

    // --- Failed uploads, and pages the GPU half has nothing of yet ---------------------------------------------------
    {
        // A failed upload leaves its pages dirty: P2 with a page the store does not have fails as a whole.
        p2_cpu[16] = 0x5C;
        auto taken = store.take_dirty();
        EXPECT(taken.size() == 1 && taken[0] == phys[2]);
        taken.push_back(5000);
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(!gpu_half.record_upload(cb, taken)); }));
        const auto again = store.take_dirty();
        EXPECT(again.size() == 1 && !again.empty() && again[0] == phys[2]);
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(gpu_half.record_upload(cb, again)); }));
        // A page with no GPU half, or one never uploaded, is not read back. A new GPU half is marked dirty, and its
        // first upload copies all of the CPU half over whatever its pool page held.
        std::vector<PhysicalPage> fresh;
        EXPECT(store.allocate(1, fresh));
        EXPECT(!read_back(fresh));
        const PageId fresh_gpu = gpu_half.gpu_page(fresh[0]);
        EXPECT(fresh_gpu != kNoPage);
        EXPECT(!read_back(fresh));
        const std::uint32_t fresh_pool = mapper.register_page_pool({fresh_gpu});
        const PageMappingRange single{PageRangeType::incrementing, 1, 0};
        const VkDeviceSize fresh_va = kVa + 4 * kPageSize;
        const PageMappingBatch at{fresh_va, &single, 1};
        EXPECT(mapper.copy_page_mappings_batch(fresh_pool, {&at, 1}, submit));
        gpu_fill(fresh_va, kPageSize, 0x99999999u);  // stands for what the pool page held
        const auto first = store.take_dirty();
        EXPECT(first.size() == 1 && !first.empty() && first[0] == fresh[0]);
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(gpu_half.record_upload(cb, first)); }));
        EXPECT(gpu_holds(fresh_va, 0, kPageSize, 0x00));
        EXPECT(mapper.clear_page_mappings(fresh_va, 1, submit));
        EXPECT(mapper.unregister_page_pool(fresh_pool));
    }

    // --- Kernel writes into a clean page ---------------------------------------------------------------------------
    {
        wchar_t dir[MAX_PATH]{}, path[MAX_PATH]{};
        EXPECT(GetTempPathW(MAX_PATH, dir) && GetTempFileNameW(dir, L"lxp", 0, path));
        const HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        EXPECT(file != INVALID_HANDLE_VALUE);
        const std::vector<std::uint8_t> pattern(kPageSize, 0xC3);
        DWORD wrote = 0;
        EXPECT(WriteFile(file, pattern.data(), DWORD(kPageSize), &wrote, nullptr) && wrote == kPageSize);
        DWORD error = 0;
        const auto read_into = [&](void* into) {
            SetFilePointer(file, 0, nullptr, FILE_BEGIN);
            DWORD got = 0;
            const bool read = ReadFile(file, into, DWORD(kPageSize), &got, nullptr) && got == kPageSize;
            error = read ? 0 : GetLastError();
            return read;
        };
        std::uint8_t* const p3_cpu = r1 + 3 * kPageSize;
        EXPECT(store.take_dirty().empty());
        EXPECT(!read_into(p3_cpu) && error == ERROR_NOACCESS);  // the fault handler never sees a kernel write
        EXPECT(PhysicalPageStore::before_kernel_write(p3_cpu, kPageSize));
        const auto during = store.take_dirty();  // taken while the write is in flight: the page stays open
        EXPECT(during.size() == 1 && !during.empty() && during[0] == phys[3]);
        EXPECT(read_into(p3_cpu));
        PhysicalPageStore::after_kernel_write(p3_cpu, kPageSize);
        EXPECT(all_bytes(p3_cpu, 0xC3, kPageSize));
        const auto after = store.take_dirty();
        EXPECT(after.size() == 1 && !after.empty() && after[0] == phys[3]);
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(gpu_half.record_upload(cb, after)); }));
        EXPECT(gpu_holds(kVa + 3 * kPageSize, 0, kPageSize, 0xC3));
        // A finished readback merges nothing into a page while a kernel write into it is in flight.
        EXPECT(PhysicalPageStore::before_kernel_write(p3_cpu, 1));
        const PhysicalPage p3_page[1]{phys[3]};
        bool recorded = false;
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { recorded = gpu_half.record_readback(cb, p3_page); }));
        EXPECT(recorded && !gpu_half.finish_readback(p3_page));
        PhysicalPageStore::after_kernel_write(p3_cpu, 1);
        (void)upload_dirty();
        // Memory no store maps is left alone.
        int local = 0;
        EXPECT(PhysicalPageStore::before_kernel_write(&local, sizeof local));
        PhysicalPageStore::after_kernel_write(&local, sizeof local);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
    }

    // --- Unmapping a view leaves the other mapping alone ----------------------------------------------------------
    const PhysicalPage none[1]{PhysicalPageStore::kUnmapped};
    EXPECT(store.map(r2 + 2 * kPageSize, none));
    EXPECT(r1[kPageSize] == 0x41);

    // --- Freeing physical pages ----------------------------------------------------------------------------------
    {
        std::vector<PhysicalPage> two;
        EXPECT(store.allocate(2, two));
        auto* r4 = static_cast<std::uint8_t*>(store.reserve(1));
        const PhysicalPage mapped_page[1]{two.empty() ? 0 : two[0]};
        EXPECT(r4 && store.map(r4, mapped_page));
        std::memset(r4, 0x6B, kPageSize);
        std::memset(store.data(two[1]), 0x6C, kPageSize);
        EXPECT(upload_dirty());  // gives two[0] a GPU half
        const auto in_use = pool.pages_in_use();
        EXPECT(!store.release(two));  // two[0] is still mapped at r4
        const PhysicalPage unmap[1]{PhysicalPageStore::kUnmapped};
        EXPECT(store.map(r4, unmap));
        gpu_half.release(two);
        EXPECT(pool.pages_in_use() == in_use - 1);  // only two[0] had a GPU half
        EXPECT(store.release(two) && !store.allocated(two[0]) && !store.allocated(two[1]));
        EXPECT(!store.release(two));  // not allocated any more
        std::vector<PhysicalPage> back;
        EXPECT(store.allocate(2, back));
        EXPECT(back.size() == 2 && back[0] == two[0] && back[1] == two[1]);  // the lowest free pages
        for (PhysicalPage b : back) EXPECT(all_bytes(store.data(b), 0, kPageSize));
        EXPECT(store.release(back));
    }

    // --- Costs -----------------------------------------------------------------------------------------------------
    {
        constexpr std::uint32_t kN = 256;
        std::vector<PhysicalPage> many;
        EXPECT(store.allocate(kN, many));
        auto* r3 = static_cast<std::uint8_t*>(store.reserve(kN));
        if (!r3 || !store.map(r3, many)) {
            std::printf("FAIL mapping %u pages into a fresh reservation\n", kN);
            return 1;  // the writes below would fault on an unmapped range
        }
        (void)store.take_dirty();
        const auto f0 = store.write_faults();
        const auto t0 = std::chrono::steady_clock::now();
        for (std::uint32_t i = 0; i < kN; ++i) r3[std::size_t(i) * kPageSize] = std::uint8_t(i);
        const auto t1 = std::chrono::steady_clock::now();
        EXPECT(store.write_faults() - f0 == kN);
        const double fault_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / kN;
        const auto taken0 = std::chrono::steady_clock::now();
        const auto pages = store.take_dirty();
        const auto taken1 = std::chrono::steady_clock::now();
        EXPECT(pages.size() == kN);
        const auto u0 = std::chrono::steady_clock::now();
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(gpu_half.record_upload(cb, pages)); }));
        const auto u1 = std::chrono::steady_clock::now();
        std::printf("INFO first write to a clean page (fault + unprotect): %.2f us; take_dirty of %u pages: %.3f ms\n",
                    fault_us, kN, std::chrono::duration<double, std::milli>(taken1 - taken0).count());
        std::printf("INFO upload of %u dirty pages (%.0f MiB: memcpy to staging + GPU copy + wait): %.3f ms (%.2f "
                    "GB/s)\n",
                    kN, double(kN * kPageSize) / (1 << 20),
                    std::chrono::duration<double, std::milli>(u1 - u0).count(),
                    double(kN * kPageSize) / std::chrono::duration<double>(u1 - u0).count() / 1e9);
        for (std::uint32_t i = 0; i < kN; ++i) r3[std::size_t(i) * kPageSize + 64] = std::uint8_t(i + 1);
        const auto again = store.take_dirty();
        EXPECT(again.size() == kN);
        const auto w0 = std::chrono::steady_clock::now();
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(gpu_half.record_upload(cb, again)); }));
        const auto w1 = std::chrono::steady_clock::now();
        std::printf("INFO upload of the same %u pages with one changed word each: %.3f ms\n", kN,
                    std::chrono::duration<double, std::milli>(w1 - w0).count());
        for (std::uint32_t i = 0; i < kN; ++i)
            std::memset(r3 + std::size_t(i) * kPageSize, int(0x80 + i % 64), kPageSize);
        const auto rewritten = store.take_dirty();
        EXPECT(rewritten.size() == kN);
        const auto a0 = std::chrono::steady_clock::now();
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(gpu_half.record_upload(cb, rewritten)); }));
        const auto a1 = std::chrono::steady_clock::now();
        std::printf("INFO upload of the same %u pages rewritten whole: %.3f ms\n", kN,
                    std::chrono::duration<double, std::milli>(a1 - a0).count());
    }

    EXPECT(mapper.clear_page_mappings(kVa, 4, submit));
    queue.idle();
    if (d.validation) {
        std::printf("INFO validation errors: %u\n", d.validation_errors());
        EXPECT(d.validation_errors() == 0);
    }
    std::printf(g_failures ? "FAILED (%d)\n" : "PASS\n", g_failures);
    return g_failures ? 1 : 0;
}
