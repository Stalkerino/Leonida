// Imports offsets [2 * page, 0] of one shared memfd, as one ordinary Vulkan buffer.
#include <vulkan/vulkan.h>
#include <linux/dma-buf.h>
#include <linux/udmabuf.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "pages.h"

using namespace synthetic;
constexpr size_t buffer_size = 2 * page;
constexpr auto dma_buf = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
#define VKOK(x)                                                                                                      \
    do {                                                                                                             \
        const auto r = (x);                                                                                          \
        if (r != VK_SUCCESS) {                                                                                       \
            std::fprintf(stderr, "%s: VkResult %d\n", #x, r);                                                        \
            return 1;                                                                                                \
        }                                                                                                            \
    } while (false)
#define REQUIRE(x)                                                                                                   \
    do {                                                                                                             \
        if (!(x)) {                                                                                                  \
            std::fprintf(stderr, "check failed: %s\n", #x);                                                          \
            return 1;                                                                                                \
        }                                                                                                            \
    } while (false)

static bool sync_cpu(int fd, uint64_t phase) {
    dma_buf_sync sync{DMA_BUF_SYNC_RW | phase};
    int r;
    do {
        r = ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
    } while (r < 0 && (errno == EINTR || errno == EAGAIN));
    return r == 0;
}

static bool run_guest(const char *guest, int section, int dma, const char *phase) {
    char path[80];
    std::snprintf(path, sizeof(path), "Z:\\proc\\%ld\\fd\\%d", static_cast<long>(getpid()), section);
    if (!sync_cpu(dma, DMA_BUF_SYNC_START))
        return false;
    const char *wine = std::getenv("WINE");
    if (!wine || !*wine)
        wine = "wine";
    pid_t pid = fork();
    if (pid == 0) {
        execlp(wine, wine, guest, path, phase, static_cast<char *>(nullptr));
        _exit(127);
    }
    int status = -1;
    pid_t done = -1;
    if (pid > 0) {
        do {
            done = waitpid(pid, &status, 0);
        } while (done < 0 && errno == EINTR);
    }
    return sync_cpu(dma, DMA_BUF_SYNC_END) && done == pid && done > 0 && WIFEXITED(status) &&
           WEXITSTATUS(status) == 0;
}

static int gpu_roundtrip(int dma) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    VkInstance inst;
    VKOK(vkCreateInstance(&ci, nullptr, &inst));
    uint32_t n = 0;
    VKOK(vkEnumeratePhysicalDevices(inst, &n, nullptr));
    std::vector<VkPhysicalDevice> devices(n);
    VKOK(vkEnumeratePhysicalDevices(inst, &n, devices.data()));
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    const char *wanted = std::getenv("PROBE_DEVICE_ID");
    for (auto candidate : devices) {
        VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
        vkGetPhysicalDeviceProperties2(candidate, &props);
        if (driver.driverID != VK_DRIVER_ID_MESA_RADV)
            continue;
        if (wanted && props.properties.deviceID != std::strtoul(wanted, nullptr, 16))
            continue;
        pd = candidate;
        std::printf("Testing %s\n", props.properties.deviceName);
        break;
    }
    if (!pd) {
        std::puts("No matching RADV device");
        return 77;
    }
    constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkPhysicalDeviceExternalBufferInfo info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
    info.usage = usage;
    info.handleType = dma_buf;
    VkExternalBufferProperties props{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
    vkGetPhysicalDeviceExternalBufferProperties(pd, &info, &props);
    REQUIRE(props.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, families.data());
    uint32_t family = 0;
    while (family < n && !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
        ++family;
    REQUIRE(family < n);
    float priority = 0.5f;
    VkDeviceQueueCreateInfo qc{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qc.queueFamilyIndex = family;
    qc.queueCount = 1;
    qc.pQueuePriorities = &priority;
    const char *extensions[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                                VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME};
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bda.bufferDeviceAddress = VK_TRUE;
    VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &bda};
    dc.queueCreateInfoCount = 1;
    dc.pQueueCreateInfos = &qc;
    dc.enabledExtensionCount = 2;
    dc.ppEnabledExtensionNames = extensions;
    VkDevice device;
    VKOK(vkCreateDevice(pd, &dc, nullptr, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto choose_type = [&](uint32_t mask, VkMemoryPropertyFlags flags) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((mask & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags)
                return i;
        return UINT32_MAX;
    };
    VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = dma_buf;
    VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    bc.size = buffer_size;
    bc.usage = usage;
    VkBuffer physical, readback;
    VKOK(vkCreateBuffer(device, &bc, nullptr, &physical));
    bc.pNext = nullptr;
    bc.size = 8;
    bc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VKOK(vkCreateBuffer(device, &bc, nullptr, &readback));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, physical, &req);
    REQUIRE(req.size == buffer_size);
    auto fd_props =
        reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryFdPropertiesKHR"));
    REQUIRE(fd_props);
    VkMemoryFdPropertiesKHR fm{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    VKOK(fd_props(device, dma_buf, dma, &fm));
    uint32_t t = choose_type(req.memoryTypeBits & fm.memoryTypeBits, 0);
    REQUIRE(t != UINT32_MAX);
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = physical;
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, &dedicated};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkImportMemoryFdInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, &flags};
    import.handleType = dma_buf;
    import.fd = fcntl(dma, F_DUPFD_CLOEXEC, 0);
    REQUIRE(import.fd >= 0);
    VkMemoryAllocateInfo al{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &import};
    al.allocationSize = req.size;
    al.memoryTypeIndex = t;
    VkDeviceMemory mem, result;
    VKOK(vkAllocateMemory(device, &al, nullptr, &mem));
    VKOK(vkBindBufferMemory(device, physical, mem, 0));
    VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address.buffer = physical;
    REQUIRE(vkGetBufferDeviceAddress(device, &address));
    vkGetBufferMemoryRequirements(device, readback, &req);
    t = choose_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    REQUIRE(t != UINT32_MAX);
    al.pNext = nullptr;
    al.allocationSize = req.size;
    al.memoryTypeIndex = t;
    VKOK(vkAllocateMemory(device, &al, nullptr, &result));
    VKOK(vkBindBufferMemory(device, readback, result, 0));
    void *mapped = nullptr;
    VKOK(vkMapMemory(device, result, 0, VK_WHOLE_SIZE, 0, &mapped));
    VkCommandPoolCreateInfo pc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pc.queueFamilyIndex = family;
    VkCommandPool pool;
    VKOK(vkCreateCommandPool(device, &pc, nullptr, &pool));
    VkCommandBufferAllocateInfo ac{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ac.commandBufferCount = 1;
    ac.commandPool = pool;
    VkCommandBuffer cmd;
    VKOK(vkAllocateCommandBuffers(device, &ac, &cmd));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKOK(vkBeginCommandBuffer(cmd, &begin));
    auto barrier = [&](VkPipelineStageFlags a, VkPipelineStageFlags b, VkAccessFlags src, VkAccessFlags dst) {
        VkMemoryBarrier br{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        br.srcAccessMask = src;
        br.dstAccessMask = dst;
        vkCmdPipelineBarrier(cmd, a, b, 0, 1, &br, 0, nullptr, 0, nullptr);
    };
    barrier(VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferCopy ranges[] = {{0, 0, 4}, {page, 4, 4}};
    vkCmdCopyBuffer(cmd, physical, readback, 2, ranges);
    barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdFillBuffer(cmd, physical, 0, 4, gpu_b);
    vkCmdFillBuffer(cmd, physical, page, 4, gpu_a);
    barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_HOST_READ_BIT);
    VKOK(vkEndCommandBuffer(cmd));
    VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    sub.commandBufferCount = 1;
    sub.pCommandBuffers = &cmd;
    VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VKOK(vkCreateFence(device, &fc, nullptr, &fence));
    VKOK(vkQueueSubmit(queue, 1, &sub, fence));
    VKOK(vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ull));
    const auto *read = static_cast<const uint32_t *>(mapped);
    REQUIRE(read[0] == cpu_b && read[1] == cpu_a);
    std::puts("GPU copied Wine's words in reverse frame order and wrote its replies");
    vkUnmapMemory(device, result);
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyBuffer(device, readback, nullptr);
    vkDestroyBuffer(device, physical, nullptr);
    vkFreeMemory(device, result, nullptr);
    vkFreeMemory(device, mem, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int node = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
    if (node < 0) {
        std::perror("open /dev/udmabuf");
        return 77;
    }
    int section = memfd_create("synthetic-xmem-pages", MFD_ALLOW_SEALING | MFD_CLOEXEC);
    REQUIRE(section >= 0 && ftruncate(section, 3 * page) == 0);
    REQUIRE(fcntl(section, F_ADD_SEALS, F_SEAL_SHRINK) >= 0);
    struct {
        uint32_t flags, count;
        udmabuf_create_item items[2];
    } list{UDMABUF_FLAGS_CLOEXEC,
           2,
           {{static_cast<uint32_t>(section), 0, 2 * page, page}, {static_cast<uint32_t>(section), 0, 0, page}}};
    static_assert(offsetof(decltype(list), items) == sizeof(udmabuf_create_list));
    int dma = ioctl(node, UDMABUF_CREATE_LIST, &list);
    if (dma < 0) {
        std::perror("UDMABUF_CREATE_LIST");
        return 1;
    }
    REQUIRE(run_guest(argv[1], section, dma, "write"));
    const int r = gpu_roundtrip(dma);
    if (r)
        return r;
    REQUIRE(run_guest(argv[1], section, dma, "read"));
    close(dma);
    close(section);
    close(node);
    std::puts("PASS two non-adjacent frames back one non-sparse Vulkan buffer; middle guard preserved");
}
