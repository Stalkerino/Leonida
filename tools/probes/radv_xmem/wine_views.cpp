// Runs inside Wine. Only documented Win32 file mappings and placeholder APIs.
#include <windows.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include "pages.h"

using namespace synthetic;
#define CHECK(x)                                                                                                     \
    do {                                                                                                             \
        if (!(x)) {                                                                                                  \
            std::fprintf(stderr, "%s failed, Win32 error %lu\n", #x, GetLastError());                                \
            return 1;                                                                                                \
        }                                                                                                            \
    } while (false)

int main(int argc, char **argv) {
    if (argc != 3 || (std::strcmp(argv[2], "write") && std::strcmp(argv[2], "read")))
        return 2;
    bool writer = !std::strcmp(argv[2], "write");
    HANDLE file = CreateFileA(argv[1], GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    CHECK(file != INVALID_HANDLE_VALUE);
    HANDLE section = CreateFileMappingW(file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    CHECK(section);
    using Alloc2 = void *(WINAPI *)(HANDLE, void *, SIZE_T, ULONG, ULONG, void *, ULONG);
    using Map3 = void *(WINAPI *)(HANDLE, HANDLE, void *, ULONG64, SIZE_T, ULONG, ULONG, void *, ULONG);
    using Unmap2 = BOOL(WINAPI *)(HANDLE, void *, ULONG);
    HMODULE kb = GetModuleHandleW(L"kernelbase.dll");
    auto alloc = reinterpret_cast<Alloc2>(GetProcAddress(kb, "VirtualAlloc2"));
    auto map = reinterpret_cast<Map3>(GetProcAddress(kb, "MapViewOfFile3"));
    auto unmap = reinterpret_cast<Unmap2>(GetProcAddress(kb, "UnmapViewOfFile2"));
    CHECK(alloc && map && unmap);
    constexpr DWORD reserve_placeholder = 0x40000, replace_placeholder = 0x4000, preserve_placeholder = 2;
    HANDLE process = GetCurrentProcess();
    auto *base =
        static_cast<volatile uint32_t *>(map(section, process, nullptr, 0, 3 * page, 0, PAGE_READWRITE, nullptr, 0));
    CHECK(base);
    void *address = alloc(process, nullptr, page, MEM_RESERVE | reserve_placeholder, PAGE_NOACCESS, nullptr, 0);
    CHECK(address);
    auto *view = static_cast<volatile uint32_t *>(address);
    auto replace = [&](unsigned offset) {
        return map(section, process, address, offset, page, replace_placeholder, PAGE_READWRITE, nullptr, 0) ==
               address;
    };
    CHECK(replace(0));
    auto *guard_begin = base + page / sizeof(uint32_t);
    auto *guard_end = base + 2 * page / sizeof(uint32_t);
    if (writer) {
        CHECK(view[0] == 0 && base[0] == 0);
        view[0] = cpu_a;
        std::fill(guard_begin, guard_end, guard);
    } else {
        CHECK(view[0] == gpu_a);
    }
    CHECK(unmap(process, address, preserve_placeholder));
    CHECK(replace(2 * page));
    if (writer) {
        CHECK(view[0] == 0);
        view[0] = cpu_b;
    } else {
        CHECK(view[0] == gpu_b);
    }
    CHECK(base[0] == (writer ? cpu_a : gpu_a));
    CHECK(std::all_of(guard_begin, guard_end, [](uint32_t word) { return word == guard; }));
    CHECK(base[2 * page / sizeof(uint32_t)] == (writer ? cpu_b : gpu_b));
    CHECK(unmap(process, address, preserve_placeholder));
    CHECK(replace(0));
    CHECK(view[0] == (writer ? cpu_a : gpu_a));
    CHECK(unmap(process, address, preserve_placeholder));
    CHECK(VirtualFree(address, 0, MEM_RELEASE));
    CHECK(UnmapViewOfFile(const_cast<uint32_t *>(base)));
    CloseHandle(section);
    CloseHandle(file);
    std::printf("Wine %s: canonical view + fixed-address remap A -> B -> A and guard passed\n", argv[2]);
}
