# Importing remappable Wine pages into a native RADV buffer

This standalone synthetic probe checks whether a Linux Vulkan consumer can see two non-adjacent 64 KiB frames that a Windows process aliases and remaps under Wine. All three pages fit in 192 KiB. The probe uses one transfer submission and creates no files containing game data.

From this directory, build in a temporary directory. You need Linux Vulkan and UAPI development headers, a Vulkan loader, a C++17 compiler, x86-64 MinGW, Wine with placeholder mapping support, and access to the existing `/dev/udmabuf` device. Wine's usual Z drive must expose procfs to the Windows test process.

```sh
build_dir=$(mktemp -d)
c++ -std=c++17 -O2 host.cpp -lvulkan -o "$build_dir/probe"
x86_64-w64-mingw32-g++ -std=c++17 -O2 -static wine_views.cpp -o "$build_dir/views.exe"
"$build_dir/probe" "$build_dir/views.exe"
```

The host selects a RADV device. If there are several, set `PROBE_DEVICE_ID` to the hexadecimal PCI device ID (for example `7550`). `WINE` can name a different Wine executable, and the child inherits `WINEPREFIX`.

1. The host creates a shrink-sealed memfd containing frames A, a guard page, and B. `UDMABUF_CREATE_LIST` exposes B then A in one 128 KiB DMA-BUF. Wine opens the same memfd through procfs and creates a normal Win32 file mapping.
2. The Windows writer maps a stable view and a second view in a 64 KiB placeholder. It moves that view from A to B and back at the same address, writes different words and checks the stable alias and the guard.
3. The native host imports the DMA-BUF as a dedicated allocation for a non-sparse Vulkan storage buffer with a nonzero device address. The GPU copies one word from each GPU page to an independent readback buffer, then fills each with a different reply. The host waits on a fence and verifies the order B, A. A second Wine process repeats the alias and remap checks against the replies. CPU access is bracketed by `DMA_BUF_IOCTL_SYNC`. [Linux DMA-BUF interface](https://docs.kernel.org/driver-api/dma-buf.html)

The GPU imports the file descriptor on a **native Linux Vulkan device**. Wine 11.22 keeps the file-descriptor import and placed-mapping extensions internal. Its opaque Win32 GPU handle also fails `MapViewOfFile3` with error 6 on the tested RADV system. A production backend would need a route to the device that executes the title's commands, a frame registry, and ordering for remaps and retirement of live GPU bindings. A remapped guest view does not change an already-created DMA-BUF list. This test uses two buffer frames, ordered CPU/GPU accesses and no shader or image. [Wine Vulkan extension list](https://github.com/wine-mirror/wine/blob/wine-11.22/dlls/winevulkan/make_vulkan#L122-L138)

The kernel limits each export's list length and total size and pins its selected pages while the exported object is alive. Adjacent runs can share one list item. The relevant parameters are readable under `/sys/module/udmabuf/parameters/`. The probe returns 77 if the device node is unavailable or the selected RADV GPU is absent. [Linux UDMABUF driver](https://github.com/torvalds/linux/blob/v7.3/drivers/dma-buf/udmabuf.c)
