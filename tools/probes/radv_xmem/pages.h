#pragma once

#include <stdint.h>

namespace synthetic {
constexpr unsigned page = 65536;
constexpr uint32_t cpu_a = 0xabcd, cpu_b = 0x1234;
constexpr uint32_t gpu_a = 0x5441, gpu_b = 0x6029;
constexpr uint32_t guard = 0x85b4;
} // namespace synthetic
