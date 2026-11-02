#pragma once

#include <cstddef>
#include <cstdint>

namespace rmi {

using Key = uint64_t;
using Position = int64_t;

constexpr size_t DEFAULT_N = 1'000'000;
constexpr size_t DEFAULT_K = 10'000;       // stage-2 models
constexpr uint64_t DEFAULT_SEED = 42;

// lognormal(0, 2) samples (paper section 3.7.1) are scaled by this and truncated
constexpr double LOGNORMAL_SCALE = 1e9;

constexpr uint64_t UNIFORM_RANGE = 1'000'000'000ULL;

// absent keys probed in test 2
constexpr size_t ABSENT_PROBES = 1000;

}  // namespace rmi
