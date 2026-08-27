#pragma once

#include <cstddef>
#include <cstdint>

namespace rmi {

using Key = uint64_t;
using Position = int64_t;

// Defaults; every one of these is overridable from the command line.
constexpr size_t DEFAULT_N = 1'000'000;    // start small for debugging
constexpr size_t DEFAULT_K = 10'000;       // stage-2 models
constexpr uint64_t DEFAULT_SEED = 42;

// LOGNORMAL: lognormal(mu=0, sigma=2) samples are scaled by this factor and
// truncated to integers.  1e9 puts the median key at 1e9 and keeps exact
// collisions rare enough to deduplicate cheaply.
constexpr double LOGNORMAL_SCALE = 1e9;

// UNIFORM_SPARSE: unique values drawn uniformly from [0, UNIFORM_RANGE).
constexpr uint64_t UNIFORM_RANGE = 1'000'000'000ULL;

// Number of absent keys probed by correctness test 2.
constexpr size_t ABSENT_PROBES = 1000;

}  // namespace rmi
