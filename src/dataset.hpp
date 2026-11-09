#pragma once

#include "common.hpp"
#include <cstddef>
#include <vector>

namespace lidx {

enum class DatasetKind { Dense, Lognormal, UniformSparse };

const char* dataset_name(DatasetKind kind);

// n unique sorted keys.
//   Dense: 0 .. n-1
//   Lognormal: lognormal(0, 2) times 1e9, like the paper's synthetic dataset
//   UniformSparse: uniform in [0, 2^63)
vu64 generate_dataset(DatasetKind kind, sz n, u64 seed);

// Reads a SOSD binary file (a u64 count, then that many u64 keys), and returns
// its unique keys sorted. If n > 0 and the file has more, every stride-th key
// is kept so the sample still has the file's distribution.
vu64 load_sosd(const char* path, sz n);

// random keys from data (with repeats), shuffled so lookups aren't sequential
vu64 sample_existing_keys(const vu64& data, sz count, u64 seed);

// keys that are not in data
vu64 sample_absent_keys(const vu64& data, sz count, u64 seed);

}  // namespace lidx
