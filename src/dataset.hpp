#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace lidx {

enum class DatasetKind { Dense, Lognormal, UniformSparse };

const char* dataset_name(DatasetKind kind);

// Exactly n unique keys, sorted ascending.
//   Dense         : 0 .. n-1 (a perfectly linear CDF)
//   Lognormal     : lognormal(mu=0, sigma=2) scaled to integers up to ~1B,
//                   matching the synthetic dataset of the paper (Section 5.1)
//   UniformSparse : uniform over [0, 2^63)
std::vector<uint64_t> generate_dataset(DatasetKind kind, size_t n, uint64_t seed);

// count keys drawn uniformly from data with replacement, then shuffled, so the
// access pattern carries no sequential locality.
std::vector<uint64_t> sample_existing_keys(const std::vector<uint64_t>& data, size_t count,
                                           uint64_t seed);

// count keys guaranteed *not* to be in data.
std::vector<uint64_t> sample_absent_keys(const std::vector<uint64_t>& data, size_t count,
                                         uint64_t seed);

}  // namespace lidx
