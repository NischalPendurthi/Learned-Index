#pragma once

#include <string>
#include <vector>

#include "common.h"

namespace rmi {

enum class DatasetKind {
    Dense,          // 0 .. N-1
    Lognormal,      // lognormal(mu=0, sigma=2), scaled to integers
    UniformSparse,  // unique uniform draws from [0, UNIFORM_RANGE)
};

const char* to_string(DatasetKind kind);
bool parse_dataset(const std::string& name, DatasetKind* out);

// Exactly n unique keys, sorted ascending.  Aborts (returns empty) if the
// request is impossible, e.g. n > UNIFORM_RANGE for UniformSparse.
std::vector<Key> generate(DatasetKind kind, size_t n, uint64_t seed);

// count keys guaranteed to be absent from `sorted`.
std::vector<Key> sample_absent(const std::vector<Key>& sorted, size_t count, uint64_t seed);

// count keys drawn from `sorted` with replacement, then shuffled.
std::vector<Key> sample_present(const std::vector<Key>& sorted, size_t count, uint64_t seed);

}  // namespace rmi
