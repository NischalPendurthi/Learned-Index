#include "dataset.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace rmi {

const char* to_string(DatasetKind kind) {
    switch (kind) {
        case DatasetKind::Dense: return "dense";
        case DatasetKind::Lognormal: return "lognormal";
        case DatasetKind::UniformSparse: return "uniform_sparse";
    }
    return "?";
}

bool parse_dataset(const std::string& name, DatasetKind* out) {
    if (name == "dense") { *out = DatasetKind::Dense; return true; }
    if (name == "lognormal") { *out = DatasetKind::Lognormal; return true; }
    if (name == "uniform_sparse" || name == "uniform") { *out = DatasetKind::UniformSparse; return true; }
    return false;
}

namespace {

// Draw-sort-dedupe until n unique keys are collected.  Fine when the value
// space is far larger than n (lognormal), where collisions are rare.
template <typename Draw>
std::vector<Key> unique_sorted_by_rejection(size_t n, Draw draw) {
    std::vector<Key> v;
    size_t initial = n + n / 50 + 16;  // 2% slack
    v.reserve(initial);
    for (size_t i = 0; i < initial; ++i) v.push_back(draw());
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());

    while (v.size() < n) {
        size_t need = (n - v.size()) * 2 + 16;
        for (size_t i = 0; i < need; ++i) v.push_back(draw());
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    v.resize(n);  // still sorted and unique
    return v;
}

// Unique uniform draws from a *bounded* range.  Sort-and-dedupe is the wrong
// tool here: drawing 200M unique values from [0, 1e9) needs ~223M draws and a
// 1.8 GB sort.  A presence bitset over the range costs 125 MB flat and emits
// the keys already sorted.
std::vector<Key> unique_uniform_bounded(size_t n, uint64_t range, uint64_t seed) {
    if (n > range) return {};

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, range - 1);
    std::vector<bool> chosen(range, false);

    // Rejection sampling. Expected draws = range * ln(range / (range - n)),
    // which is only ~1.12n at the densest configuration used here (n/range=0.2).
    size_t placed = 0;
    while (placed < n) {
        uint64_t v = dist(rng);
        if (!chosen[v]) {
            chosen[v] = true;
            ++placed;
        }
    }

    std::vector<Key> out;
    out.reserve(n);
    for (uint64_t i = 0; i < range; ++i) {
        if (chosen[i]) out.push_back(i);
    }
    return out;
}

}  // namespace

std::vector<Key> generate(DatasetKind kind, size_t n, uint64_t seed) {
    switch (kind) {
        case DatasetKind::Dense: {
            std::vector<Key> v(n);
            for (size_t i = 0; i < n; ++i) v[i] = static_cast<Key>(i);
            return v;
        }
        case DatasetKind::Lognormal: {
            std::mt19937_64 rng(seed);
            std::lognormal_distribution<double> dist(0.0, 2.0);
            return unique_sorted_by_rejection(n, [&]() -> Key {
                double x = dist(rng) * LOGNORMAL_SCALE;
                if (!(x >= 0.0)) x = 0.0;
                if (x > 1.8e19) x = 1.8e19;
                return static_cast<Key>(x);
            });
        }
        case DatasetKind::UniformSparse: {
            if (n > UNIFORM_RANGE) {
                std::fprintf(stderr,
                             "error: cannot draw %zu unique keys from [0, %llu)\n", n,
                             static_cast<unsigned long long>(UNIFORM_RANGE));
                return {};
            }
            return unique_uniform_bounded(n, UNIFORM_RANGE, seed);
        }
    }
    return {};
}

std::vector<Key> sample_absent(const std::vector<Key>& sorted, size_t count, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, ~0ULL);
    std::vector<Key> out;
    out.reserve(count);
    while (out.size() < count) {
        Key k = dist(rng);
        if (!std::binary_search(sorted.begin(), sorted.end(), k)) out.push_back(k);
    }
    return out;
}

std::vector<Key> sample_present(const std::vector<Key>& sorted, size_t count, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, sorted.size() - 1);
    std::vector<Key> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = sorted[pick(rng)];
    std::shuffle(out.begin(), out.end(), rng);
    return out;
}

}  // namespace rmi
