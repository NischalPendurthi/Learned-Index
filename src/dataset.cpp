#include "dataset.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace lidx {

const char* dataset_name(DatasetKind kind) {
    switch (kind) {
        case DatasetKind::Dense: return "dense";
        case DatasetKind::Lognormal: return "lognormal";
        case DatasetKind::UniformSparse: return "uniform_sparse";
    }
    return "?";
}

namespace {

// Draws until exactly n unique sorted keys are collected.  Collisions are rare
// for both random generators used here, so this normally converges in one pass.
template <typename Draw>
std::vector<uint64_t> unique_sorted(size_t n, Draw draw) {
    std::vector<uint64_t> v;
    v.reserve(n + n / 50 + 16);
    size_t target = n + n / 50 + 16;
    for (size_t i = 0; i < target; ++i) v.push_back(draw());
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());

    while (v.size() < n) {
        size_t need = (n - v.size()) * 2 + 16;
        for (size_t i = 0; i < need; ++i) v.push_back(draw());
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    v.resize(n);  // still sorted and unique; no shrink, to avoid a 2x peak
    return v;
}

}  // namespace

std::vector<uint64_t> generate_dataset(DatasetKind kind, size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);

    switch (kind) {
        case DatasetKind::Dense: {
            std::vector<uint64_t> v(n);
            for (size_t i = 0; i < n; ++i) v[i] = i;
            return v;
        }
        case DatasetKind::Lognormal: {
            // lognormal(mu=0, sigma=2), scaled so the median key is 1e9.  The
            // scale keeps the heavy tail intact while making exact collisions
            // rare enough to dedupe cheaply.
            std::lognormal_distribution<double> dist(0.0, 2.0);
            return unique_sorted(n, [&]() -> uint64_t {
                double x = dist(rng) * 1e9;
                if (!(x >= 0.0)) x = 0.0;
                if (x > 1.8e19) x = 1.8e19;
                return static_cast<uint64_t>(x);
            });
        }
        case DatasetKind::UniformSparse: {
            std::uniform_int_distribution<uint64_t> dist(0, (1ULL << 63) - 1);
            return unique_sorted(n, [&]() -> uint64_t { return dist(rng); });
        }
    }
    return {};
}

std::vector<uint64_t> sample_existing_keys(const std::vector<uint64_t>& data, size_t count,
                                           uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, data.size() - 1);
    std::vector<uint64_t> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = data[pick(rng)];
    std::shuffle(out.begin(), out.end(), rng);  // no sequential access pattern
    return out;
}

std::vector<uint64_t> sample_absent_keys(const std::vector<uint64_t>& data, size_t count,
                                         uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, ~0ULL);
    std::vector<uint64_t> out;
    out.reserve(count);
    while (out.size() < count) {
        uint64_t k = dist(rng);
        if (!std::binary_search(data.begin(), data.end(), k)) out.push_back(k);
    }
    return out;
}

}  // namespace lidx
