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

// Draw a few extra keys, sort, remove duplicates, and repeat until we have n.
// Duplicates are rare, so one round is usually enough.
template <typename Draw>
vu64 unique_sorted(sz n, Draw draw) {
    vu64 v;
    v.reserve(n + n / 50 + 16);
    sz target = n + n / 50 + 16;
    rep(i, 0, target) v.push_back(draw());
    std::sort(ALL(v));
    v.erase(std::unique(ALL(v)), v.end());

    while (v.size() < n) {
        sz need = (n - v.size()) * 2 + 16;
        rep(i, 0, need) v.push_back(draw());
        std::sort(ALL(v));
        v.erase(std::unique(ALL(v)), v.end());
    }
    v.resize(n);  // no shrink_to_fit, it would briefly need 2x memory
    return v;
}

}  // namespace

vu64 generate_dataset(DatasetKind kind, sz n, u64 seed) {
    std::mt19937_64 rng(seed);

    switch (kind) {
        case DatasetKind::Dense: {
            vu64 v(n);
            rep(i, 0, n) v[i] = i;
            return v;
        }
        case DatasetKind::Lognormal: {
            // scaled by 1e9 so the median key is 1e9 and duplicates are rare
            std::lognormal_distribution<double> dist(0.0, 2.0);
            return unique_sorted(n, [&]() -> u64 {
                double x = dist(rng) * 1e9;
                if (!(x >= 0.0)) x = 0.0;
                if (x > 1.8e19) x = 1.8e19;
                return SC(u64, x);
            });
        }
        case DatasetKind::UniformSparse: {
            std::uniform_int_distribution<u64> dist(0, (1ULL << 63) - 1);
            return unique_sorted(n, [&]() -> u64 { return dist(rng); });
        }
    }
    return {};
}

vu64 sample_existing_keys(const vu64& data, sz count, u64 seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<sz> pick(0, data.size() - 1);
    vu64 out(count);
    rep(i, 0, count) out[i] = data[pick(rng)];
    std::shuffle(ALL(out), rng);
    return out;
}

vu64 sample_absent_keys(const vu64& data, sz count, u64 seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<u64> dist(0, ~0ULL);
    vu64 out;
    out.reserve(count);
    while (out.size() < count) {
        u64 k = dist(rng);
        if (!std::binary_search(ALL(data), k)) out.push_back(k);
    }
    return out;
}

}  // namespace lidx
