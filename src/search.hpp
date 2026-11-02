// Binary search used by both the RMI and the B-Tree, so the comparison
// between them isn't affected by the search code.
#pragma once

#include <cstddef>
#include "common.hpp"
#include <cstdlib>

namespace lidx {

// Same result as std::lower_bound, but the comparison compiles to a cmov
// instead of a branch, so there are no mispredictions. n must be >= 1.
inline sz lower_bound_branchless(const u64* arr, sz n, u64 key) {
    const u64* base = arr;
    while (n > 1) {
        sz half = n >> 1;
        base = (base[half] < key) ? base + half : base;  // cmov
        n -= half;
    }
    return SC(sz, base - arr) + (*base < key);
}

// same as above but counts elements <= key (like std::upper_bound)
inline sz upper_bound_branchless(const u64* arr, sz n, u64 key) {
    const u64* base = arr;
    while (n > 1) {
        sz half = n >> 1;
        base = (base[half] <= key) ? base + half : base;  // cmov
        n -= half;
    }
    return SC(sz, base - arr) + (*base <= key);
}

// Key array aligned to a 64-byte cache line, used for the B-Tree levels.
struct AlignedKeys {
    u64* keys = nullptr;
    sz n = 0;

    AlignedKeys() = default;
    AlignedKeys(const AlignedKeys&) = delete;
    AlignedKeys& operator=(const AlignedKeys&) = delete;
    AlignedKeys(AlignedKeys&& o) noexcept : keys(o.keys), n(o.n) { o.keys = nullptr; o.n = 0; }
    AlignedKeys& operator=(AlignedKeys&& o) noexcept {
        if (this != &o) { std::free(keys); keys = o.keys; n = o.n; o.keys = nullptr; o.n = 0; }
        return *this;
    }
    ~AlignedKeys() { std::free(keys); }

    void alloc(sz count) {
        std::free(keys);
        n = count;
        sz bytes = ((count * sizeof(u64) + 63) / 64) * 64;
        if (bytes == 0) bytes = 64;
        keys = SC(u64*, std::aligned_alloc(64, bytes));
    }
};

}  // namespace lidx
