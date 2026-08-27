// Branch-free search primitives shared by the RMI and the B-Tree baseline.
//
// Both indexes use the *same* primitives so that any measured difference comes
// from the index structure, not from the search implementation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace lidx {

// Number of elements in arr[0, n) that are < key.  Equivalent to
// std::lower_bound(arr, arr + n, key) - arr, but the branch is compiled to a
// cmov so it does not depend on the (unpredictable) comparison outcome.
// Requires n >= 1.
inline size_t lower_bound_branchless(const uint64_t* arr, size_t n, uint64_t key) {
    const uint64_t* base = arr;
    while (n > 1) {
        size_t half = n >> 1;
        base = (base[half] < key) ? base + half : base;  // cmov
        n -= half;
    }
    return static_cast<size_t>(base - arr) + (*base < key);
}

// Number of elements in arr[0, n) that are <= key (an upper bound).
// Requires n >= 1.
inline size_t upper_bound_branchless(const uint64_t* arr, size_t n, uint64_t key) {
    const uint64_t* base = arr;
    while (n > 1) {
        size_t half = n >> 1;
        base = (base[half] <= key) ? base + half : base;  // cmov
        n -= half;
    }
    return static_cast<size_t>(base - arr) + (*base <= key);
}

// 64-byte (cache-line) aligned array of keys.  Used for B-Tree internal levels
// so that a node's keys are packed contiguously and start on a line boundary.
struct AlignedKeys {
    uint64_t* keys = nullptr;
    size_t n = 0;

    AlignedKeys() = default;
    AlignedKeys(const AlignedKeys&) = delete;
    AlignedKeys& operator=(const AlignedKeys&) = delete;
    AlignedKeys(AlignedKeys&& o) noexcept : keys(o.keys), n(o.n) { o.keys = nullptr; o.n = 0; }
    AlignedKeys& operator=(AlignedKeys&& o) noexcept {
        if (this != &o) { std::free(keys); keys = o.keys; n = o.n; o.keys = nullptr; o.n = 0; }
        return *this;
    }
    ~AlignedKeys() { std::free(keys); }

    void alloc(size_t count) {
        std::free(keys);
        n = count;
        size_t bytes = ((count * sizeof(uint64_t) + 63) / 64) * 64;
        if (bytes == 0) bytes = 64;
        keys = static_cast<uint64_t*>(std::aligned_alloc(64, bytes));
    }
};

}  // namespace lidx
