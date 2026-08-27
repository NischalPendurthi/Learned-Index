#pragma once

#include <cstdint>

#include "common.h"

namespace rmi {

// Plain binary search over the inclusive range data[lo..hi].
// Returns the index of key, or -1 if absent.
inline int64_t binary_search_bounded(const Key* data, int64_t lo, int64_t hi, Key key) {
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;  // no overflow
        Key v = data[mid];
        if (v == key) return mid;
        if (v < key) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

// Exponential ("galloping") search over the inclusive range data[lo..hi],
// seeded at `start` -- normally the model's predicted position.  When the model
// is accurate the key is found after a couple of steps, without paying
// log2(hi - lo) probes.  Returns the index of key, or -1 if absent.
inline int64_t exponential_search(const Key* data, int64_t lo, int64_t hi, int64_t start,
                                  Key key) {
    if (lo > hi) return -1;
    if (start < lo) start = lo;
    if (start > hi) start = hi;

    Key v = data[start];
    if (v == key) return start;

    int64_t step = 1;
    if (v < key) {
        // Walk right until we overshoot, then binary search the bracket.
        int64_t prev = start;
        int64_t cur = start + step;
        while (cur <= hi && data[cur] < key) {
            prev = cur;
            step <<= 1;
            cur = start + step;
        }
        return binary_search_bounded(data, prev + 1, cur < hi ? cur : hi, key);
    } else {
        // Walk left.
        int64_t prev = start;
        int64_t cur = start - step;
        while (cur >= lo && data[cur] > key) {
            prev = cur;
            step <<= 1;
            cur = start - step;
        }
        return binary_search_bounded(data, cur > lo ? cur : lo, prev - 1, key);
    }
}

}  // namespace rmi
