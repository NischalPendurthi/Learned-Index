#pragma once

#include <cstdint>

#include "common.h"

namespace rmi {

// Linear model: pos ~= slope * key + intercept.
// error = predicted - actual, so positive means the model guessed too high.
// Since actual = predicted - error, the search range is
// [predicted - max_error, predicted - min_error] (looks backwards at first).
struct LinearModel {
    double slope = 0.0;
    double intercept = 0.0;
    int64_t min_error = 0;  // most negative error
    int64_t max_error = 0;  // most positive error

    inline int64_t predict(Key key) const {
        double p = slope * static_cast<double>(key) + intercept;
        // clamp so the double to int64 cast is never out of range (this form also catches NaN)
        if (!(p > -9.0e18)) p = -9.0e18;
        if (p > 9.0e18) p = 9.0e18;
        return static_cast<int64_t>(p);
    }

    // inclusive [lo, hi] that holds the key's position if it is in this bucket
    inline void search_bounds(Key key, int64_t n, int64_t* lo, int64_t* hi) const {
        int64_t pred = predict(key);
        int64_t l = pred - max_error;
        int64_t h = pred - min_error;
        if (l < 0) l = 0;
        if (h > n - 1) h = n - 1;
        *lo = l;
        *hi = h;
    }
};

static_assert(sizeof(LinearModel) == 32, "LinearModel is the unit of reported index size");

}  // namespace rmi
