#pragma once

#include <cstdint>

#include "common.h"

namespace rmi {

// One linear regression model: pos ~= slope * key + intercept.
//
// ERROR SIGN CONVENTION -- read this before changing anything.
//
//   error = predicted_position - actual_position
//
// so a *positive* error means the model predicted too high (over-prediction)
// and a *negative* error means it predicted too low (under-prediction), which
// is what the min_error/max_error field names below describe.  Inverting that:
//
//   actual = predicted - error,   error in [min_error, max_error]
//   => actual in [predicted - max_error, predicted - min_error]
//
// That is the search interval, and it is the reverse of what you might expect.
// See search_bounds() and the note in README.md.
struct LinearModel {
    double slope = 0.0;
    double intercept = 0.0;
    int64_t min_error = 0;  // most negative error (worst under-prediction)
    int64_t max_error = 0;  // most positive error (worst over-prediction)

    inline int64_t predict(Key key) const {
        double p = slope * static_cast<double>(key) + intercept;
        // Guard the double->int64 conversion, which is undefined out of range.
        // Cannot trigger for a model fit on real data; it only makes a
        // degenerate model safe instead of undefined.  The comparison order
        // also rejects NaN.
        if (!(p > -9.0e18)) p = -9.0e18;
        if (p > 9.0e18) p = 9.0e18;
        return static_cast<int64_t>(p);
    }

    // Inclusive search interval [lo, hi] guaranteed to contain the key's true
    // position, if the key is present in the model's bucket.
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
