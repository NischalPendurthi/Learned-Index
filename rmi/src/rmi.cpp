#include "rmi.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <limits>

namespace rmi {
namespace {

// Closed-form least squares over data[lo..hi), with position as the label:
//
//   slope     = cov(x, y) / var(x)
//   intercept = mean_y - slope * mean_x
//
// Accumulated in long double.  Keys reach 2^63 and n reaches 2e8, so the
// uncentred normal equations lose far too much precision; the centred form
// plus a wider accumulator keeps the fit stable.
LinearModel fit_segment(const Key* keys, size_t lo, size_t hi) {
    LinearModel m;
    size_t count = hi - lo;
    if (count == 0) {
        m.slope = 0.0;
        m.intercept = static_cast<double>(lo);
        return m;
    }
    if (count == 1) {
        // var_x == 0: the model degenerates to "always this position", which
        // is exact for the single key it owns.
        m.slope = 0.0;
        m.intercept = static_cast<double>(lo);
        return m;
    }

    long double mean_x = 0.0L, mean_y = 0.0L;
    for (size_t i = lo; i < hi; ++i) {
        mean_x += static_cast<long double>(static_cast<double>(keys[i]));
        mean_y += static_cast<long double>(i);
    }
    mean_x /= static_cast<long double>(count);
    mean_y /= static_cast<long double>(count);

    long double cov_xy = 0.0L, var_x = 0.0L;
    for (size_t i = lo; i < hi; ++i) {
        long double dx = static_cast<long double>(static_cast<double>(keys[i])) - mean_x;
        long double dy = static_cast<long double>(i) - mean_y;
        cov_xy += dx * dy;
        var_x += dx * dx;
    }

    if (var_x <= 0.0L) {  // all keys in the bucket are identical
        m.slope = 0.0;
        m.intercept = static_cast<double>(mean_y);
        return m;
    }

    long double slope = cov_xy / var_x;
    long double intercept = mean_y - slope * mean_x;
    double s = static_cast<double>(slope);
    double b = static_cast<double>(intercept);
    if (!std::isfinite(s) || !std::isfinite(b) || s < 0.0) {
        // A negative slope would break the monotone routing that the
        // contiguous-bucket assignment depends on.  Fall back to a flat model.
        m.slope = 0.0;
        m.intercept = static_cast<double>(mean_y);
        return m;
    }
    m.slope = s;
    m.intercept = b;
    return m;
}

template <typename T>
double percentile(std::vector<T>& sorted_vals, double p) {
    if (sorted_vals.empty()) return 0.0;
    double idx = p * static_cast<double>(sorted_vals.size() - 1);
    size_t i = static_cast<size_t>(idx);
    if (i + 1 >= sorted_vals.size()) return static_cast<double>(sorted_vals.back());
    double frac = idx - static_cast<double>(i);
    return static_cast<double>(sorted_vals[i]) * (1.0 - frac) +
           static_cast<double>(sorted_vals[i + 1]) * frac;
}

}  // namespace

void RMI::train(const Key* keys, size_t n, size_t num_stage2) {
    auto t_start = std::chrono::high_resolution_clock::now();

    data_ = keys;
    n_ = n;
    k_ = (num_stage2 < 1) ? 1 : num_stage2;

    // --- Step 2: root model over every (key, position) pair ------------------
    root_ = fit_segment(keys, 0, n);

    // --- Step 3: route keys to buckets ---------------------------------------
    // bucket_for() is monotone non-decreasing in the key (slope >= 0, and both
    // truncation and floor preserve order), and the input array is sorted, so
    // each bucket owns a *contiguous* run of positions.  Storing the run
    // boundaries is exactly equivalent to materializing a key vector per
    // bucket, and avoids a second copy of the whole dataset -- which is what
    // makes N = 100M+ fit in memory at all.
    stage2_.assign(k_, LinearModel{});
    std::vector<size_t> start(k_ + 1, n);
    {
        size_t prev = 0;
        start[0] = 0;
        for (size_t i = 0; i < n; ++i) {
            size_t b = bucket_for(keys[i]);
            assert(b >= prev && "routing must be monotone over sorted keys");
            while (prev < b) start[++prev] = i;
        }
        while (prev < k_) start[++prev] = n;
    }

    // --- Step 4/5: fit each bucket and measure its error bounds --------------
    std::vector<size_t> bucket_counts(k_);
    std::vector<int64_t> err_widths;
    err_widths.reserve(k_);

    size_t empty = 0, singles = 0;
    long double width_sum = 0.0L;

    for (size_t b = 0; b < k_; ++b) {
        size_t lo = start[b], hi = start[b + 1];
        bucket_counts[b] = hi - lo;

        LinearModel m = fit_segment(keys, lo, hi);

        if (hi == lo) {
            // Empty bucket.  Only keys absent from the dataset can route here:
            // every key present at training time routes to the bucket that
            // owns it, and that bucket is non-empty by construction.  We still
            // point the dummy model at the boundary and widen it by one slot on
            // each side, so a lookup inspects the neighbouring keys rather than
            // trusting the routing blindly.
            m.slope = 0.0;
            m.intercept = static_cast<double>(lo);
            m.min_error = -1;
            m.max_error = 1;
            stage2_[b] = m;
            ++empty;
            continue;
        }
        if (hi - lo == 1) ++singles;

        int64_t min_err = std::numeric_limits<int64_t>::max();
        int64_t max_err = std::numeric_limits<int64_t>::min();
        for (size_t i = lo; i < hi; ++i) {
            // Identical arithmetic to the lookup path -- this is what makes the
            // stored bounds valid at query time.
            int64_t pred = m.predict(keys[i]);
            int64_t err = pred - static_cast<int64_t>(i);  // see LinearModel
            if (err < min_err) min_err = err;
            if (err > max_err) max_err = err;
        }
        m.min_error = min_err;
        m.max_error = max_err;
        stage2_[b] = m;

        int64_t width = max_err - min_err;
        err_widths.push_back(width);
        width_sum += static_cast<long double>(width);
    }

    // --- statistics -----------------------------------------------------------
    TrainStats s;
    s.n = n;
    s.num_models = k_;
    s.empty_models = empty;
    s.single_key_models = singles;
    s.index_size_bytes = index_size_bytes();

    std::vector<size_t> counts_sorted = bucket_counts;
    std::sort(counts_sorted.begin(), counts_sorted.end());
    s.keys_min = counts_sorted.front();
    s.keys_max = counts_sorted.back();
    s.keys_mean = static_cast<double>(n) / static_cast<double>(k_);
    s.keys_median = percentile(counts_sorted, 0.5);
    s.max_bucket_share = static_cast<double>(s.keys_max) / static_cast<double>(n);

    if (!err_widths.empty()) {
        std::sort(err_widths.begin(), err_widths.end());
        s.err_min = err_widths.front();
        s.err_max = err_widths.back();
        s.err_mean = static_cast<double>(width_sum / static_cast<long double>(err_widths.size()));
        s.err_p50 = percentile(err_widths, 0.50);
        s.err_p99 = percentile(err_widths, 0.99);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    s.train_seconds = std::chrono::duration<double>(t_end - t_start).count();
    stats_ = s;
}

}  // namespace rmi
