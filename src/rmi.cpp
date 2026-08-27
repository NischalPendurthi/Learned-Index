#include "rmi.hpp"

#include <cassert>
#include <limits>

namespace lidx {
namespace {

// Least squares fit of y = slope*x + intercept, computed in centred form with
// long double accumulators.  Keys reach ~2^63, so the naive normal equations
// lose too much precision.
struct Fit {
    double slope;
    double intercept;
};

// Fits over a range of the sorted array, with x supplied by a transform.
template <typename XForm>
Fit fit_range(const uint64_t* data, size_t lo, size_t hi, XForm xf) {
    size_t cnt = hi - lo;
    if (cnt == 0) return {0.0, static_cast<double>(lo)};
    if (cnt == 1) return {0.0, static_cast<double>(lo)};

    long double mx = 0.0L, my = 0.0L;
    for (size_t i = lo; i < hi; ++i) {
        mx += xf(data[i]);
        my += static_cast<long double>(i);
    }
    mx /= cnt;
    my /= cnt;

    long double sxx = 0.0L, sxy = 0.0L;
    for (size_t i = lo; i < hi; ++i) {
        long double dx = static_cast<long double>(xf(data[i])) - mx;
        long double dy = static_cast<long double>(i) - my;
        sxx += dx * dx;
        sxy += dx * dy;
    }

    if (sxx <= 0.0L) return {0.0, static_cast<double>(lo)};
    long double slope = sxy / sxx;
    long double intercept = my - slope * mx;
    double s = static_cast<double>(slope);
    double b = static_cast<double>(intercept);
    if (!std::isfinite(s) || !std::isfinite(b) || s <= 0.0) return {0.0, static_cast<double>(lo)};
    return {s, b};
}

struct XLinear {
    long double operator()(uint64_t k) const { return static_cast<long double>(static_cast<double>(k)); }
};
struct XLog {
    long double operator()(uint64_t k) const {
        return static_cast<long double>(std::log(static_cast<double>(k) + 1.0));
    }
};

}  // namespace

void RMI::build(const uint64_t* sorted_data, size_t n, size_t num_models) {
    data_ = sorted_data;
    N_ = n;
    if (num_models < 1) num_models = 1;
    models_.assign(num_models, LinearModel{0.0, 0.0, 0, 0});

    const double scale = static_cast<double>(num_models) / static_cast<double>(n);

    // --- Stage 1 -----------------------------------------------------------
    // Fit both candidate transforms.  Each maps a key to a position in [0, N),
    // which we rescale to a model index in [0, K).
    struct Cand {
        Stage1Kind kind;
        double slope, intercept;
        long double cost;
    };
    Cand cands[2];

    {
        Fit f = fit_range(sorted_data, 0, n, XLinear{});
        cands[0] = {Stage1Kind::Linear, f.slope * scale, f.intercept * scale, 0.0L};
        Fit g = fit_range(sorted_data, 0, n, XLog{});
        cands[1] = {Stage1Kind::LogLinear, g.slope * scale, g.intercept * scale, 0.0L};
    }

    // Score each candidate by sum of squared per-model key counts.  A stage-2
    // model's search window grows with the number of keys routed to it, and a
    // uniformly sampled query hits a model in proportion to its count, so
    // sum(count^2) is a cheap proxy for expected search work.  Degenerate
    // candidates (slope <= 0) are rejected outright.
    std::vector<uint32_t> counts(num_models);
    int best = -1;
    for (int c = 0; c < 2; ++c) {
        if (!(cands[c].slope > 0.0) || !std::isfinite(cands[c].intercept)) {
            cands[c].cost = std::numeric_limits<long double>::infinity();
            continue;
        }
        std::fill(counts.begin(), counts.end(), 0u);
        s1_kind_ = cands[c].kind;
        s1_slope_ = cands[c].slope;
        s1_intercept_ = cands[c].intercept;
        for (size_t i = 0; i < n; ++i) counts[model_index(sorted_data[i])]++;
        long double cost = 0.0L;
        for (size_t m = 0; m < num_models; ++m) {
            long double c2 = static_cast<long double>(counts[m]);
            cost += c2 * c2;
        }
        cands[c].cost = cost;
        if (best < 0 || cost < cands[best].cost) best = c;
    }

    if (best < 0) {
        // Fall back to a linear spline between the smallest and largest key.
        s1_kind_ = Stage1Kind::Linear;
        double k0 = static_cast<double>(sorted_data[0]);
        double k1 = static_cast<double>(sorted_data[n - 1]);
        double span = (k1 > k0) ? (k1 - k0) : 1.0;
        s1_slope_ = static_cast<double>(num_models) / span;
        s1_intercept_ = -s1_slope_ * k0;
    } else {
        s1_kind_ = cands[best].kind;
        s1_slope_ = cands[best].slope;
        s1_intercept_ = cands[best].intercept;
    }

    // --- Stage 2 -----------------------------------------------------------
    // Stage 1 is monotone in the key, so the chosen model index is
    // non-decreasing over the sorted array: each model owns a contiguous run.
    std::vector<size_t> start(num_models + 1, n);
    {
        size_t prev = 0;
        start[0] = 0;
        for (size_t i = 0; i < n; ++i) {
            size_t m = model_index(sorted_data[i]);
            assert(m >= prev);
            while (prev < m) start[++prev] = i;
        }
        while (prev < num_models) start[++prev] = n;
    }

    long double total_abs_err = 0.0L;
    long double total_window = 0.0L;
    max_window_ = 0;

    for (size_t m = 0; m < num_models; ++m) {
        size_t lo = start[m], hi = start[m + 1];
        Fit f = fit_range(sorted_data, lo, hi, XLinear{});
        LinearModel& lm = models_[m];
        lm.slope = f.slope;
        lm.intercept = f.intercept;
        lm.min_err = 0;
        lm.max_err = 0;

        if (hi == lo) continue;  // empty: only unseen keys can route here

        int64_t mn = std::numeric_limits<int64_t>::max();
        int64_t mx = std::numeric_limits<int64_t>::min();
        for (size_t i = lo; i < hi; ++i) {
            int64_t pred = predict_pos(lm, sorted_data[i]);  // identical to the lookup path
            int64_t err = static_cast<int64_t>(i) - pred;
            if (err < mn) mn = err;
            if (err > mx) mx = err;
            total_abs_err += static_cast<long double>(err < 0 ? -err : err);
        }
        lm.min_err = static_cast<int32_t>(mn);
        lm.max_err = static_cast<int32_t>(mx);
        // Guard against a segment so large the error exceeds int32 range.
        assert(static_cast<int64_t>(lm.min_err) == mn && static_cast<int64_t>(lm.max_err) == mx);

        size_t window = static_cast<size_t>(mx - mn + 1);
        total_window += static_cast<long double>(window) * static_cast<long double>(hi - lo);
        if (window > max_window_) max_window_ = window;
    }

    avg_abs_error_ = static_cast<double>(total_abs_err / static_cast<long double>(n));
    avg_window_ = static_cast<double>(total_window / static_cast<long double>(n));
}

}  // namespace lidx
