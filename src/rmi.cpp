#include "rmi.hpp"

#include <cassert>
#include <limits>

namespace lidx {
namespace {

struct Fit {
    double slope;
    double intercept;
};

// Least squares over data[lo, hi) with y = position. Keys go up to 2^63, so we
// subtract the means first and sum in ld to keep precision.
template <typename XForm>
Fit fit_range(const u64* data, sz lo, sz hi, XForm xf) {
    sz cnt = hi - lo;
    if (cnt == 0) return {0.0, static_cast<double>(lo)};
    if (cnt == 1) return {0.0, static_cast<double>(lo)};

    ld mx = 0.0L, my = 0.0L;
    rep(i, lo, hi) {
        mx += xf(data[i]);
        my += static_cast<ld>(i);
    }
    mx /= cnt;
    my /= cnt;

    ld sxx = 0.0L, sxy = 0.0L;
    rep(i, lo, hi) {
        ld dx = static_cast<ld>(xf(data[i])) - mx;
        ld dy = static_cast<ld>(i) - my;
        sxx += dx * dx;
        sxy += dx * dy;
    }

    if (sxx <= 0.0L) return {0.0, static_cast<double>(lo)};
    ld slope = sxy / sxx;
    ld intercept = my - slope * mx;
    double s = static_cast<double>(slope);
    double b = static_cast<double>(intercept);
    if (!std::isfinite(s) || !std::isfinite(b) || s <= 0.0) return {0.0, static_cast<double>(lo)};
    return {s, b};
}

struct XLinear {
    ld operator()(u64 k) const { return static_cast<ld>(static_cast<double>(k)); }
};
struct XLog {
    ld operator()(u64 k) const {
        return static_cast<ld>(std::log(static_cast<double>(k) + 1.0));
    }
};

}  // namespace

void RMI::build(const u64* sorted_data, sz n, sz num_models) {
    data_ = sorted_data;
    N_ = n;
    if (num_models < 1) num_models = 1;
    models_.assign(num_models, LinearModel{0.0, 0.0, 0, 0});

    const double scale = static_cast<double>(num_models) / static_cast<double>(n);

    // Stage 1: fit on key and on log(key), scaled from [0, N) down to [0, K).
    struct Cand {
        Stage1Kind kind;
        double slope, intercept;
        ld cost;
    };
    Cand cands[2];

    {
        Fit f = fit_range(sorted_data, 0, n, XLinear{});
        cands[0] = {Stage1Kind::Linear, f.slope * scale, f.intercept * scale, 0.0L};
        Fit g = fit_range(sorted_data, 0, n, XLog{});
        cands[1] = {Stage1Kind::LogLinear, g.slope * scale, g.intercept * scale, 0.0L};
    }

    // Keep the candidate with the smaller sum of (keys per model)^2. Bigger
    // buckets mean bigger search windows, and they also get more queries.
    std::vector<u32> counts(num_models);
    int best = -1;
    for (int c = 0; c < 2; ++c) {
        if (!(cands[c].slope > 0.0) || !std::isfinite(cands[c].intercept)) {
            cands[c].cost = std::numeric_limits<ld>::infinity();
            continue;
        }
        std::fill(ALL(counts), 0u);
        s1_kind_ = cands[c].kind;
        s1_slope_ = cands[c].slope;
        s1_intercept_ = cands[c].intercept;
        rep(i, 0, n) counts[model_index(sorted_data[i])]++;
        ld cost = 0.0L;
        rep(m, 0, num_models) {
            ld c2 = static_cast<ld>(counts[m]);
            cost += c2 * c2;
        }
        cands[c].cost = cost;
        if (best < 0 || cost < cands[best].cost) best = c;
    }

    if (best < 0) {
        // both fits failed: just spread min..max key evenly over the models
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

    // Stage 2. Stage 1 never decreases as the key grows, so each model gets a
    // contiguous slice of the array and we only need to store where it starts.
    std::vector<sz> start(num_models + 1, n);
    {
        sz prev = 0;
        start[0] = 0;
        rep(i, 0, n) {
            sz m = model_index(sorted_data[i]);
            assert(m >= prev);
            while (prev < m) start[++prev] = i;
        }
        while (prev < num_models) start[++prev] = n;
    }

    ld total_abs_err = 0.0L;
    ld total_window = 0.0L;
    max_window_ = 0;

    rep(m, 0, num_models) {
        sz lo = start[m], hi = start[m + 1];
        Fit f = fit_range(sorted_data, lo, hi, XLinear{});
        LinearModel& lm = models_[m];
        lm.slope = f.slope;
        lm.intercept = f.intercept;
        lm.min_err = 0;
        lm.max_err = 0;

        if (hi == lo) continue;  // empty, only absent keys can land here

        i64 mn = std::numeric_limits<i64>::max();
        i64 mx = std::numeric_limits<i64>::min();
        rep(i, lo, hi) {
            i64 pred = predict_pos(lm, sorted_data[i]);  // same code as lookup()
            i64 err = static_cast<i64>(i) - pred;
            if (err < mn) mn = err;
            if (err > mx) mx = err;
            total_abs_err += static_cast<ld>(err < 0 ? -err : err);
        }
        lm.min_err = static_cast<i32>(mn);
        lm.max_err = static_cast<i32>(mx);
        // errors are stored as int32
        assert(static_cast<i64>(lm.min_err) == mn && static_cast<i64>(lm.max_err) == mx);

        sz window = static_cast<sz>(mx - mn + 1);
        total_window += static_cast<ld>(window) * static_cast<ld>(hi - lo);
        if (window > max_window_) max_window_ = window;
    }

    avg_abs_error_ = static_cast<double>(total_abs_err / static_cast<ld>(n));
    avg_window_ = static_cast<double>(total_window / static_cast<ld>(n));
}

}  // namespace lidx
