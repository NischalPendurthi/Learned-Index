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
    if (cnt == 0) return {0.0, SC(double, lo)};
    if (cnt == 1) return {0.0, SC(double, lo)};

    ld mx = 0.0L, my = 0.0L;
    rep(i, lo, hi) {
        mx += xf(data[i]);
        my += SC(ld, i);
    }
    mx /= cnt;
    my /= cnt;

    ld sxx = 0.0L, sxy = 0.0L;
    rep(i, lo, hi) {
        ld dx = SC(ld, xf(data[i])) - mx;
        ld dy = SC(ld, i) - my;
        sxx += dx * dx;
        sxy += dx * dy;
    }

    if (sxx <= 0.0L) return {0.0, SC(double, lo)};
    ld slope = sxy / sxx;
    ld intercept = my - slope * mx;
    double s = SC(double, slope);
    double b = SC(double, intercept);
    if (!std::isfinite(s) || !std::isfinite(b) || s <= 0.0) return {0.0, SC(double, lo)};
    return {s, b};
}

struct XLinear {
    ld operator()(u64 k) const { return SC(ld, SC(double, k)); }
};
struct XLog {
    ld operator()(u64 k) const {
        return SC(ld, std::log(SC(double, k) + 1.0));
    }
};

}  // namespace

// Equal-depth knots: roughly the same number of keys falls between each pair,
// so the fit follows the CDF wherever the keys are dense. Returns false if the
// keys don't give at least one segment.
bool RMI::fit_piecewise(const u64* sorted_data, sz n, sz num_models) {
    knot_x_.clear();
    knot_y_.clear();
    knot_slope_.clear();

    // Each doubling of the segments adds a dependent load to every lookup, and
    // more than a few hundred stop improving the model balance.
    const sz segments = clampv<sz>(num_models / 4, 16, 256);
    knot_x_.reserve(segments + 1);
    knot_y_.reserve(segments + 1);

    rep(s, 0, segments + 1) {
        sz pos = (s == segments) ? n - 1 : SC(sz, SC(ld, s) * n / segments);
        double x = SC(double, sorted_data[pos]);
        // Large keys can round to the same double. Keep only strictly
        // increasing knots so no segment has zero width.
        if (!knot_x_.empty() && x <= knot_x_.back()) continue;
        knot_x_.push_back(x);
        knot_y_.push_back(SC(double, SC(ld, pos) * num_models / n));
    }
    if (knot_x_.size() < 2) {
        knot_x_.clear();
        knot_y_.clear();
        return false;
    }

    knot_y_.back() = SC(double, num_models);
    knot_slope_.resize(knot_x_.size());
    rep(i, 0, knot_x_.size() - 1) {
        knot_slope_[i] = (knot_y_[i + 1] - knot_y_[i]) / (knot_x_[i + 1] - knot_x_[i]);
    }
    knot_slope_.back() = 0.0;
    return true;
}

void RMI::build(const u64* sorted_data, sz n, sz num_models, bool allow_piecewise) {
    data_ = sorted_data;
    N_ = n;
    if (num_models < 1) num_models = 1;
    models_.assign(num_models, LinearModel{0.0, 0.0, 0, 0});

    const double scale = SC(double, num_models) / SC(double, n);

    // Stage 1: fit on key and on log(key), scaled from [0, N) down to [0, K).
    struct Cand {
        Stage1Kind kind;
        double slope, intercept;
        ld cost;
    };
    Cand cands[3];

    {
        Fit f = fit_range(sorted_data, 0, n, XLinear{});
        cands[0] = {Stage1Kind::Linear, f.slope * scale, f.intercept * scale, 0.0L};
        Fit g = fit_range(sorted_data, 0, n, XLog{});
        cands[1] = {Stage1Kind::LogLinear, g.slope * scale, g.intercept * scale, 0.0L};
        // slope 1 is only a "this candidate is usable" marker, the knots hold the fit
        bool ok = allow_piecewise && fit_piecewise(sorted_data, n, num_models);
        cands[2] = {Stage1Kind::Piecewise, ok ? 1.0 : 0.0, 0.0, 0.0L};
    }

    // Keep the candidate with the smallest sum of (keys per model)^2. Bigger
    // buckets mean bigger search windows, and they also get more queries.
    // Piecewise costs a knot search on every lookup, so it has to be clearly
    // better (5%) to replace a plain line.
    std::vector<u32> counts(num_models);
    int best = -1;
    for (int c = 0; c < 3; ++c) {
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
            ld c2 = SC(ld, counts[m]);
            cost += c2 * c2;
        }
        cands[c].cost = cost;
        ld to_beat = (best >= 0) ? cands[best].cost : 0.0L;
        if (cands[c].kind == Stage1Kind::Piecewise) to_beat *= 0.95L;
        if (best < 0 || cost < to_beat) best = c;
    }

    if (best < 0 || cands[best].kind != Stage1Kind::Piecewise) {
        // not used, so don't carry it or count it in the index size
        knot_x_.clear();
        knot_y_.clear();
        knot_slope_.clear();
    }

    if (best < 0) {
        // both fits failed: just spread min..max key evenly over the models
        s1_kind_ = Stage1Kind::Linear;
        double k0 = SC(double, sorted_data[0]);
        double k1 = SC(double, sorted_data[n - 1]);
        double span = (k1 > k0) ? (k1 - k0) : 1.0;
        s1_slope_ = SC(double, num_models) / span;
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
            i64 err = SC(i64, i) - pred;
            if (err < mn) mn = err;
            if (err > mx) mx = err;
            total_abs_err += SC(ld, err < 0 ? -err : err);
        }
        lm.min_err = SC(i32, mn);
        lm.max_err = SC(i32, mx);
        // errors are stored as int32
        assert(SC(i64, lm.min_err) == mn && SC(i64, lm.max_err) == mx);

        sz window = SC(sz, mx - mn + 1);
        total_window += SC(ld, window) * SC(ld, hi - lo);
        if (window > max_window_) max_window_ = window;
    }

    avg_abs_error_ = SC(double, total_abs_err / SC(ld, n));
    avg_window_ = SC(double, total_window / SC(ld, n));
}

}  // namespace lidx
