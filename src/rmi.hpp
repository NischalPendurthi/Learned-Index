// Two-stage RMI from Kraska et al. (SIGMOD 2018).
// Stage 1 picks one of K models, that model predicts the position, and we
// search only inside the error range the model saw during build.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <vector>

#include "common.hpp"
#include "search.hpp"

namespace lidx {

    struct LinearModel {
        double slope;
        double intercept;
        i32 min_err;  // worst (actual - predicted), <= 0
        i32 max_err;  // worst (actual - predicted), >= 0
    };

    static_assert(sizeof(LinearModel) == 24, "LinearModel must be densely packed");

    // Stage 1 is a line on key or log(key), or a monotone piecewise-linear fit
    // through equal-depth points of the CDF. build() keeps whichever balances
    // the buckets best.
    enum class Stage1Kind { Linear, LogLinear, Piecewise };

    class RMI {
    public:
        // allow_piecewise = false limits stage 1 to the two straight-line fits
        void build(const u64* sorted_data, sz n, sz num_models, bool allow_piecewise = true);

        // returns -1 if key is not present
        inline i64 lookup(u64 key) const;

        // both stages, no search (used to time the model alone)
        inline sz predict_only(u64 key) const;

        // stage 1 alone, and the rest of a lookup once the model is known
        sz route(u64 key) const { return model_index(key); }
        inline i64 lookup_in_model(sz m, u64 key) const;

        // width of the positions model m has to search
        sz model_window(sz m) const { return SC(sz, SC(i64, models_[m].max_err) - models_[m].min_err + 1); }

        // model bytes only, the data array is not counted
        sz index_size_bytes() const {
            return models_.size() * sizeof(LinearModel) + sizeof(double) * 2 + sizeof(Stage1Kind)
                 + knot_x_.size() * sizeof(double) * 3;  // x, y, slope per knot
        }

        sz num_models() const { return models_.size(); }
        Stage1Kind stage1_kind() const { return s1_kind_; }
        const char* stage1_name() const {
            switch (s1_kind_) {
                case Stage1Kind::Linear: return "linear";
                case Stage1Kind::LogLinear: return "loglinear";
                case Stage1Kind::Piecewise: return "piecewise";
            }
            return "?";
        }

        double avg_abs_error() const { return avg_abs_error_; }
        double avg_search_window() const { return avg_window_; }
        sz max_search_window() const { return max_window_; }

    private:
        // build() uses these same functions to compute the error bounds, so the
        // bounds match exactly what lookup() computes.
        inline sz model_index(u64 key) const;
        inline i64 predict_pos(const LinearModel& m, u64 key) const;
        inline double piecewise_eval(double x) const;
        bool fit_piecewise(const u64* sorted_data, sz n, sz num_models);

        std::vector<LinearModel> models_;
        double s1_slope_ = 0.0;
        double s1_intercept_ = 0.0;
        Stage1Kind s1_kind_ = Stage1Kind::Linear;

        // Piecewise stage 1. knot_x_ is strictly increasing, and segment i runs
        // from (knot_x_[i], knot_y_[i]) with slope knot_slope_[i].
        std::vector<double> knot_x_;
        std::vector<double> knot_y_;
        std::vector<double> knot_slope_;

        const u64* data_ = nullptr;
        sz N_ = 0;

        double avg_abs_error_ = 0.0;
        double avg_window_ = 0.0;
        sz max_window_ = 0;
    };

    // Clamping to the segment's own y range keeps the function monotone across
    // knots, where rounding could otherwise step backwards. build() relies on
    // that so each stage-2 model owns a contiguous slice.
    inline double RMI::piecewise_eval(double x) const {
        const double* base = knot_x_.data();
        sz n = knot_x_.size() - 1;  // segments; the last knot only closes them
        while (n > 1) {
            sz half = n >> 1;
            // multiply instead of ?: because gcc turns a double compare into a branch
            base += half * SC(sz, base[half] <= x);
            n -= half;
        }
        sz i = SC(sz, base - knot_x_.data());
        double y = knot_y_[i] + (x - knot_x_[i]) * knot_slope_[i];
        return clampv(y, knot_y_[i], knot_y_[i + 1]);
    }

    inline sz RMI::model_index(u64 key) const {
        double x = SC(double, key);
        double m;
        if (s1_kind_ == Stage1Kind::Piecewise) {
            m = piecewise_eval(x);
        } else {
            if (s1_kind_ == Stage1Kind::LogLinear) x = std::log(x + 1.0);
            m = s1_slope_ * x + s1_intercept_;
        }
        double hi = SC(double, models_.size() - 1);
        m = clampv(m, 0.0, hi);
        return SC(sz, m);
    }

    inline i64 RMI::predict_pos(const LinearModel& m, u64 key) const {
        double p = m.slope * SC(double, key) + m.intercept;
        double hi = SC(double, N_ - 1);
        p = clampv(p, 0.0, hi);
        return SC(i64, p);
    }

    inline sz RMI::predict_only(u64 key) const {
        return SC(sz, predict_pos(models_[model_index(key)], key));
    }

    inline i64 RMI::lookup(u64 key) const {
        return lookup_in_model(model_index(key), key);
    }

    inline i64 RMI::lookup_in_model(sz model, u64 key) const {
        const LinearModel& m = models_[model];
        i64 pred = predict_pos(m, key);

        i64 lo = pred + m.min_err;
        i64 hi = pred + m.max_err;
        if (lo < 0) lo = 0;
        if (hi > SC(i64, N_) - 1) hi = SC(i64, N_) - 1;
        if (lo > hi) return -1;

        sz off = lower_bound_branchless(data_ + lo, SC(sz, hi - lo + 1), key);
        i64 idx = lo + SC(i64, off);
        if (idx <= hi && data_[idx] == key) return idx;
        return -1;
    }

}  // namespace lidx
