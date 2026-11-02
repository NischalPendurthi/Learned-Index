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

    // Stage 1 fits on either key or log(key); build() keeps whichever balances
    // the buckets better.
    enum class Stage1Kind { Linear, LogLinear };

    class RMI {
    public:
        void build(const u64* sorted_data, sz n, sz num_models);

        // returns -1 if key is not present
        inline i64 lookup(u64 key) const;

        // both stages, no search (used to time the model alone)
        inline sz predict_only(u64 key) const;

        // model bytes only, the data array is not counted
        sz index_size_bytes() const {
            return models_.size() * sizeof(LinearModel) + sizeof(double) * 2 + sizeof(Stage1Kind);
        }

        sz num_models() const { return models_.size(); }
        Stage1Kind stage1_kind() const { return s1_kind_; }
        const char* stage1_name() const { return s1_kind_ == Stage1Kind::Linear ? "linear" : "loglinear"; }

        double avg_abs_error() const { return avg_abs_error_; }
        double avg_search_window() const { return avg_window_; }
        sz max_search_window() const { return max_window_; }

    private:
        // build() uses these same functions to compute the error bounds, so the
        // bounds match exactly what lookup() computes.
        inline sz model_index(u64 key) const;
        inline i64 predict_pos(const LinearModel& m, u64 key) const;

        std::vector<LinearModel> models_;
        double s1_slope_ = 0.0;
        double s1_intercept_ = 0.0;
        Stage1Kind s1_kind_ = Stage1Kind::Linear;

        const u64* data_ = nullptr;
        sz N_ = 0;

        double avg_abs_error_ = 0.0;
        double avg_window_ = 0.0;
        sz max_window_ = 0;
    };

    inline sz RMI::model_index(u64 key) const {
        double x = static_cast<double>(key);
        if (s1_kind_ == Stage1Kind::LogLinear) x = std::log(x + 1.0);
        double m = s1_slope_ * x + s1_intercept_;
        double hi = static_cast<double>(models_.size() - 1);
        m = clampv(m, 0.0, hi);
        return static_cast<sz>(m);
    }

    inline i64 RMI::predict_pos(const LinearModel& m, u64 key) const {
        double p = m.slope * static_cast<double>(key) + m.intercept;
        double hi = static_cast<double>(N_ - 1);
        p = clampv(p, 0.0, hi);
        return static_cast<i64>(p);
    }

    inline sz RMI::predict_only(u64 key) const {
        return static_cast<sz>(predict_pos(models_[model_index(key)], key));
    }

    inline i64 RMI::lookup(u64 key) const {
        const LinearModel& m = models_[model_index(key)];
        i64 pred = predict_pos(m, key);

        i64 lo = pred + m.min_err;
        i64 hi = pred + m.max_err;
        if (lo < 0) lo = 0;
        if (hi > static_cast<i64>(N_) - 1) hi = static_cast<i64>(N_) - 1;
        if (lo > hi) return -1;

        sz off = lower_bound_branchless(data_ + lo, static_cast<sz>(hi - lo + 1), key);
        i64 idx = lo + static_cast<i64>(off);
        if (idx <= hi && data_[idx] == key) return idx;
        return -1;
    }

}  // namespace lidx
