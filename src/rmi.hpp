// Recursive Model Index (2-stage), following Kraska et al.,
// "The Case for Learned Index Structures", Section 3.2 / 3.5.
//
//   stage 1: one model  key -> index of a stage-2 model in [0, K)
//   stage 2: K linear models  key -> predicted position in [0, N)
//
// Each stage-2 model stores the worst over- and under-prediction observed over
// the keys assigned to it (min_err / max_err).  A lookup therefore only has to
// search data[pred + min_err, pred + max_err], which is guaranteed to contain
// the key if it is present.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <vector>

#include "search.hpp"

namespace lidx {

// 24 bytes: this is the unit the reported RMI index size is counted in.
struct LinearModel {
    double slope;
    double intercept;
    int32_t min_err;  // most negative (actual - predicted), <= 0
    int32_t max_err;  // most positive (actual - predicted), >= 0
};
static_assert(sizeof(LinearModel) == 24, "LinearModel must be densely packed");

// Stage-1 feature transform.  The paper tunes the first-stage model by grid
// search; we do the same over two cheap monotone candidates and keep whichever
// balances the stage-2 models better (see rmi.cpp).
enum class Stage1Kind { Linear, LogLinear };

class RMI {
public:
    void build(const uint64_t* sorted_data, size_t n, size_t num_models);

    // Position of key, or -1 if absent.
    inline int64_t lookup(uint64_t key) const;

    // Runs both stages but skips the search: this is the model-only cost.
    inline size_t predict_only(uint64_t key) const;

    // Bytes of model parameters.  Excludes the sorted data array, which the
    // B-Tree baseline shares.
    size_t index_size_bytes() const {
        return models_.size() * sizeof(LinearModel) + sizeof(double) * 2 + sizeof(Stage1Kind);
    }

    size_t num_models() const { return models_.size(); }
    Stage1Kind stage1_kind() const { return s1_kind_; }
    const char* stage1_name() const { return s1_kind_ == Stage1Kind::Linear ? "linear" : "loglinear"; }

    // Build-time statistics (not used on the lookup path).
    double avg_abs_error() const { return avg_abs_error_; }
    double avg_search_window() const { return avg_window_; }
    size_t max_search_window() const { return max_window_; }

private:
    // The exact arithmetic used at build time to derive min_err/max_err, so
    // that the stored bounds are valid for the identical computation at lookup.
    inline size_t model_index(uint64_t key) const;
    inline int64_t predict_pos(const LinearModel& m, uint64_t key) const;

    std::vector<LinearModel> models_;
    double s1_slope_ = 0.0;
    double s1_intercept_ = 0.0;
    Stage1Kind s1_kind_ = Stage1Kind::Linear;

    const uint64_t* data_ = nullptr;
    size_t N_ = 0;

    double avg_abs_error_ = 0.0;
    double avg_window_ = 0.0;
    size_t max_window_ = 0;
};

inline size_t RMI::model_index(uint64_t key) const {
    double x = static_cast<double>(key);
    if (s1_kind_ == Stage1Kind::LogLinear) x = std::log(x + 1.0);
    double m = s1_slope_ * x + s1_intercept_;
    double hi = static_cast<double>(models_.size() - 1);
    m = m < 0.0 ? 0.0 : (m > hi ? hi : m);
    return static_cast<size_t>(m);
}

inline int64_t RMI::predict_pos(const LinearModel& m, uint64_t key) const {
    double p = m.slope * static_cast<double>(key) + m.intercept;
    double hi = static_cast<double>(N_ - 1);
    p = p < 0.0 ? 0.0 : (p > hi ? hi : p);
    return static_cast<int64_t>(p);
}

inline size_t RMI::predict_only(uint64_t key) const {
    return static_cast<size_t>(predict_pos(models_[model_index(key)], key));
}

inline int64_t RMI::lookup(uint64_t key) const {
    const LinearModel& m = models_[model_index(key)];
    int64_t pred = predict_pos(m, key);

    int64_t lo = pred + m.min_err;
    int64_t hi = pred + m.max_err;
    if (lo < 0) lo = 0;
    if (hi > static_cast<int64_t>(N_) - 1) hi = static_cast<int64_t>(N_) - 1;
    if (lo > hi) return -1;

    size_t off = lower_bound_branchless(data_ + lo, static_cast<size_t>(hi - lo + 1), key);
    int64_t idx = lo + static_cast<int64_t>(off);
    if (idx <= hi && data_[idx] == key) return idx;
    return -1;
}

}  // namespace lidx
