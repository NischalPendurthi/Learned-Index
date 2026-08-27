#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "common.h"
#include "linear_model.h"
#include "search.h"

namespace rmi {

struct LookupResult {
    bool found = false;
    size_t position = 0;
};

enum class SearchMode { Binary, Exponential };

// Everything reported by main.cpp after training.
struct TrainStats {
    size_t n = 0;
    size_t num_models = 0;
    size_t empty_models = 0;
    size_t single_key_models = 0;

    // Keys per stage-2 model.
    size_t keys_min = 0;
    size_t keys_max = 0;
    double keys_mean = 0.0;
    double keys_median = 0.0;
    double max_bucket_share = 0.0;  // largest bucket / N

    // Per-model error width (max_error - min_error), over non-empty models.
    // Width 0 means the model is exact and the search inspects a single slot.
    int64_t err_min = 0;
    int64_t err_max = 0;
    double err_mean = 0.0;
    double err_p50 = 0.0;
    double err_p99 = 0.0;

    size_t index_size_bytes = 0;
    double train_seconds = 0.0;
};

class RMI {
public:
    // Trains the root model on all n keys, routes each key to one of
    // num_stage2 buckets, and fits a linear model per bucket.
    void train(const Key* keys, size_t n, size_t num_stage2);

    inline LookupResult lookup(Key key) const;

    // Both stages, no search.  Useful for splitting model time from search time.
    inline int64_t predict_position(Key key) const;

    inline size_t bucket_for(Key key) const;

    void set_search_mode(SearchMode m) { mode_ = m; }
    SearchMode search_mode() const { return mode_; }

    const TrainStats& stats() const { return stats_; }
    const LinearModel& root() const { return root_; }
    const std::vector<LinearModel>& stage2() const { return stage2_; }

    // All model parameters: the K stage-2 models plus the root.
    size_t index_size_bytes() const { return (stage2_.size() + 1) * sizeof(LinearModel); }
    size_t num_models() const { return stage2_.size(); }

private:
    LinearModel root_;
    std::vector<LinearModel> stage2_;
    const Key* data_ = nullptr;
    size_t n_ = 0;
    size_t k_ = 0;
    SearchMode mode_ = SearchMode::Binary;
    TrainStats stats_;
};

// stage2_index = clamp(floor(root.predict(key) * K / N), 0, K-1)
inline size_t RMI::bucket_for(Key key) const {
    int64_t p = root_.predict(key);
    double b = std::floor(static_cast<double>(p) * static_cast<double>(k_) /
                          static_cast<double>(n_));
    if (b <= 0.0) return 0;
    if (b >= static_cast<double>(k_ - 1)) return k_ - 1;
    return static_cast<size_t>(b);
}

inline int64_t RMI::predict_position(Key key) const {
    return stage2_[bucket_for(key)].predict(key);
}

inline LookupResult RMI::lookup(Key key) const {
    const LinearModel& m = stage2_[bucket_for(key)];
    int64_t lo, hi;
    m.search_bounds(key, static_cast<int64_t>(n_), &lo, &hi);
    if (lo > hi) return {false, 0};

    int64_t idx;
    if (mode_ == SearchMode::Exponential) {
        idx = exponential_search(data_, lo, hi, m.predict(key), key);
    } else {
        idx = binary_search_bounded(data_, lo, hi, key);
    }
    if (idx < 0) return {false, 0};
    return {true, static_cast<size_t>(idx)};
}

}  // namespace rmi
