#include "hybrid.hpp"

namespace lidx {

void HybridRMI::build(const u64* sorted_data, sz n, sz num_models, sz threshold, int page_size) {
    rmi_.build(sorted_data, n, num_models);
    tree_of_.assign(num_models, -1);
    trees_.clear();
    tree_base_.clear();

    sz worst_window = 0;
    sz lo = 0;
    while (lo < n) {
        // stage 1 is monotone, so each model owns one contiguous run
        sz m = rmi_.route(sorted_data[lo]);
        sz hi = lo + 1;
        while (hi < n && rmi_.route(sorted_data[hi]) == m) ++hi;

        sz window = rmi_.model_window(m);
        if (window > worst_window) {
            worst_window = window;
            worst_lo_ = lo;
            worst_hi_ = hi;
        }
        if (window > threshold) {
            tree_of_[m] = SC(i32, trees_.size());
            trees_.emplace_back();
            trees_.back().build(sorted_data + lo, hi - lo, page_size);
            tree_base_.push_back(lo);
        }
        lo = hi;
    }
}

}  // namespace lidx
