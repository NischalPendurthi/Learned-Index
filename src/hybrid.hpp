// Hybrid index, Algorithm 1 of Kraska et al. A stage-2 model whose search window
// is above a threshold is replaced by a small B-Tree over the keys it owns, so
// the keys a model handles worst get B-Tree behaviour instead.
#pragma once

#include <vector>

#include "btree.hpp"
#include "common.hpp"
#include "rmi.hpp"

namespace lidx {

class HybridRMI {
public:
    void build(const u64* sorted_data, sz n, sz num_models, sz threshold, int page_size);

    // returns -1 if key is not present
    inline i64 lookup(u64 key) const;

    // models, the model -> tree table, and the trees
    sz index_size_bytes() const {
        sz total = rmi_.index_size_bytes() + tree_of_.size() * sizeof(i32);
        for (const BTreeIndex& t : trees_) total += t.index_size_bytes();
        return total;
    }

    sz num_replaced() const { return trees_.size(); }
    const RMI& rmi() const { return rmi_; }

    // key range of the model that had the widest window before any replacement
    sz worst_lo() const { return worst_lo_; }
    sz worst_hi() const { return worst_hi_; }

private:
    RMI rmi_;
    std::vector<i32> tree_of_;  // tree number for each model, -1 if it kept its model
    std::vector<BTreeIndex> trees_;
    std::vector<sz> tree_base_;  // array offset where each tree's slice begins
    sz worst_lo_ = 0;
    sz worst_hi_ = 0;
};

inline i64 HybridRMI::lookup(u64 key) const {
    sz m = rmi_.route(key);
    i32 t = tree_of_[m];
    if (t < 0) return rmi_.lookup_in_model(m, key);
    i64 pos = trees_[SC(sz, t)].lookup(key);
    return pos < 0 ? -1 : pos + SC(i64, tree_base_[SC(sz, t)]);
}

}  // namespace lidx
