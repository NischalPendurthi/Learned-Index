// Read-only B-Tree over a sorted array.
// The sorted array itself is the leaf level. Each level above is one
// cache-aligned array holding the first key of every node below it. Pages are
// always full, so the child of slot j in node i is node i*P + j and we don't
// need to store child pointers.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <vector>

#include "common.hpp"
#include "search.hpp"

namespace lidx {

class BTreeIndex {
public:
    void build(const u64* sorted_data, sz n, int page_size);

    // returns -1 if key is not present
    inline i64 lookup(u64 key) const;

    // walks root to leaf without searching the leaf page (to time traversal alone)
    inline sz traverse_only(u64 key) const;

    // upper levels only, the data array is not counted
    sz index_size_bytes() const { return size_bytes_; }

    int height() const { return SC(int, levels_.size()) + 1; }
    int page_size() const { return page_size_; }
    sz num_leaf_pages() const { return levels_.empty() ? 0 : levels_[0].n; }

private:
    // levels_[0] is just above the leaves, levels_.back() is the root
    std::vector<AlignedKeys> levels_;
    const u64* data_ = nullptr;
    sz N_ = 0;
    int page_size_ = 128;
    sz size_bytes_ = 0;
};

inline sz BTreeIndex::traverse_only(u64 key) const {
    const sz P = SC(sz, page_size_);
    sz node = 0;  // root is node 0 of the top level
    for (sz l = levels_.size(); l-- > 0;) {
        const AlignedKeys& lv = levels_[l];
        sz begin = node * P;
        sz end = std::min(lv.n, begin + P);
        // last key <= search key (slot 0 if the key is smaller than all of them)
        sz cnt = upper_bound_branchless(lv.keys + begin, end - begin, key);
        node = begin + (cnt ? cnt - 1 : 0);
    }
    return node;  // leaf page index
}

inline i64 BTreeIndex::lookup(u64 key) const {
    const sz P = SC(sz, page_size_);
    sz page = traverse_only(key);
    sz lo = page * P;
    sz hi = std::min(N_, lo + P);
    sz idx = lo + lower_bound_branchless(data_ + lo, hi - lo, key);
    if (idx < hi && data_[idx] == key) return SC(i64, idx);
    return -1;
}

}  // namespace lidx
