// Cache-optimized, read-only B-Tree over a sorted array.
//
// Layout: the leaf level *is* the sorted array (no copies).  Above it sits one
// densely packed, 64-byte-aligned key array per level; level L holds one fence
// key per node of level L-1, namely that node's first key.  With a fixed fanout
// and a 100% fill factor the child of slot j in node i is node (i*P + j) of the
// level below -- so the fence key's own index in the level array *is* the child
// index, and no child pointers need to be stored or dereferenced.
//
// That makes the baseline both smaller and faster than a pointer-carrying node
// layout, i.e. it is the conservative choice on both axes we compare against.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <vector>

#include "search.hpp"

namespace lidx {

class BTreeIndex {
public:
    void build(const uint64_t* sorted_data, size_t n, int page_size);

    // Position of key, or -1 if absent.
    inline int64_t lookup(uint64_t key) const;

    // Root-to-leaf descent only, skipping the search inside the leaf page.
    // The counterpart of RMI::predict_only.
    inline size_t traverse_only(uint64_t key) const;

    // Bytes of internal fence keys.  Excludes the sorted data array (the leaf
    // level), which the RMI shares.
    size_t index_size_bytes() const { return size_bytes_; }

    // Internal levels plus the leaf level.
    int height() const { return static_cast<int>(levels_.size()) + 1; }
    int page_size() const { return page_size_; }
    size_t num_leaf_pages() const { return levels_.empty() ? 0 : levels_[0].n; }

private:
    // levels_[0] sits directly above the leaves; levels_.back() is the root.
    std::vector<AlignedKeys> levels_;
    const uint64_t* data_ = nullptr;
    size_t N_ = 0;
    int page_size_ = 128;
    size_t size_bytes_ = 0;
};

inline size_t BTreeIndex::traverse_only(uint64_t key) const {
    const size_t P = static_cast<size_t>(page_size_);
    size_t node = 0;  // root is node 0 of the top level
    for (size_t l = levels_.size(); l-- > 0;) {
        const AlignedKeys& lv = levels_[l];
        size_t begin = node * P;
        size_t end = std::min(lv.n, begin + P);
        // Rightmost fence key <= search key; slot 0 if the key precedes them all.
        size_t cnt = upper_bound_branchless(lv.keys + begin, end - begin, key);
        node = begin + (cnt ? cnt - 1 : 0);
    }
    return node;  // leaf page index
}

inline int64_t BTreeIndex::lookup(uint64_t key) const {
    const size_t P = static_cast<size_t>(page_size_);
    size_t page = traverse_only(key);
    size_t lo = page * P;
    size_t hi = std::min(N_, lo + P);
    size_t idx = lo + lower_bound_branchless(data_ + lo, hi - lo, key);
    if (idx < hi && data_[idx] == key) return static_cast<int64_t>(idx);
    return -1;
}

}  // namespace lidx
