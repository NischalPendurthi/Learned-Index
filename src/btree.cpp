#include "btree.hpp"

namespace lidx {

void BTreeIndex::build(const u64* sorted_data, sz n, int page_size) {
    data_ = sorted_data;
    N_ = n;
    page_size_ = page_size;
    levels_.clear();
    size_bytes_ = 0;

    const sz P = static_cast<sz>(page_size);

    // first key of every leaf page
    sz count = (n + P - 1) / P;
    {
        AlignedKeys lv;
        lv.alloc(count);
        rep(i, 0, count) lv.keys[i] = sorted_data[i * P];
        size_bytes_ += count * sizeof(u64);
        levels_.push_back(std::move(lv));
    }

    // add levels until the top one fits in one node
    while (count > P) {
        sz next = (count + P - 1) / P;
        AlignedKeys lv;
        lv.alloc(next);
        const AlignedKeys& below = levels_.back();
        rep(i, 0, next) lv.keys[i] = below.keys[i * P];
        size_bytes_ += next * sizeof(u64);
        levels_.push_back(std::move(lv));
        count = next;
    }
}

}  // namespace lidx
