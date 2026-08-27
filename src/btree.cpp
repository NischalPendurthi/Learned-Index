#include "btree.hpp"

namespace lidx {

void BTreeIndex::build(const uint64_t* sorted_data, size_t n, int page_size) {
    data_ = sorted_data;
    N_ = n;
    page_size_ = page_size;
    levels_.clear();
    size_bytes_ = 0;

    const size_t P = static_cast<size_t>(page_size);

    // Level above the leaves: one fence key per leaf page, taken from the
    // sorted array itself.
    size_t count = (n + P - 1) / P;
    {
        AlignedKeys lv;
        lv.alloc(count);
        for (size_t i = 0; i < count; ++i) lv.keys[i] = sorted_data[i * P];
        size_bytes_ += count * sizeof(uint64_t);
        levels_.push_back(std::move(lv));
    }

    // Keep adding levels until the top one fits in a single node.
    while (count > P) {
        size_t next = (count + P - 1) / P;
        AlignedKeys lv;
        lv.alloc(next);
        const AlignedKeys& below = levels_.back();
        for (size_t i = 0; i < next; ++i) lv.keys[i] = below.keys[i * P];
        size_bytes_ += next * sizeof(uint64_t);
        levels_.push_back(std::move(lv));
        count = next;
    }
}

}  // namespace lidx
