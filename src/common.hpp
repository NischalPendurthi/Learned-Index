#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lidx {

using u64 = uint64_t;
using i64 = int64_t;
using u32 = uint32_t;
using i32 = int32_t;
using sz = size_t;
using ld = long double;
typedef std::vector<u64> vu64;

template <class T>
inline T clampv(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }

}  // namespace lidx

#define rep(i, a, b) for (lidx::sz i = (a); i < (b); ++i)
#define ALL(v) (v).begin(), (v).end()
