# 2-stage Recursive Model Index

A learned index over a sorted array of 64-bit keys, following Kraska et al.,
*The Case for Learned Index Structures* (SIGMOD 2018), §3.2 and §3.5. Pure C++17,
closed-form linear regression only — no ML framework, no neural networks, no
gradient descent.

## Build & run

```
make                 # -> rmi_main, rmi_bench
./rmi_main --help
```

or with CMake:

```
cmake -B build && cmake --build build
```

Everything is configurable; nothing about `N` or `K` is hardcoded.

```
./rmi_main --dataset all --n 1000000 --k 10000            # debug scale
./rmi_main --dataset all --n 10000000,100000000 --k 1000,10000,100000
./rmi_main --dataset lognormal --n 10000000 --k 10000 --search exponential
```

| flag | default | meaning |
|---|---|---|
| `--dataset LIST` | `all` | `dense`, `lognormal`, `uniform_sparse`, or `all` |
| `--n LIST` | `1000000` | dataset sizes, comma-separated |
| `--k LIST` | `10000` | stage-2 model counts, comma-separated |
| `--seed S` | `42` | RNG seed |
| `--search MODE` | `binary` | `binary` or `exponential` |
| `--verify-limit M` | `0` (all) | check `M` strided keys instead of all `N` |
| `--csv PATH` | `rmi_results.csv` | results for the benchmark harness |
| `--json PATH` | `rmi_results.json` | same data as JSON |

`rmi_bench` times lookups and splits model time from search time:

```
./rmi_bench --dataset all --n 10000000 --k 1000,10000,100000
```

## Layout

```
rmi/
├── CMakeLists.txt / Makefile
├── include/common.h          typedefs (Key = uint64_t), tunable constants
├── src/
│   ├── linear_model.h        LinearModel: slope, intercept, min/max_error, predict()
│   ├── search.h              binary_search_bounded(), exponential_search()
│   ├── dataset.h/.cpp        dense / lognormal / uniform_sparse generation
│   ├── rmi.h/.cpp            RMI: train(), lookup(), stats
│   └── main.cpp              train + report + correctness tests
└── bench/benchmark.cpp       lookup timing harness
```

## Two corrections to the spec

**1. The search-bounds formula has its sign flipped.** The spec defines
`error = pred - actual_position`, and names the fields "most negative error
(under-prediction)" / "most positive error (over-prediction)" — both consistent
with each other. But it then gives:

```cpp
int64_t lo = std::max<int64_t>(0, pred + model.min_error);   // wrong
int64_t hi = std::min<int64_t>(N - 1, pred + model.max_error);
```

Inverting `error = pred - actual` gives `actual = pred - error`, so with
`error ∈ [min_error, max_error]` the true position lies in
`[pred - max_error, pred - min_error]`. The spec's version adds where it should
subtract, and swaps which bound is which. Using it produces false negatives
whenever a model is not exact. This implementation uses the corrected form —
see `LinearModel::search_bounds()`.

**2. `uniform_sparse` is not sparse at large N.** The spec draws uniformly from
`[0, 1B)`. At the target sizes that is 1% occupancy at N=10M, 10% at N=100M and
20% at N=200M — closer to the dense dataset than to a sparse one, and the root
model fits it almost perfectly as a result. Kept as specified, but the name
oversells it, and generation uses a presence bitset rather than
draw-sort-dedupe: pulling 200M unique values from a 1e9 range needs ~223M draws
and a 1.8 GB sort, versus a flat 125 MB for the bitset.

## Implementation notes

**Contiguous buckets, not per-bucket key vectors.** `bucket_for()` is monotone
non-decreasing in the key (the root slope is clamped to ≥ 0, and both truncation
and `floor` preserve order), and the input is sorted — so each bucket owns a
contiguous run of positions. `train()` stores run boundaries instead of
materializing a key vector per bucket. This is exactly equivalent, and it is
what makes N ≥ 100M fit in memory at all. `train()` asserts the monotonicity it
relies on (build with `make debug` to enable).

**`-ffp-contract=off` is load-bearing.** GCC defaults to `-ffp-contract=fast`,
which may contract `slope * key + intercept` into an FMA at one call site and
not another. Training and lookup would then compute slightly different
predictions, a stored error bound would be off by one, and a key would go
missing. The flag is in both build files; do not remove it.

**Precision.** Sums are accumulated in `long double` and stored as `double`.
Keys reach 2^63 and N reaches 2e8, where the uncentred normal equations lose too
much precision; the centred form plus a wider accumulator keeps the fit stable.

**Empty and single-key buckets.** A single-key bucket has `var_x = 0`, so the
closed form naturally degenerates to `slope = 0, intercept = position` — exact
for the one key it owns, no special case needed. Empty buckets get a dummy model
pointing at the bucket boundary, widened by one slot on each side so a lookup
inspects the neighbours. Strictly, that widening is unnecessary: every key
present at training time routes to the bucket that owns it, and that bucket is
non-empty by construction, so only absent keys can reach an empty bucket. It is
kept as cheap insurance against the floating-point hazard above.

## Correctness tests

`rmi_main` runs these before reporting, and exits non-zero if 1 or 2 fail.
Tests 3–5 are quality diagnostics — they say the index is *badly shaped*, not
*wrong* — so they print `WARN` and do not fail the build.

| test | check | severity |
|---|---|---|
| 1 | every key found at its exact position (zero false negatives) | hard fail |
| 2 | 1000 absent keys report not-found (zero false positives) | hard fail |
| 3 | no single bucket holds >50% of keys | warn |
| 4 | mean error width shrinks as K grows | warn |
| 5 | on `dense`, the fit is exact (max error width 0) | warn |

## Output

Per configuration: dataset name, N, key range, training time, index size in
bytes (all `LinearModel` structs including the root, 32 B each), keys per model
(min/max/mean/median), error-width distribution (min/max/mean/p50/p99), and the
correctness verdicts. Written to `rmi_results.csv` and `rmi_results.json` for
the benchmark harness to consume.

Error width is `max_error - min_error`, reported over non-empty models; a width
of 0 means the model is exact and the search inspects a single slot. Empty
buckets are excluded from the width statistics and counted separately, so they
cannot drag the median to zero.

## Results

27 configurations — 3 datasets × N ∈ {10M, 100M, 200M} × K ∈ {1K, 10K, 100K} —
on a 13th Gen Intel Core i7-13620H, g++ 13.3. **All 27 pass tests 1–5 with zero
warnings**: every key found at its exact position, no false positives, no bucket
over 50%, error width monotonically decreasing in K, and an exact fit on dense.
Full data in `rmi_results.csv` / `rmi_results.json`; logs in `run_full.log` and
`run_200M.log`.

Error width (`max_error - min_error`) over non-empty models:

| dataset | N | K=1000 | K=10000 | K=100000 |
|---|---|---|---|---|
| dense | any | **0** | **0** | **0** |
| lognormal | 100M | 7425 | 275 | 33 |
| lognormal | 200M | 6168 | 439 | 69 |
| uniform_sparse | 100M | 355 | 112 | 35 |
| uniform_sparse | 200M | 470 | 149 | 47 |

Training is linear in N and essentially independent of K: 0.17 s at 10M, 1.65 s
at 100M, 3.3 s at 200M. Index size depends only on K — 32 KB, 305 KB, 3.05 MB —
so at N=200M the K=1000 index is 32 KB describing 1.6 GB of keys.

Peak RSS at N=200M was 1.60 GB, which fit without swapping.

### What the numbers show

**Dense is exact.** Every model fits its bucket perfectly at every N and K, so
the search inspects a single slot. Test 5 passes trivially.

**Lognormal has a heavy tail, not a balance problem.** A plain linear root does
*not* dump the keys into one bucket — the largest holds 16.9% at worst (100M,
K=1000) and 0.78% at K=100000. The damage shows up in the shape of the error
distribution instead: at 200M/K=100000 the median width is 27 while the max is
1,459,433. Raising K splits the ordinary buckets but barely touches the worst
one (1,466,145 → 1,460,018 → 1,459,433 across the three K values), because that
bucket sits where the lognormal CDF is steepest and no amount of *uniform*
subdivision of a linear root's output range resolves it. Roughly 42% of buckets
end up empty, which is the same skew seen from the other side.

**`uniform_sparse` behaves almost like dense**, for the reason given above — at
these sizes it is 10–20% occupancy of `[0, 1B)`, not a sparse set. Its widths are
tight and scale cleanly with K.

### Lookup timing

`rmi_bench --dataset all --n 10000000 --k 1000,10000,100000` (median of 5 × 1M
lookups), in `rmi_lookup_bench.csv`:

| dataset | K=1000 | K=10000 | K=100000 | model share |
|---|---|---|---|---|
| dense | 43.1 ns | 45.7 ns | 50.5 ns | 18-22% |
| uniform_sparse | 161.6 ns | 128.7 ns | 106.5 ns | 5-10% |
| lognormal | 382.3 ns | 199.8 ns | 161.5 ns | 2-6% |

The two-stage model costs a flat 7.6-11.2 ns regardless of dataset; everything
else is search. Exponential search is within noise of binary search overall —
slightly worse on dense (the window is one slot, so galloping only adds a probe)
and slightly better on lognormal and uniform at K=100000, where it can exploit an
accurate prediction inside a wide window.

These are throughput numbers under memory-level parallelism: consecutive lookups
in the timing loop are independent, so the CPU overlaps their cache misses.

## Relationship to the parent project

`../src` and `../bench` hold a separate, self-contained harness that compares an
RMI against a cache-optimized B-Tree. That one uses a 24-byte model struct, an
auto-selected log-linear stage-1 transform, and uniform keys over `[0, 2^63)`.
This project follows the present spec instead: 32-byte models, a plain linear
root, and uniform keys over `[0, 1B)`. The numbers are therefore **not**
directly comparable between the two.
