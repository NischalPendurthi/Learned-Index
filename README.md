# RMI vs. cache-optimized B-Tree — benchmark harness

Reproduces the core read-path comparison from Kraska et al., *The Case for
Learned Index Structures* (SIGMOD 2018): a 2-stage Recursive Model Index against
a read-only, dense-page B-Tree, over the same sorted array and the same lookup
workload.

## Build & run

```
g++ -std=c++17 -O3 -march=native -o benchmark src/*.cpp bench/benchmark.cpp
./benchmark                       # defaults: N = 10M and 100M, 1M lookups, 5 reps
```

Options:

| flag | default | meaning |
|---|---|---|
| `--sizes A,B` | `10000000,100000000` | dataset sizes to test |
| `--lookups N` | `1000000` | lookups per timed repetition |
| `--reps R` | `5` | repetitions; the **median** is reported |
| `--out FILE` | `benchmark_results.csv` | CSV output path |
| `--verify-sample K` | `0` (check every key) | check `K` strided keys instead of all `N` |

The committed results were produced as two runs (`--sizes 10000000` and
`--sizes 100000000`, into `results_10M.csv` / `results_100M.csv`) so that an OOM
at the largest size could not discard the smaller one; `benchmark_results.csv`
is their concatenation. Speedup and size ratio are computed within each
`(dataset, N)` group, so the merge is exact. A single default `./benchmark` run
covers both sizes in one table.

Results are printed as a table to stdout and written to CSV with columns:

```
dataset,index_type,config,num_keys,index_size_bytes,index_size_mb,
lookup_ns,model_ns,search_ns,model_pct,speedup_vs_btree128,size_ratio_vs_btree128
```

## Layout

| file | contents |
|---|---|
| [src/rmi.hpp](src/rmi.hpp), [src/rmi.cpp](src/rmi.cpp) | 2-stage RMI: `lookup()`, `predict_only()` |
| [src/btree.hpp](src/btree.hpp), [src/btree.cpp](src/btree.cpp) | read-only B-Tree: `lookup()`, `traverse_only()` |
| [src/search.hpp](src/search.hpp) | branch-free binary search + 64-byte-aligned key arrays |
| [src/dataset.hpp](src/dataset.hpp), [src/dataset.cpp](src/dataset.cpp) | dataset and workload generation |
| [bench/benchmark.cpp](bench/benchmark.cpp) | correctness checks, timing, reporting |

Hot paths (`lookup`, `predict_only`, `traverse_only`) live in headers so `-O3`
can inline them into the timing loop; only the build code is in `.cpp` files.

## The RMI

The RMI was **not** present in this repository — it holds only the paper PDF —
so it is implemented here from §3.2 and §3.5 of the paper.

* **Stage 1** — one linear model mapping a key to a stage-2 model index in
  `[0, K)`. The paper tunes this stage by grid search; we do the same over two
  cheap monotone candidates, `x = key` and `x = log(key + 1)`, and keep whichever
  minimizes `sum_m count_m^2` over the resulting key-to-model assignment. That
  sum is a proxy for expected search work: a model's error window grows with the
  number of keys routed to it, and a uniformly sampled query hits a model in
  proportion to its count. The chosen transform is printed in the structure
  report.
* **Stage 2** — `K` linear models, each least-squares fit on the keys routed to
  it. Fits use centred sums with `long double` accumulators; keys reach `2^63`,
  where the naive normal equations lose too much precision.
* **Error bounds** — each stage-2 model stores the worst over- and
  under-prediction over its own keys (`min_err` / `max_err`, §3.5), so a lookup
  searches only `data[pred + min_err, pred + max_err]`. Those bounds are computed
  at build time with the *exact* arithmetic the lookup path uses, which is what
  makes 100% recall guaranteed rather than merely likely.
* **Size accounting** — `K * sizeof(LinearModel)` = `K * 24` bytes
  (slope, intercept, min_err, max_err) plus the stage-1 parameters.

Stage 1 is monotone in the key, so model indices are non-decreasing over the
sorted array and each stage-2 model owns a contiguous run — the build exploits
this to partition in a single pass. Models that end up empty can only be reached
by keys that are absent from the dataset, and correctly return -1.

## The B-Tree baseline

Read-only, 100% fill factor, built bottom-up over the same sorted array.

* The **leaf level is the sorted array itself** — no copies, and the array is
  excluded from both indexes' reported size.
* Above it, one densely packed, 64-byte-aligned key array per level. Level *L*
  holds one fence key per node of level *L−1*, namely that node's first key.
* With a fixed fanout and 100% fill, the child of slot *j* in node *i* is node
  `i*P + j` of the level below — so a fence key's index in the level array *is*
  the child index. **No child pointers are stored or dereferenced.** This makes
  the baseline both smaller and faster than a pointer-carrying node layout, i.e.
  it is the conservative choice on both axes under comparison. (The task sketch
  called for explicit `children` vectors; storing them would have inflated the
  baseline's reported size and slowed its descent, flattering the RMI.)
* Node search uses the same branch-free binary search as the RMI, so no measured
  difference comes from the search primitive.

## Fairness controls

* Build/training time is never inside a timing loop.
* One lookup sample per `(dataset, N)` group, shared by every index in it:
  1M keys drawn from the dataset with replacement, then shuffled, so there is no
  sequential locality.
* 1000 warm-up lookups per measurement, discarded.
* 5 repetitions of 1M lookups; the median is reported.
* Every timed iteration stores to a `volatile` sink, identically for both
  indexes, so nothing is optimized away.
* Both indexes are compiled into the same binary with the same flags.
* Correctness runs *before* any timing and aborts the run on failure: every key
  in the dataset must be found at its exact position, and 1000 keys known to be
  absent must all return -1.
* Fixed seeds throughout, so datasets and workloads are reproducible.

## Datasets

| name | definition |
|---|---|
| `dense` | `0 .. N-1` — a perfectly linear CDF |
| `lognormal` | lognormal(μ=0, σ=2) scaled so the median key is 1e9 (the paper's synthetic heavy-tail dataset, §5.1), deduplicated |
| `uniform_sparse` | uniform over `[0, 2^63)`, deduplicated |

## Results

Measured on a 13th Gen Intel Core i7-13620H (L1d 416 KB, L2 9.5 MB, L3 24 MB),
g++ 13.3, `-O3 -march=native`. Full data in [benchmark_results.csv](benchmark_results.csv);
raw run logs in `run_100M.stderr.log`. Every one of the 42 configurations passed
100% recall over all N keys and returned -1 for all 1000 absent keys.

Headline, versus BTree page=128 on the same dataset and N:

| Dataset | N | best RMI | lookup | vs BTree-128 | size vs BTree-128 |
|---|---|---|---|---|---|
| dense | 100M | K=1000 | 33.1 ns | **11.09×** faster | **262× smaller** |
| uniform_sparse | 100M | K=100000 | 110.9 ns | **3.36×** faster | 2.6× smaller |
| lognormal | 100M | K=100000 | 196.5 ns | **1.92×** faster | 2.6× smaller |
| dense | 10M | K=1000 | 26.5 ns | **10.28×** faster | 26× smaller |
| uniform_sparse | 10M | K=100000 | 69.3 ns | **3.95×** faster | 3.8× *larger* |
| lognormal | 10M | K=100000 | 118.2 ns | **2.31×** faster | 3.8× *larger* |

The RMI is faster than BTree-128 in all 18 of its configurations.

### Where the results diverge from the usual expectations

Four claims commonly made about this comparison do **not** survive measurement
here, and the divergences are the interesting part:

1. **"On dense data, search time ≈ 0."** Not so. The RMI does achieve exact
   prediction on dense (`avg |err| = 0.00`, `max window = 1` at every K), yet
   search still accounts for ~87% of the lookup (29.3 ns of 33.1 ns at 100M).
   A perfect model drives the *comparison count* to zero, not the *memory
   access*: you still pay one DRAM round trip to fetch `data[pred]` and confirm
   the key. Model accuracy cannot buy that back — it is the floor for any index
   over an out-of-cache array.

2. **"Model execution is 20-40% of RMI lookup time."** It is 1.9-16.9%
   (typically 2-5% at 100M). Two stages of linear arithmetic cost 3.8-5.5 ns and
   never miss cache; the search around it is DRAM-bound. Amusingly, 20-40% is a
   good description of the *B-Tree's* traversal share (14-48%), which is where
   that intuition really belongs.

3. **"Smaller B-Tree pages are faster but larger."** True at 10M (page=32 at
   226 ns vs page=256 at 288 ns) and gone at 100M, where all four page sizes sit
   within 6% of each other (350-378 ns). Once every level misses cache, tree
   height stops mattering much and the traversal is pure pointer-chase latency.
   Note page=32 has the *largest* index (24.6 MB) yet is never the slowest.

4. **"RMI is 10-100× smaller."** Strongly N-dependent, because the RMI's size
   is set by K alone while the B-Tree's grows with N. At 100M, K=1000 is 262×
   smaller and K=10000 is 26× smaller. At 10M, only K=1000 clears 10× — and
   K=100000 is 3.8× *larger* than BTree-128 (2.29 MB of models against 0.60 MB
   of fence keys). Speed and size come from different K values, so there is no
   single configuration that wins both by the advertised margin.

### The lognormal weak spot

Lognormal is the one dataset where the RMI underperforms the 1.5-3× band at
small K (1.12× at K=1000, 1.32× at K=10000, both at 100M). The cause is
structural rather than a defect: log-transforming lognormal keys makes them
*normally* distributed, so the CDF in log-space is a sigmoid. A single **linear**
stage-1 model — under either candidate transform — compresses the dense middle
of that sigmoid, and the stage-2 models landing there inherit enormous key
counts (avg search window 92,261, max 2,240,722). Only at K=100000 do the
segments get fine enough to recover (avg window 2,916, 1.92×).

This is precisely the case the paper handles with a richer first stage: §5.1
reports that "simple (0 hidden layers) to semi-complex (2 hidden layers and 8-
or 16-wide) models for the first stage work the best." A linear stage 1 is the
floor of the design space, not its ceiling. The obvious next step, if this
number matters, is a stage-1 model that can represent a sigmoid — a small MLP,
or a piecewise-linear fit on the empirical CDF.

## Interpreting `Model(ns)` and `Search(ns)`

`Model(ns)` is the RMI's two-stage prediction (`predict_only`), or the B-Tree's
root-to-leaf descent (`traverse_only`) — both measured as separate timed loops
over the same sample. `Search(ns)` is the remainder, `lookup_ns - model_ns`.

Because the lookups in the timing loop are independent, the CPU overlaps many of
their cache misses. These are therefore **throughput** numbers under
memory-level parallelism, not isolated single-lookup latencies. That is the
right model for a scan-heavy or batched workload, and it is applied identically
to both structures; a dependency-chained variant would raise the absolute
numbers for both and widen the RMI's lead, since it issues one dependent miss
where the B-Tree issues one per level.
# Learned-Index
