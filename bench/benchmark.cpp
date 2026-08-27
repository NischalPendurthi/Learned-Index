// Benchmark harness: Recursive Model Index vs. a cache-optimized read-only
// B-Tree, over the same sorted array and the same lookup workload.
//
//   g++ -std=c++17 -O3 -march=native -o benchmark src/*.cpp bench/benchmark.cpp
//
// Usage:
//   ./benchmark [--sizes 10000000,100000000] [--lookups 1000000] [--reps 5]
//               [--out benchmark_results.csv] [--verify-sample 0]
//
// --verify-sample K checks K sampled keys instead of all N during the
// correctness phase (0 = check every key, the default).

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/btree.hpp"
#include "../src/dataset.hpp"
#include "../src/rmi.hpp"

using namespace lidx;
using Clock = std::chrono::high_resolution_clock;

namespace {

constexpr uint64_t kDataSeed = 0x5EED1234ULL;
constexpr uint64_t kSampleSeed = 0xA11CE99ULL;
constexpr uint64_t kAbsentSeed = 0xBEEF4242ULL;
constexpr size_t kWarmup = 1000;
constexpr size_t kAbsentChecks = 1000;
constexpr int kReferencePage = 128;  // BTree-128 is the reference, per the paper

// Written on every timed iteration so the lookup cannot be optimized away.
volatile int64_t g_sink = 0;

struct Row {
    std::string dataset;
    std::string index_type;
    std::string config;
    size_t num_keys = 0;
    size_t index_size_bytes = 0;
    double lookup_ns = 0;
    double model_ns = 0;
    // Filled in once the BTree-128 reference for this (dataset, N) is known.
    double speedup = 0;
    double size_ratio = 0;
    size_t group = 0;  // (dataset, N) group id
    bool is_reference = false;
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    if (v.size() % 2) return v[m];
    return 0.5 * (v[m - 1] + v[m]);
}

// Times fn over the sample, reps times, and returns the median ns per call.
// Build/training time is never inside this loop.
template <typename Fn>
double measure_ns_per_op(Fn&& fn, const std::vector<uint64_t>& sample, size_t num_lookups,
                         int reps) {
    for (size_t i = 0; i < kWarmup && i < sample.size(); ++i) g_sink = static_cast<int64_t>(fn(sample[i]));

    std::vector<double> per_rep;
    per_rep.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        auto start = Clock::now();
        for (size_t i = 0; i < num_lookups; ++i) {
            g_sink = static_cast<int64_t>(fn(sample[i]));
        }
        auto end = Clock::now();
        double total_ns =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        per_rep.push_back(total_ns / static_cast<double>(num_lookups));
    }
    return median(std::move(per_rep));
}

// 100% recall over the dataset, and -1 for keys that are absent.
template <typename Index>
bool verify(const Index& index, const std::vector<uint64_t>& data,
            const std::vector<uint64_t>& absent, const char* label, size_t verify_sample) {
    size_t n = data.size();
    size_t bad = 0;
    uint64_t first_bad_key = 0;
    int64_t first_bad_pos = 0;
    size_t first_bad_i = 0;

    if (verify_sample == 0 || verify_sample >= n) {
        for (size_t i = 0; i < n; ++i) {
            int64_t pos = index.lookup(data[i]);
            if (pos != static_cast<int64_t>(i)) {
                if (!bad) { first_bad_key = data[i]; first_bad_pos = pos; first_bad_i = i; }
                ++bad;
            }
        }
    } else {
        // Deterministic stride sample, plus both endpoints.
        size_t stride = n / verify_sample;
        if (stride == 0) stride = 1;
        for (size_t i = 0; i < n; i += stride) {
            int64_t pos = index.lookup(data[i]);
            if (pos != static_cast<int64_t>(i)) {
                if (!bad) { first_bad_key = data[i]; first_bad_pos = pos; first_bad_i = i; }
                ++bad;
            }
        }
        if (index.lookup(data[n - 1]) != static_cast<int64_t>(n - 1)) ++bad;
    }

    size_t bad_absent = 0;
    for (uint64_t k : absent) {
        if (index.lookup(k) != -1) ++bad_absent;
    }

    if (bad || bad_absent) {
        std::fprintf(stderr,
                     "  [FAIL] %-22s  missed/misplaced keys: %zu   false positives: %zu\n",
                     label, bad, bad_absent);
        if (bad) {
            std::fprintf(stderr,
                         "         first failure: key %" PRIu64 " expected pos %zu, got %" PRId64 "\n",
                         first_bad_key, first_bad_i, first_bad_pos);
        }
        return false;
    }
    std::fprintf(stderr, "  [ok]   %-22s  recall 100%%, %zu absent keys -> -1\n", label,
                 absent.size());
    return true;
}

// 200000 -> "200K", 10000000 -> "10M"
std::string format_count(size_t n) {
    char buf[32];
    if (n >= 1'000'000 && n % 1'000'000 == 0) std::snprintf(buf, sizeof(buf), "%zuM", n / 1'000'000);
    else if (n >= 1000 && n % 1000 == 0) std::snprintf(buf, sizeof(buf), "%zuK", n / 1000);
    else std::snprintf(buf, sizeof(buf), "%zu", n);
    return buf;
}

std::vector<size_t> parse_sizes(const char* s) {
    std::vector<size_t> out;
    const char* p = s;
    while (*p) {
        char* endp = nullptr;
        unsigned long long v = std::strtoull(p, &endp, 10);
        if (endp == p) break;
        out.push_back(static_cast<size_t>(v));
        p = endp;
        while (*p == ',' || *p == ' ') ++p;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<size_t> sizes = {10'000'000, 100'000'000};
    std::vector<size_t> rmi_ks = {1000, 10000, 100000};
    std::vector<int> pages = {32, 64, 128, 256};
    size_t num_lookups = 1'000'000;
    int reps = 5;
    size_t verify_sample = 0;
    std::string out_csv = "benchmark_results.csv";

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--sizes") && i + 1 < argc) sizes = parse_sizes(argv[++i]);
        else if (!std::strcmp(argv[i], "--lookups") && i + 1 < argc) num_lookups = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_csv = argv[++i];
        else if (!std::strcmp(argv[i], "--verify-sample") && i + 1 < argc) verify_sample = std::strtoull(argv[++i], nullptr, 10);
        else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    if (num_lookups < 100'000) {
        std::fprintf(stderr,
                     "warning: %zu lookups is below the 100K minimum; timings will be noisy\n",
                     num_lookups);
    }

    const DatasetKind kinds[] = {DatasetKind::Dense, DatasetKind::Lognormal,
                                 DatasetKind::UniformSparse};

    std::vector<Row> rows;
    size_t group_id = 0;

    for (DatasetKind kind : kinds) {
        for (size_t n : sizes) {
            const char* dname = dataset_name(kind);
            std::fprintf(stderr, "\n===== %s, N = %zu =====\n", dname, n);

            std::fprintf(stderr, "generating dataset...\n");
            auto t0 = Clock::now();
            std::vector<uint64_t> data = generate_dataset(kind, n, kDataSeed);
            auto t1 = Clock::now();
            std::fprintf(stderr, "  %zu unique sorted keys in %.1f s  [min %" PRIu64 ", max %" PRIu64 "]\n",
                         data.size(), std::chrono::duration<double>(t1 - t0).count(), data.front(),
                         data.back());

            // ---- build (never timed as part of a lookup measurement) --------
            std::vector<BTreeIndex> btrees(pages.size());
            for (size_t p = 0; p < pages.size(); ++p) {
                btrees[p].build(data.data(), data.size(), pages[p]);
            }
            std::vector<RMI> rmis(rmi_ks.size());
            for (size_t k = 0; k < rmi_ks.size(); ++k) {
                std::fprintf(stderr, "building RMI K=%zu...\n", rmi_ks[k]);
                rmis[k].build(data.data(), data.size(), rmi_ks[k]);
            }

            // ---- structure report -------------------------------------------
            std::fprintf(stderr, "\nstructure:\n");
            for (size_t p = 0; p < pages.size(); ++p) {
                std::fprintf(stderr,
                             "  BTree page=%-4d  size %9.3f MB   height %d   leaf pages %zu\n",
                             pages[p], btrees[p].index_size_bytes() / 1048576.0, btrees[p].height(),
                             btrees[p].num_leaf_pages());
            }
            for (size_t k = 0; k < rmi_ks.size(); ++k) {
                std::fprintf(stderr,
                             "  RMI   K=%-6zu  size %9.3f MB   stage1 %-9s  avg |err| %10.2f   "
                             "avg window %10.2f   max window %zu\n",
                             rmi_ks[k], rmis[k].index_size_bytes() / 1048576.0,
                             rmis[k].stage1_name(), rmis[k].avg_abs_error(),
                             rmis[k].avg_search_window(), rmis[k].max_search_window());
            }

            // ---- correctness, before any timing ------------------------------
            std::fprintf(stderr, "\ncorrectness checks:\n");
            std::vector<uint64_t> absent = sample_absent_keys(data, kAbsentChecks, kAbsentSeed);
            bool ok = true;
            for (size_t p = 0; p < pages.size(); ++p) {
                char label[64];
                std::snprintf(label, sizeof(label), "BTree page=%d", pages[p]);
                ok &= verify(btrees[p], data, absent, label, verify_sample);
            }
            for (size_t k = 0; k < rmi_ks.size(); ++k) {
                char label[64];
                std::snprintf(label, sizeof(label), "RMI K=%zu", rmi_ks[k]);
                ok &= verify(rmis[k], data, absent, label, verify_sample);
            }
            if (!ok) {
                std::fprintf(stderr,
                             "\ncorrectness FAILED for %s N=%zu -- refusing to benchmark.\n",
                             dname, n);
                return 1;
            }

            // ---- workload ----------------------------------------------------
            // One sample, shared by every index in this group.
            std::vector<uint64_t> sample = sample_existing_keys(data, num_lookups, kSampleSeed);

            std::fprintf(stderr, "\nbenchmarking (%zu lookups x %d reps, median)...\n", num_lookups,
                         reps);

            for (size_t p = 0; p < pages.size(); ++p) {
                const BTreeIndex& bt = btrees[p];
                double lookup_ns = measure_ns_per_op([&](uint64_t k) { return bt.lookup(k); },
                                                     sample, num_lookups, reps);
                double trav_ns = measure_ns_per_op(
                    [&](uint64_t k) { return static_cast<int64_t>(bt.traverse_only(k)); }, sample,
                    num_lookups, reps);
                Row r;
                r.dataset = dname;
                r.index_type = "BTree";
                r.config = "page=" + std::to_string(pages[p]);
                r.num_keys = n;
                r.index_size_bytes = bt.index_size_bytes();
                r.lookup_ns = lookup_ns;
                r.model_ns = trav_ns;
                r.group = group_id;
                r.is_reference = (pages[p] == kReferencePage);
                rows.push_back(std::move(r));
                std::fprintf(stderr, "  BTree page=%-4d  lookup %7.2f ns   traverse %7.2f ns\n",
                             pages[p], lookup_ns, trav_ns);
            }

            for (size_t k = 0; k < rmi_ks.size(); ++k) {
                const RMI& rmi = rmis[k];
                double lookup_ns = measure_ns_per_op([&](uint64_t key) { return rmi.lookup(key); },
                                                     sample, num_lookups, reps);
                double model_ns = measure_ns_per_op(
                    [&](uint64_t key) { return static_cast<int64_t>(rmi.predict_only(key)); },
                    sample, num_lookups, reps);
                Row r;
                r.dataset = dname;
                r.index_type = "RMI";
                r.config = "K=" + std::to_string(rmi_ks[k]);
                r.num_keys = n;
                r.index_size_bytes = rmi.index_size_bytes();
                r.lookup_ns = lookup_ns;
                r.model_ns = model_ns;
                r.group = group_id;
                rows.push_back(std::move(r));
                std::fprintf(stderr, "  RMI   K=%-6zu  lookup %7.2f ns   model    %7.2f ns\n",
                             rmi_ks[k], lookup_ns, model_ns);
            }

            ++group_id;
        }
    }

    // ---- normalize against BTree-128 within each (dataset, N) group ---------
    for (size_t g = 0; g < group_id; ++g) {
        double ref_lookup = 0, ref_size = 0;
        for (const Row& r : rows) {
            if (r.group == g && r.is_reference) {
                ref_lookup = r.lookup_ns;
                ref_size = static_cast<double>(r.index_size_bytes);
            }
        }
        for (Row& r : rows) {
            if (r.group != g) continue;
            r.speedup = (r.lookup_ns > 0) ? ref_lookup / r.lookup_ns : 0.0;
            r.size_ratio = (ref_size > 0) ? static_cast<double>(r.index_size_bytes) / ref_size : 0.0;
        }
    }

    // ---- table --------------------------------------------------------------
    const int widths[] = {20, 5, 10, 11, 10, 10, 10, 7, 8, 10};
    std::string sep = "|";
    for (int w : widths) sep += std::string(w + 2, '-') + "|";

    std::printf("\n");
    std::printf("| %-20s | %-5s | %-10s | %11s | %10s | %10s | %10s | %7s | %8s | %10s |\n",
                "Dataset", "Index", "Config", "Size (MB)", "Lookup(ns)", "Model(ns)", "Search(ns)",
                "Model %", "Speedup", "Size ratio");
    std::printf("%s\n", sep.c_str());

    size_t last_group = SIZE_MAX;
    for (const Row& r : rows) {
        if (last_group != SIZE_MAX && r.group != last_group) std::printf("%s\n", sep.c_str());
        last_group = r.group;
        double search_ns = r.lookup_ns - r.model_ns;
        double model_pct = (r.lookup_ns > 0) ? 100.0 * r.model_ns / r.lookup_ns : 0.0;
        std::string dset = r.dataset + "/" + format_count(r.num_keys);
        std::printf("| %-20s | %-5s | %-10s | %11.4f | %10.2f | %10.2f | %10.2f | %6.1f%% | %7.2fx | %10.4f |\n",
                    dset.c_str(), r.index_type.c_str(), r.config.c_str(),
                    r.index_size_bytes / 1048576.0, r.lookup_ns, r.model_ns, search_ns, model_pct,
                    r.speedup, r.size_ratio);
    }
    std::printf(
        "\nSpeedup and size ratio are relative to BTree page=128 on the same dataset and N.\n"
        "Model(ns) is the RMI's two-stage prediction, or the B-Tree's root-to-leaf descent.\n"
        "Index size excludes the sorted data array, which both structures share.\n");

    // ---- CSV ----------------------------------------------------------------
    FILE* f = std::fopen(out_csv.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "could not open %s for writing\n", out_csv.c_str());
        return 1;
    }
    std::fprintf(f,
                 "dataset,index_type,config,num_keys,index_size_bytes,index_size_mb,"
                 "lookup_ns,model_ns,search_ns,model_pct,speedup_vs_btree128,size_ratio_vs_btree128\n");
    for (const Row& r : rows) {
        double search_ns = r.lookup_ns - r.model_ns;
        double model_pct = (r.lookup_ns > 0) ? 100.0 * r.model_ns / r.lookup_ns : 0.0;
        std::fprintf(f, "%s,%s,%s,%zu,%zu,%.6f,%.3f,%.3f,%.3f,%.2f,%.4f,%.6f\n", r.dataset.c_str(),
                     r.index_type.c_str(), r.config.c_str(), r.num_keys, r.index_size_bytes,
                     r.index_size_bytes / 1048576.0, r.lookup_ns, r.model_ns, search_ns, model_pct,
                     r.speedup, r.size_ratio);
    }
    std::fclose(f);
    std::fprintf(stderr, "\nwrote %s (%zu rows)\n", out_csv.c_str(), rows.size());
    return 0;
}
