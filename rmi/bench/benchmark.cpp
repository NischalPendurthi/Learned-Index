// Lookup-timing harness for the RMI on its own.
//
// The full RMI-vs-B-Tree comparison lives in the parent project
// (../../bench/benchmark.cpp).  This one answers a narrower question: how long
// does an RMI lookup take, and how much of that is the two-stage model versus
// the search inside the predicted bounds?
//
//   ./rmi_bench --dataset all --n 10000000 --k 1000,10000,100000

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"
#include "dataset.h"
#include "rmi.h"

using namespace rmi;
using Clock = std::chrono::high_resolution_clock;

namespace {

constexpr size_t kWarmup = 1000;

// Written every iteration so the lookup cannot be optimized away.
volatile int64_t g_sink = 0;

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

template <typename Fn>
double measure_ns(Fn&& fn, const std::vector<Key>& sample, int reps) {
    for (size_t i = 0; i < kWarmup && i < sample.size(); ++i) g_sink = fn(sample[i]);
    std::vector<double> per_rep;
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        for (size_t i = 0; i < sample.size(); ++i) g_sink = fn(sample[i]);
        auto t1 = Clock::now();
        double ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        per_rep.push_back(ns / static_cast<double>(sample.size()));
    }
    return median(std::move(per_rep));
}

std::vector<size_t> parse_size_list(const std::string& s) {
    std::vector<size_t> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        std::string part = (comma == std::string::npos) ? s.substr(start)
                                                        : s.substr(start, comma - start);
        if (!part.empty()) out.push_back(std::strtoull(part.c_str(), nullptr, 10));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string datasets_arg = "all";
    std::vector<size_t> ns = {10'000'000};
    std::vector<size_t> ks = {1000, 10000, 100000};
    size_t num_lookups = 1'000'000;
    int reps = 5;
    uint64_t seed = DEFAULT_SEED;
    std::string csv_path = "rmi_lookup_bench.csv";

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--dataset") && i + 1 < argc) datasets_arg = argv[++i];
        else if (!std::strcmp(argv[i], "--n") && i + 1 < argc) ns = parse_size_list(argv[++i]);
        else if (!std::strcmp(argv[i], "--k") && i + 1 < argc) ks = parse_size_list(argv[++i]);
        else if (!std::strcmp(argv[i], "--lookups") && i + 1 < argc) num_lookups = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
        else { std::fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
    }

    std::vector<DatasetKind> kinds;
    if (datasets_arg == "all") {
        kinds = {DatasetKind::Dense, DatasetKind::Lognormal, DatasetKind::UniformSparse};
    } else {
        DatasetKind kind;
        if (!parse_dataset(datasets_arg, &kind)) {
            std::fprintf(stderr, "unknown dataset: %s\n", datasets_arg.c_str());
            return 2;
        }
        kinds.push_back(kind);
    }

    FILE* f = std::fopen(csv_path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "could not write %s\n", csv_path.c_str()); return 1; }
    std::fprintf(f, "dataset,n,k,search_mode,index_size_bytes,err_mean,"
                    "lookup_ns,model_ns,search_ns,model_pct\n");

    std::printf("| %-14s | %9s | %8s | %-11s | %10s | %10s | %10s | %7s |\n", "Dataset", "N", "K",
                "Search", "Lookup(ns)", "Model(ns)", "Search(ns)", "Model %");
    std::printf("|----------------|-----------|----------|-------------|------------|"
                "------------|------------|---------|\n");

    for (DatasetKind kind : kinds) {
        for (size_t n : ns) {
            std::vector<Key> data = generate(kind, n, seed);
            if (data.size() != n) {
                std::fprintf(stderr, "dataset generation failed for %s N=%zu\n", to_string(kind), n);
                return 1;
            }
            std::vector<Key> sample = sample_present(data, num_lookups, seed ^ 0xA11CE99ULL);

            for (size_t k : ks) {
                for (SearchMode mode : {SearchMode::Binary, SearchMode::Exponential}) {
                    RMI index;
                    index.set_search_mode(mode);
                    index.train(data.data(), data.size(), k);

                    // Sanity check before timing: no point measuring a broken index.
                    bool ok = true;
                    for (size_t i = 0; i < data.size() && ok; i += (data.size() / 10000) + 1) {
                        LookupResult r = index.lookup(data[i]);
                        if (!r.found || r.position != i) ok = false;
                    }
                    if (!ok) {
                        std::fprintf(stderr, "correctness spot-check failed for %s N=%zu K=%zu\n",
                                     to_string(kind), n, k);
                        return 1;
                    }

                    double lookup_ns = measure_ns(
                        [&](Key key) { return static_cast<int64_t>(index.lookup(key).position); },
                        sample, reps);
                    double model_ns = measure_ns(
                        [&](Key key) { return index.predict_position(key); }, sample, reps);
                    double search_ns = lookup_ns - model_ns;
                    double model_pct = (lookup_ns > 0) ? 100.0 * model_ns / lookup_ns : 0.0;
                    const char* mname = (mode == SearchMode::Exponential) ? "exponential" : "binary";

                    std::printf("| %-14s | %9zu | %8zu | %-11s | %10.2f | %10.2f | %10.2f | %6.1f%% |\n",
                                to_string(kind), n, k, mname, lookup_ns, model_ns, search_ns,
                                model_pct);
                    std::fflush(stdout);

                    std::fprintf(f, "%s,%zu,%zu,%s,%zu,%.3f,%.3f,%.3f,%.3f,%.2f\n",
                                 to_string(kind), n, k, mname, index.index_size_bytes(),
                                 index.stats().err_mean, lookup_ns, model_ns, search_ns, model_pct);
                    std::fflush(f);
                }
            }
        }
    }

    std::fclose(f);
    std::printf("\nwrote %s\n", csv_path.c_str());
    return 0;
}
