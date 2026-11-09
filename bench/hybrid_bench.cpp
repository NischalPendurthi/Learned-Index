// Hybrid index (Algorithm 1): how much does replacing badly-fitting models with
// small B-Trees help, on average and for the keys in the worst model?
//   g++ -std=c++17 -O3 -march=native -o hybrid_bench src/*.cpp bench/hybrid_bench.cpp
//   ./hybrid_bench [--n 100000000] [--lookups 1000000] [--reps 5] [--out FILE]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../src/dataset.hpp"
#include "../src/hybrid.hpp"

using namespace lidx;
using Clock = std::chrono::high_resolution_clock;

namespace {

constexpr u64 kDataSeed = 0x5EED1234ULL;
constexpr u64 kSampleSeed = 0xA11CE99ULL;
constexpr u64 kAbsentSeed = 0xBEEF4242ULL;
constexpr int kPageSize = 128;

volatile i64 g_sink = 0;

template <typename Fn>
double measure_ns(Fn&& fn, const vu64& sample, int reps) {
    for (sz i = 0; i < std::min<sz>(1000, sample.size()); ++i) g_sink = fn(sample[i]);
    std::vector<double> per_rep;
    for (int r = 0; r < reps; ++r) {
        auto start = Clock::now();
        for (u64 k : sample) g_sink = fn(k);
        auto end = Clock::now();
        per_rep.push_back(std::chrono::duration<double, std::nano>(end - start).count() / sample.size());
    }
    std::sort(ALL(per_rep));
    return per_rep[per_rep.size() / 2];
}

vu64 shuffled_slice(const vu64& data, sz lo, sz hi, sz count) {
    std::mt19937_64 rng(kSampleSeed);
    std::uniform_int_distribution<sz> pick(lo, hi - 1);
    vu64 out(count);
    for (u64& k : out) k = data[pick(rng)];
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    sz n = 100'000'000;
    sz lookups = 1'000'000;
    int reps = 5;
    std::string out_csv = "hybrid_results.csv";

    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--n")) n = std::strtoull(argv[i + 1], nullptr, 10);
        else if (!std::strcmp(argv[i], "--lookups")) lookups = std::strtoull(argv[i + 1], nullptr, 10);
        else if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--out")) out_csv = argv[i + 1];
        else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    FILE* f = std::fopen(out_csv.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "could not open %s\n", out_csv.c_str());
        return 1;
    }
    std::fprintf(f, "dataset,n,k,threshold,replaced_models,index_size_bytes,avg_lookup_ns,"
                    "worst_model_keys,worst_lookup_ns\n");

    for (DatasetKind kind : {DatasetKind::Lognormal, DatasetKind::UniformSparse}) {
        const char* name = dataset_name(kind);
        vu64 data = generate_dataset(kind, n, kDataSeed);
        vu64 sample = sample_existing_keys(data, lookups, kSampleSeed);
        vu64 absent = sample_absent_keys(data, 1000, kAbsentSeed);

        for (sz k : {1000UL, 10000UL, 100000UL}) {
            // SIZE_MAX is the pure RMI: no window can exceed it. Shown as -1 in the CSV.
            for (sz threshold : {SIZE_MAX, 4096UL, 512UL, 64UL}) {
                HybridRMI h;
                h.build(data.data(), data.size(), k, threshold, kPageSize);

                for (sz i = 0; i < data.size(); ++i) {
                    if (h.lookup(data[i]) != SC(i64, i)) {
                        std::fprintf(stderr, "FAIL %s k=%zu: key at %zu not found\n", name, k, i);
                        return 1;
                    }
                }
                for (u64 a : absent) {
                    if (h.lookup(a) != -1) {
                        std::fprintf(stderr, "FAIL %s k=%zu: absent key found\n", name, k);
                        return 1;
                    }
                }

                vu64 worst = shuffled_slice(data, h.worst_lo(), h.worst_hi(), std::min<sz>(lookups, 200000));
                double avg_ns = measure_ns([&](u64 key) { return h.lookup(key); }, sample, reps);
                double worst_ns = measure_ns([&](u64 key) { return h.lookup(key); }, worst, reps);
                long long shown = threshold == SIZE_MAX ? -1 : SC(long long, threshold);

                std::fprintf(stderr, "%-15s K=%-7zu t=%-6lld replaced %-6zu size %8.3f MB  avg %7.1f ns  worst model %7.1f ns\n",
                             name, k, shown, h.num_replaced(), h.index_size_bytes() / 1048576.0, avg_ns, worst_ns);
                std::fprintf(f, "%s,%zu,%zu,%lld,%zu,%zu,%.2f,%zu,%.2f\n", name, n, k, shown,
                             h.num_replaced(), h.index_size_bytes(), avg_ns,
                             h.worst_hi() - h.worst_lo(), worst_ns);
                std::fflush(f);
            }
        }
    }
    std::fclose(f);
    return 0;
}
