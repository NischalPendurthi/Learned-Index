// Straight-line stage 1 against piecewise stage 1, in one process with the two
// taking turns, so a change in machine speed can't favour either.
//   g++ -std=c++17 -O3 -march=native -o stage1_bench src/*.cpp bench/stage1_bench.cpp
//   ./stage1_bench [--n 100000000] [--lookups 1000000] [--rounds 7] [--out FILE] [--sosd PATH]

#include <algorithm>
#include <chrono>
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

constexpr u64 kDataSeed = 0x5EED1234ULL;
constexpr u64 kSampleSeed = 0xA11CE99ULL;
constexpr u64 kAbsentSeed = 0xBEEF4242ULL;

volatile i64 g_sink = 0;

template <typename Index>
double run_once(const Index& index, const vu64& sample) {
    auto start = Clock::now();
    for (u64 k : sample) g_sink = index.lookup(k);
    auto end = Clock::now();
    return std::chrono::duration<double, std::nano>(end - start).count() / sample.size();
}

double median(std::vector<double> v) {
    std::sort(ALL(v));
    return v[v.size() / 2];
}

bool verify(const RMI& rmi, const vu64& data, const vu64& absent) {
    for (sz i = 0; i < data.size(); ++i)
        if (rmi.lookup(data[i]) != SC(i64, i)) return false;
    for (u64 k : absent)
        if (rmi.lookup(k) != -1) return false;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    sz n = 100'000'000;
    sz lookups = 1'000'000;
    int rounds = 7;
    std::string out_csv = "stage1_results.csv";
    std::string sosd_path;  // empty = generated lognormal keys

    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--n")) n = std::strtoull(argv[i + 1], nullptr, 10);
        else if (!std::strcmp(argv[i], "--lookups")) lookups = std::strtoull(argv[i + 1], nullptr, 10);
        else if (!std::strcmp(argv[i], "--rounds")) rounds = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--out")) out_csv = argv[i + 1];
        else if (!std::strcmp(argv[i], "--sosd")) sosd_path = argv[i + 1];
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
    std::fprintf(f, "dataset,n,k,stage1,avg_window,max_window,index_size_bytes,lookup_ns\n");

    const char* dataset = sosd_path.empty() ? "lognormal" : "sosd";
    vu64 data = sosd_path.empty() ? generate_dataset(DatasetKind::Lognormal, n, kDataSeed)
                                  : load_sosd(sosd_path.c_str(), n);
    n = data.size();
    vu64 sample = sample_existing_keys(data, lookups, kSampleSeed);
    vu64 absent = sample_absent_keys(data, 1000, kAbsentSeed);

    BTreeIndex btree;
    btree.build(data.data(), data.size(), 128);
    std::vector<double> btree_ns;

    for (sz k : {1000UL, 10000UL, 100000UL}) {
        RMI straight, piecewise;
        straight.build(data.data(), data.size(), k, false);
        piecewise.build(data.data(), data.size(), k, true);

        if (!verify(straight, data, absent) || !verify(piecewise, data, absent)) {
            std::fprintf(stderr, "FAIL: correctness check at K=%zu\n", k);
            return 1;
        }

        std::vector<double> a, b;
        for (int r = 0; r < rounds; ++r) {
            btree_ns.push_back(run_once(btree, sample));
            if (r % 2 == 0) {
                a.push_back(run_once(straight, sample));
                b.push_back(run_once(piecewise, sample));
            } else {
                b.push_back(run_once(piecewise, sample));
                a.push_back(run_once(straight, sample));
            }
        }

        for (const RMI* rmi : {&straight, &piecewise}) {
            double ns = median(rmi == &straight ? a : b);
            std::fprintf(stderr, "K=%-7zu %-10s avg window %10.1f  max window %8zu  lookup %7.1f ns\n", k,
                         rmi->stage1_name(), rmi->avg_search_window(), rmi->max_search_window(), ns);
            std::fprintf(f, "%s,%zu,%zu,%s,%.2f,%zu,%zu,%.2f\n", dataset, n, k, rmi->stage1_name(),
                         rmi->avg_search_window(), rmi->max_search_window(), rmi->index_size_bytes(), ns);
        }
        std::fflush(f);
    }
    double bt = median(btree_ns);
    std::fprintf(stderr, "btree page=128  lookup %7.1f ns\n", bt);
    std::fprintf(f, "%s,%zu,0,btree128,0,0,%zu,%.2f\n", dataset, n, btree.index_size_bytes(), bt);
    std::fclose(f);
    return 0;
}
