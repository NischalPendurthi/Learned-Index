// Writes a dataset's keys to stdout in random order, one per line. Uses the same
// generators as the C++ benchmark, so the Postgres runs see the same keys.
//   gen_keys dense|lognormal|uniform_sparse|sosd:PATH N SEED
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "../../src/dataset.hpp"

using namespace lidx;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s dataset n seed\n", argv[0]);
        return 2;
    }
    sz n_arg = std::strtoull(argv[2], nullptr, 10);
    u64 seed_arg = std::strtoull(argv[3], nullptr, 10);

    // "sosd:PATH" reads a real dataset. A bigint can't hold keys of 2^63 or
    // more, so those are dropped.
    if (!std::strncmp(argv[1], "sosd:", 5)) {
        vu64 keys = load_sosd(argv[1] + 5, n_arg);
        sz before = keys.size();
        keys.erase(std::remove_if(ALL(keys), [](u64 k) { return k >> 63; }), keys.end());
        std::fprintf(stderr, "sosd: %zu keys, dropped %zu above bigint range\n", keys.size(), before - keys.size());
        std::mt19937_64 rng(seed_arg);
        std::shuffle(ALL(keys), rng);
        static char sbuf[1 << 20];
        std::setvbuf(stdout, sbuf, _IOFBF, sizeof(sbuf));
        for (u64 k : keys) std::printf("%llu\n", static_cast<unsigned long long>(k));
        return 0;
    }

    DatasetKind kind;
    if (!std::strcmp(argv[1], "dense")) kind = DatasetKind::Dense;
    else if (!std::strcmp(argv[1], "lognormal")) kind = DatasetKind::Lognormal;
    else if (!std::strcmp(argv[1], "uniform_sparse")) kind = DatasetKind::UniformSparse;
    else {
        std::fprintf(stderr, "unknown dataset %s\n", argv[1]);
        return 2;
    }
    sz n = std::strtoull(argv[2], nullptr, 10);
    u64 seed = std::strtoull(argv[3], nullptr, 10);

    vu64 keys = generate_dataset(kind, n, seed);
    // A heap is not in key order, and the first lines double as a random sample.
    std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ULL);
    std::shuffle(ALL(keys), rng);

    static char buf[1 << 20];
    std::setvbuf(stdout, buf, _IOFBF, sizeof(buf));
    for (u64 k : keys) std::printf("%llu\n", static_cast<unsigned long long>(k));
    return 0;
}
