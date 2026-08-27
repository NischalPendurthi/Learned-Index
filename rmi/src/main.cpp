// Entry point: build a 2-stage RMI over each requested dataset, print the
// training report, run the correctness tests, and save results for the
// benchmark harness to consume.
//
//   ./rmi_main --dataset all --n 1000000 --k 10000

#include <algorithm>
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

namespace {

struct Result {
    std::string dataset;
    size_t n = 0;
    size_t k = 0;
    Key key_min = 0, key_max = 0;
    TrainStats stats;
    bool all_keys_found = false;
    size_t false_negatives = 0;
    size_t misplaced = 0;
    size_t false_positives = 0;
    bool routing_balanced = false;
};

// ---- test 1 & 2 ---------------------------------------------------------
// Every key must be found at its exact position; no absent key may be found.
void run_correctness(const RMI& index, const std::vector<Key>& data,
                     const std::vector<Key>& absent, size_t verify_limit, Result* out) {
    size_t n = data.size();
    size_t stride = 1;
    if (verify_limit > 0 && verify_limit < n) stride = n / verify_limit;
    if (stride == 0) stride = 1;

    size_t missing = 0, misplaced = 0;
    Key first_bad = 0;
    size_t first_bad_i = 0;
    bool have_bad = false;

    for (size_t i = 0; i < n; i += stride) {
        LookupResult r = index.lookup(data[i]);
        if (!r.found) {
            ++missing;
            if (!have_bad) { first_bad = data[i]; first_bad_i = i; have_bad = true; }
        } else if (r.position != i) {
            ++misplaced;
            if (!have_bad) { first_bad = data[i]; first_bad_i = i; have_bad = true; }
        }
    }
    // Always probe both endpoints, whatever the stride.
    if (!index.lookup(data[0]).found) ++missing;
    if (!index.lookup(data[n - 1]).found) ++missing;

    size_t fp = 0;
    for (Key k : absent) {
        if (index.lookup(k).found) ++fp;
    }

    out->false_negatives = missing;
    out->misplaced = misplaced;
    out->false_positives = fp;
    out->all_keys_found = (missing == 0 && misplaced == 0);

    if (have_bad) {
        std::printf("      first failure: key %" PRIu64 " at position %zu\n", first_bad,
                    first_bad_i);
    }
}

void print_report(const Result& r) {
    const TrainStats& s = r.stats;
    std::printf("\n");
    std::printf("  Dataset            : %s\n", r.dataset.c_str());
    std::printf("  N keys             : %zu\n", r.n);
    std::printf("  Key range          : [%" PRIu64 ", %" PRIu64 "]\n", r.key_min, r.key_max);
    std::printf("  Training time      : %.3f s\n", s.train_seconds);
    std::printf("  Stage-2 models (K) : %zu   (empty: %zu, single-key: %zu)\n", s.num_models,
                s.empty_models, s.single_key_models);
    std::printf("  Index size         : %zu bytes (%.3f MB, %zu models x %zu B incl. root)\n",
                s.index_size_bytes, s.index_size_bytes / 1048576.0, s.num_models + 1,
                sizeof(LinearModel));
    std::printf("  Keys per model     : min %zu  max %zu  mean %.1f  median %.1f\n", s.keys_min,
                s.keys_max, s.keys_mean, s.keys_median);
    std::printf("  Largest bucket     : %zu keys (%.2f%% of N)\n", s.keys_max,
                100.0 * s.max_bucket_share);
    std::printf("  Model error width  : min %" PRId64 "  max %" PRId64 "  mean %.2f  p50 %.2f  p99 %.2f\n",
                s.err_min, s.err_max, s.err_mean, s.err_p50, s.err_p99);
    std::printf("                       (width = max_error - min_error over non-empty models;\n");
    std::printf("                        0 means the model is exact and search inspects 1 slot)\n");
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, comma - start));
        start = comma + 1;
    }
    return out;
}

std::vector<size_t> parse_size_list(const std::string& s) {
    std::vector<size_t> out;
    for (const std::string& part : split_csv(s)) {
        if (part.empty()) continue;
        out.push_back(std::strtoull(part.c_str(), nullptr, 10));
    }
    return out;
}

void write_csv(const std::string& path, const std::vector<Result>& rows) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "could not write %s\n", path.c_str()); return; }
    std::fprintf(f,
                 "dataset,n,k,key_min,key_max,train_seconds,index_size_bytes,num_models,"
                 "empty_models,single_key_models,keys_min,keys_max,keys_mean,keys_median,"
                 "max_bucket_share,err_min,err_max,err_mean,err_p50,err_p99,"
                 "all_keys_found,false_negatives,misplaced,false_positives\n");
    for (const Result& r : rows) {
        const TrainStats& s = r.stats;
        std::fprintf(f,
                     "%s,%zu,%zu,%" PRIu64 ",%" PRIu64 ",%.6f,%zu,%zu,%zu,%zu,%zu,%zu,%.3f,%.3f,"
                     "%.6f,%" PRId64 ",%" PRId64 ",%.3f,%.3f,%.3f,%d,%zu,%zu,%zu\n",
                     r.dataset.c_str(), r.n, r.k, r.key_min, r.key_max, s.train_seconds,
                     s.index_size_bytes, s.num_models, s.empty_models, s.single_key_models,
                     s.keys_min, s.keys_max, s.keys_mean, s.keys_median, s.max_bucket_share,
                     s.err_min, s.err_max, s.err_mean, s.err_p50, s.err_p99,
                     r.all_keys_found ? 1 : 0, r.false_negatives, r.misplaced, r.false_positives);
    }
    std::fclose(f);
    std::printf("\nwrote %s (%zu rows)\n", path.c_str(), rows.size());
}

void write_json(const std::string& path, const std::vector<Result>& rows) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "could not write %s\n", path.c_str()); return; }
    std::fprintf(f, "[\n");
    for (size_t i = 0; i < rows.size(); ++i) {
        const Result& r = rows[i];
        const TrainStats& s = r.stats;
        std::fprintf(f,
                     "  {\"dataset\": \"%s\", \"n\": %zu, \"k\": %zu, \"key_min\": %" PRIu64
                     ", \"key_max\": %" PRIu64 ", \"train_seconds\": %.6f, "
                     "\"index_size_bytes\": %zu, \"num_models\": %zu, \"empty_models\": %zu, "
                     "\"single_key_models\": %zu, \"keys_min\": %zu, \"keys_max\": %zu, "
                     "\"keys_mean\": %.3f, \"keys_median\": %.3f, \"max_bucket_share\": %.6f, "
                     "\"err_min\": %" PRId64 ", \"err_max\": %" PRId64 ", \"err_mean\": %.3f, "
                     "\"err_p50\": %.3f, \"err_p99\": %.3f, \"all_keys_found\": %s, "
                     "\"false_negatives\": %zu, \"misplaced\": %zu, \"false_positives\": %zu}%s\n",
                     r.dataset.c_str(), r.n, r.k, r.key_min, r.key_max, s.train_seconds,
                     s.index_size_bytes, s.num_models, s.empty_models, s.single_key_models,
                     s.keys_min, s.keys_max, s.keys_mean, s.keys_median, s.max_bucket_share,
                     s.err_min, s.err_max, s.err_mean, s.err_p50, s.err_p99,
                     r.all_keys_found ? "true" : "false", r.false_negatives, r.misplaced,
                     r.false_positives, (i + 1 < rows.size()) ? "," : "");
    }
    std::fprintf(f, "]\n");
    std::fclose(f);
    std::printf("wrote %s\n", path.c_str());
}

void usage(const char* prog) {
    std::printf(
        "usage: %s [options]\n"
        "  --dataset LIST   dense,lognormal,uniform_sparse or 'all'  (default: all)\n"
        "  --n LIST         dataset sizes, comma separated           (default: %zu)\n"
        "  --k LIST         stage-2 model counts, comma separated    (default: %zu)\n"
        "  --seed S         RNG seed                                 (default: %llu)\n"
        "  --search MODE    binary | exponential                     (default: binary)\n"
        "  --verify-limit M check M strided keys instead of all N    (default: 0 = all)\n"
        "  --csv PATH       results CSV        (default: rmi_results.csv)\n"
        "  --json PATH      results JSON       (default: rmi_results.json)\n",
        prog, DEFAULT_N, DEFAULT_K, static_cast<unsigned long long>(DEFAULT_SEED));
}

}  // namespace

int main(int argc, char** argv) {
    std::string datasets_arg = "all";
    std::vector<size_t> ns = {DEFAULT_N};
    std::vector<size_t> ks = {DEFAULT_K};
    uint64_t seed = DEFAULT_SEED;
    SearchMode mode = SearchMode::Binary;
    size_t verify_limit = 0;
    std::string csv_path = "rmi_results.csv";
    std::string json_path = "rmi_results.json";

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--dataset") && i + 1 < argc) datasets_arg = argv[++i];
        else if (!std::strcmp(argv[i], "--n") && i + 1 < argc) ns = parse_size_list(argv[++i]);
        else if (!std::strcmp(argv[i], "--k") && i + 1 < argc) ks = parse_size_list(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--verify-limit") && i + 1 < argc) verify_limit = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
        else if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
        else if (!std::strcmp(argv[i], "--search") && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "exponential") mode = SearchMode::Exponential;
            else if (m == "binary") mode = SearchMode::Binary;
            else { std::fprintf(stderr, "unknown search mode: %s\n", m.c_str()); return 2; }
        } else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    std::vector<DatasetKind> kinds;
    if (datasets_arg == "all") {
        kinds = {DatasetKind::Dense, DatasetKind::Lognormal, DatasetKind::UniformSparse};
    } else {
        for (const std::string& name : split_csv(datasets_arg)) {
            DatasetKind kind;
            if (!parse_dataset(name, &kind)) {
                std::fprintf(stderr, "unknown dataset: %s\n", name.c_str());
                return 2;
            }
            kinds.push_back(kind);
        }
    }
    if (ns.empty() || ks.empty()) { std::fprintf(stderr, "empty --n or --k list\n"); return 2; }

    std::vector<Result> results;
    bool hard_failure = false;   // tests 1 and 2
    int quality_warnings = 0;    // tests 3, 4, 5

    for (DatasetKind kind : kinds) {
        for (size_t n : ns) {
            std::printf("\n============================================================\n");
            std::printf("  %s, N = %zu\n", to_string(kind), n);
            std::printf("============================================================\n");

            std::printf("generating dataset...\n");
            std::vector<Key> data = generate(kind, n, seed);
            if (data.size() != n) {
                std::fprintf(stderr, "dataset generation failed for %s N=%zu\n", to_string(kind), n);
                return 1;
            }
            std::vector<Key> absent = sample_absent(data, ABSENT_PROBES, seed ^ 0x9E3779B9ULL);

            std::vector<double> mean_widths;  // for test 4, across the K values

            for (size_t k : ks) {
                RMI index;
                index.set_search_mode(mode);
                index.train(data.data(), data.size(), k);

                Result r;
                r.dataset = to_string(kind);
                r.n = n;
                r.k = k;
                r.key_min = data.front();
                r.key_max = data.back();
                r.stats = index.stats();
                r.routing_balanced = (index.stats().max_bucket_share <= 0.5);

                print_report(r);

                std::printf("\n  correctness:\n");
                run_correctness(index, data, absent, verify_limit, &r);

                bool t1 = r.all_keys_found;
                bool t2 = (r.false_positives == 0);
                bool t3 = r.routing_balanced;
                bool t5 = (kind != DatasetKind::Dense) || (r.stats.err_max == 0);

                std::printf("    [%s] test 1  every key found at its position%s\n",
                            t1 ? "PASS" : "FAIL",
                            t1 ? "" : "  <-- FALSE NEGATIVES");
                if (!t1) {
                    std::printf("           missing: %zu, misplaced: %zu\n", r.false_negatives,
                                r.misplaced);
                }
                std::printf("    [%s] test 2  %zu absent keys report not-found%s\n",
                            t2 ? "PASS" : "FAIL", absent.size(),
                            t2 ? "" : "  <-- FALSE POSITIVES");
                std::printf("    [%s] test 3  no bucket holds >50%% of keys (largest %.2f%%)\n",
                            t3 ? "PASS" : "WARN", 100.0 * r.stats.max_bucket_share);
                if (kind == DatasetKind::Dense) {
                    std::printf("    [%s] test 5  dense fit is exact (max error width %" PRId64 ")\n",
                                t5 ? "PASS" : "WARN", r.stats.err_max);
                }

                if (!t1 || !t2) hard_failure = true;
                if (!t3 || !t5) ++quality_warnings;

                mean_widths.push_back(r.stats.err_mean);
                results.push_back(std::move(r));
            }

            // ---- test 4: error width should shrink as K grows -----------------
            if (ks.size() >= 2) {
                bool shrinks = true;
                for (size_t i = 1; i < mean_widths.size(); ++i) {
                    if (!(mean_widths[i] <= mean_widths[i - 1])) shrinks = false;
                }
                std::printf("\n  [%s] test 4  mean error width shrinks with more stage-2 models\n",
                            shrinks ? "PASS" : "WARN");
                std::printf("           ");
                for (size_t i = 0; i < ks.size(); ++i) {
                    std::printf("K=%zu: %.1f%s", ks[i], mean_widths[i],
                                (i + 1 < ks.size()) ? "   " : "\n");
                }
                if (!shrinks) ++quality_warnings;
            }
        }
    }

    write_csv(csv_path, results);
    write_json(json_path, results);

    std::printf("\n============================================================\n");
    if (hard_failure) {
        std::printf("  RESULT: CORRECTNESS FAILED (tests 1/2) in at least one config\n");
    } else {
        std::printf("  RESULT: correctness passed for all %zu configurations\n", results.size());
    }
    if (quality_warnings) {
        std::printf("  %d quality warning(s) from tests 3/4/5 -- see above\n", quality_warnings);
    }
    std::printf("============================================================\n");
    return hard_failure ? 1 : 0;
}
