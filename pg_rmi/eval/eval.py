#!/usr/bin/env python3
"""rmi vs btree inside PostgreSQL. Needs the private cluster from ../cluster.sh.

    python3 eval/eval.py [--rows 10000000] [--seconds 15] [--reps 3]

Each dataset is loaded into two tables with the same rows in the same heap order,
one indexed by btree and one by rmi. Results go to eval/results/*.csv.

Parallel index builds are turned off so both indexes build on one core.
"""
import argparse
import csv
import os
import statistics
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PG_RMI = os.path.dirname(HERE)
CLUSTER = os.path.join(PG_RMI, "cluster.sh")
REPO = os.path.dirname(PG_RMI)
RESULTS = os.path.join(HERE, "results")
SEED, NEW_KEY_SEED = 20261109, 777001
SAMPLE = 200_000
DATASETS = ["dense", "uniform_sparse", "lognormal"]
KINDS = ["btree", "rmi"]

ENV = dict(os.environ)
for pair in subprocess.check_output([CLUSTER, "env"], text=True).split():
    name, value = pair.split("=", 1)
    ENV[name] = value
BIN = ENV["BIN"]
DATA_ROOT = os.path.join(os.path.dirname(ENV["DATA"]), "eval_data")


def sql(statement):
    return subprocess.run([f"{BIN}/psql", "-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c", statement],
                          env=ENV, check=True, capture_output=True, text=True).stdout.strip()


def timed(statement):
    start = time.perf_counter()
    sql(statement)
    return time.perf_counter() - start


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def write_csv(name, rows):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log("wrote", os.path.relpath(path, REPO))


def pgbench(script_text, *flags):
    """Runs a script and returns the average latency in microseconds of its last
    statement, which keeps key sampling out of the measurement."""
    with tempfile.NamedTemporaryFile("w", suffix=".sql") as f:
        f.write(script_text)
        f.flush()
        out = subprocess.run([f"{BIN}/pgbench", "-n", "-M", "prepared", "-r", *flags, "-f", f.name],
                             env=ENV, check=True, capture_output=True, text=True).stdout
    if "number of failed transactions: 0 " not in out:
        raise RuntimeError(out)
    last_statement = script_text.strip().splitlines()[-1].strip().split()[0]
    in_table, latency = False, None
    for line in out.splitlines():
        if line.startswith("statement latencies"):
            in_table = True
        elif in_table and last_statement in line:
            latency = float(line.split()[0]) * 1000.0
    tps = next(float(l.split("=")[1].split()[0]) for l in out.splitlines() if l.startswith("tps"))
    return latency, tps


def point_script(table, sample):
    return (f"\\set i random(1, {SAMPLE})\n"
            f"SELECT k AS key FROM {sample} WHERE id = :i \\gset\n"
            f"SELECT payload FROM {table} WHERE k = :key;\n")


def range_script(table, ranges):
    return (f"\\set i random(1, 2000)\n"
            f"SELECT lo, hi FROM {ranges} WHERE id = :i \\gset\n"
            f"SELECT sum(payload) FROM {table} WHERE k BETWEEN :lo AND :hi;\n")


def prewarm(table, index):
    sql(f"SELECT pg_prewarm('{table}'), pg_prewarm('{index}')")


def evict(table, index):
    sql(f"SELECT pg_buffercache_evict_relation('{table}'), pg_buffercache_evict_relation('{index}')")


def index_buffers(table, sample, absent=False):
    offset = 1 if absent else 0
    return float(sql(f"SELECT eval_index_buffers('{table}', '{sample}', 2000, {offset})"))


def median_latency(script, reps, *flags):
    results = [pgbench(script, *flags) for _ in range(reps)]
    return statistics.median(r[0] for r in results), statistics.median(r[1] for r in results)


def setup_sql():
    sql("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
    sql("CREATE EXTENSION IF NOT EXISTS pg_buffercache")
    sql("CREATE EXTENSION IF NOT EXISTS pageinspect")
    sql("CREATE EXTENSION IF NOT EXISTS rmi")
    # Bitmap Index Scan nodes report the index's own page accesses, without the heap.
    sql(r"""
CREATE OR REPLACE FUNCTION eval_index_buffers(tbl regclass, sample regclass, n int, key_offset bigint)
RETURNS float8 AS $$
DECLARE
    k_val bigint; plan json; node json; total bigint := 0; cnt int := 0;
BEGIN
    PERFORM set_config('enable_seqscan', 'off', true);
    PERFORM set_config('enable_indexscan', 'off', true);
    PERFORM set_config('enable_bitmapscan', 'on', true);
    -- The first scan in a backend loads the rmi models. That is a one-time cost
    -- (reported as model_kb), so keep it out of the per-lookup average.
    EXECUTE format('SELECT count(*) FROM %s WHERE k = 0', tbl);
    FOR k_val IN EXECUTE format('SELECT k FROM %s WHERE id <= %s', sample, n) LOOP
        EXECUTE format('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT payload FROM %s WHERE k = %s',
                       tbl, k_val + key_offset) INTO plan;
        node := plan->0->'Plan'->'Plans'->0;
        IF node->>'Node Type' <> 'Bitmap Index Scan' THEN
            RAISE EXCEPTION 'unexpected plan node %', node->>'Node Type';
        END IF;
        total := total + (node->>'Shared Hit Blocks')::bigint + (node->>'Shared Read Blocks')::bigint;
        cnt := cnt + 1;
    END LOOP;
    RETURN total::float8 / cnt;
END $$ LANGUAGE plpgsql;
""")


def key_file(name, dataset, n, seed):
    os.makedirs(DATA_ROOT, exist_ok=True)
    path = os.path.join(DATA_ROOT, name)
    if not os.path.exists(path):
        log("generating", name)
        with open(path, "w") as f:
            subprocess.run([os.path.join(DATA_ROOT, "gen_keys"), dataset, str(n), str(seed)],
                           stdout=f, check=True)
    return path


def build_generator():
    subprocess.run(["g++", "-std=c++17", "-O2", "-o", os.path.join(DATA_ROOT, "gen_keys"),
                    os.path.join(HERE, "gen_keys.cpp"), os.path.join(REPO, "src", "dataset.cpp")],
                   check=True)


def index_create(kind, table, index, options=""):
    using = "btree" if kind == "btree" else "rmi"
    return (f"SET max_parallel_maintenance_workers = 0; SET maintenance_work_mem = '2GB'; "
            f"CREATE INDEX {index} ON {table} USING {using} (k) {options}")


def load_dataset(ds, rows, source=None):
    path = key_file(f"{ds}_{rows}.txt", source or ds, rows, SEED)
    results = []
    for kind in KINDS:
        table, index = f"{ds}_{kind}", f"{ds}_{kind}_idx"
        sql(f"DROP TABLE IF EXISTS {table}")
        sql(f"CREATE TABLE {table} (k bigint NOT NULL, payload int NOT NULL DEFAULT 1) "
            f"WITH (autovacuum_enabled = off)")
        log("loading", table)
        sql(f"COPY {table} (k) FROM '{path}'")
        sql(f"VACUUM (FREEZE, ANALYZE) {table}")
        build_seconds = timed(index_create(kind, table, index))
        sql(f"ANALYZE {table}")
        results.append({
            "dataset": ds, "index": kind, "rows": rows, "build_s": round(build_seconds, 2),
            "index_mb": round(int(sql(f"SELECT pg_relation_size('{index}')")) / 1048576, 2),
            "table_mb": round(int(sql(f"SELECT pg_relation_size('{table}')")) / 1048576, 2),
        })

    sample = f"{ds}_sample"
    sql(f"DROP TABLE IF EXISTS {sample}")
    sql(f"CREATE TABLE {sample} (id serial PRIMARY KEY, k bigint)")
    sql(f"CREATE TEMP TABLE x (k bigint); COPY x FROM PROGRAM 'head -n {SAMPLE} {path}'; "
        f"INSERT INTO {sample} (k) SELECT k FROM x")
    sql(f"ANALYZE {sample}")

    # Ranges covering about 100 and 10,000 rows, from the btree table.
    for width in (100, 10000):
        ranges = f"{ds}_ranges_{width}"
        sql(f"DROP TABLE IF EXISTS {ranges}")
        sql(f"CREATE TABLE {ranges} (id serial PRIMARY KEY, lo bigint, hi bigint)")
        sql(f"""INSERT INTO {ranges} (lo, hi)
                SELECT s.k, (SELECT k FROM {ds}_btree b WHERE b.k >= s.k ORDER BY b.k OFFSET {width - 1} LIMIT 1)
                FROM (SELECT k FROM {sample} WHERE id <= 2400) s""")
        sql(f"DELETE FROM {ranges} WHERE hi IS NULL")
        sql(f"WITH o AS (SELECT id, row_number() OVER (ORDER BY id) rn FROM {ranges}) "
            f"UPDATE {ranges} r SET id = o.rn FROM o WHERE r.id = o.id")
        sql(f"DELETE FROM {ranges} WHERE id > 2000")
        sql(f"ANALYZE {ranges}")
    return results


def warm_measure(ds, kind, seconds, reps):
    table, index, sample = f"{ds}_{kind}", f"{ds}_{kind}_idx", f"{ds}_sample"
    prewarm(table, index)
    pgbench(point_script(table, sample), "-c", "1", "-T", "3")  # settle
    latency, _ = median_latency(point_script(table, sample), reps, "-c", "1", "-T", str(seconds))
    _, tps = median_latency(point_script(table, sample), reps, "-c", "8", "-j", "8", "-T", str(seconds))
    return latency, tps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    rows, seconds, reps = args.rows, args.seconds, args.reps

    os.makedirs(DATA_ROOT, exist_ok=True)
    build_generator()
    setup_sql()

    summary = []
    for ds in DATASETS:
        for row in load_dataset(ds, rows):
            summary.append(row)
    sql("CHECKPOINT")

    log("warm point lookups, range scans, buffers")
    ranges_out = []
    for row in summary:
        ds, kind = row["dataset"], row["index"]
        table, index = f"{ds}_{kind}", f"{ds}_{kind}_idx"
        prewarm(table, index)
        row["index_buffers_hit"] = round(index_buffers(table, f"{ds}_sample"), 3)
        row["index_buffers_absent"] = round(index_buffers(table, f"{ds}_sample", absent=True), 3)
        row["warm_latency_us"], row["warm_tps_8c"] = (round(v, 2) for v in warm_measure(ds, kind, seconds, reps))
        if kind == "rmi":
            row["stats"] = sql(f"SELECT models || ' models, ' || stage1 || ', avg window ' || round(avg_window::numeric, 1) "
                               f"|| ', max window ' || max_window FROM rmi_stats('{index}')")
        else:
            row["stats"] = "height " + sql(f"SELECT level + 1 FROM bt_metap('{index}')")
        log(ds, kind, row["warm_latency_us"], "us", row["index_buffers_hit"], "index buffers")

        for width in (100, 10000):
            latency, _ = median_latency(range_script(table, f"{ds}_ranges_{width}"), reps, "-c", "1", "-T", str(seconds))
            ranges_out.append({"dataset": ds, "index": kind, "rows_per_range": width, "latency_us": round(latency, 2)})
    write_csv("pg_summary.csv", summary)
    write_csv("pg_range.csv", ranges_out)

    log("cold lookups with direct I/O")
    cold_out = []
    sql("ALTER SYSTEM SET debug_io_direct = 'data'")
    subprocess.run([CLUSTER, "restart"], check=True)
    try:
        for ds in DATASETS:
            for kind in KINDS:
                table, index, sample = f"{ds}_{kind}", f"{ds}_{kind}_idx", f"{ds}_sample"
                sql(f"SELECT pg_prewarm('{sample}'), pg_prewarm('{sample}_pkey')")
                for lookups in (500, 5000):
                    latencies = []
                    for _ in range(reps):
                        evict(table, index)
                        latencies.append(pgbench(point_script(table, sample), "-c", "1", "-t", str(lookups))[0])
                    cold_out.append({"dataset": ds, "index": kind, "lookups_after_evict": lookups,
                                     "latency_us": round(statistics.median(latencies), 2)})
                    log(ds, kind, "cold", lookups, cold_out[-1]["latency_us"], "us")
    finally:
        sql("ALTER SYSTEM RESET debug_io_direct")
        subprocess.run([CLUSTER, "restart"], check=True)
    write_csv("pg_cold.csv", cold_out)

    log("model count sweep (lognormal)")
    sweep = []
    ds = "lognormal"
    table, index = f"{ds}_rmi", f"{ds}_rmi_idx"
    for models in (100, 1000, 10000, 100000, 500000):
        sql(f"DROP INDEX {index}")
        build_seconds = timed(index_create("rmi", table, index, f"WITH (models = {models})"))
        sql(f"ANALYZE {table}")
        prewarm(table, index)
        latency, _ = median_latency(point_script(table, f"{ds}_sample"), reps, "-c", "1", "-T", str(seconds))
        stats = sql(f"SELECT stage1, avg_window, max_window FROM rmi_stats('{index}')").split("|")
        sweep.append({"models": models, "build_s": round(build_seconds, 2),
                      "model_kb": round(models * 24 / 1024, 1), "stage1": stats[0],
                      "avg_window": round(float(stats[1]), 1), "max_window": stats[2],
                      "index_buffers_hit": round(index_buffers(table, f"{ds}_sample"), 3),
                      "index_buffers_absent": round(index_buffers(table, f"{ds}_sample", absent=True), 3),
                      "warm_latency_us": round(latency, 2)})
        log(models, sweep[-1])
    write_csv("pg_ksweep.csv", sweep)
    sql(f"DROP INDEX {index}")
    sql(index_create("rmi", table, index))
    sql(f"ANALYZE {table}")

    log("drift experiment (lognormal)")
    new_total = rows // 4 + 1000
    new_path = key_file(f"lognormal_new_{new_total}.txt", "lognormal", new_total, NEW_KEY_SEED)
    old_path = key_file(f"lognormal_{rows}.txt", "lognormal", rows, SEED)
    sql("DROP TABLE IF EXISTS drift_sample")
    sql("CREATE TABLE drift_sample (id serial PRIMARY KEY, k bigint)")

    def drift_point(label, fraction):
        new_in_sample = int(SAMPLE * fraction / (1 + fraction))
        sql("TRUNCATE drift_sample RESTART IDENTITY")
        sql(f"CREATE TEMP TABLE x (k bigint); "
            f"COPY x FROM PROGRAM 'head -n {SAMPLE - new_in_sample} {old_path}'; "
            + (f"COPY x FROM PROGRAM 'head -n {new_in_sample} {new_path}'; " if new_in_sample else "")
            + "INSERT INTO drift_sample (k) SELECT k FROM x ORDER BY random()")
        sql("ANALYZE drift_sample")
        out = []
        for kind in KINDS:
            table, index = f"lognormal_{kind}", f"lognormal_{kind}_idx"
            sql(f"ANALYZE {table}")
            prewarm(table, index)
            pgbench(point_script(table, "drift_sample"), "-c", "1", "-T", "3")
            latency, _ = median_latency(point_script(table, "drift_sample"), reps, "-c", "1", "-T", str(seconds))
            row = {"step": label, "new_key_pct": round(fraction * 100, 1), "index": kind,
                   "index_mb": round(int(sql(f"SELECT pg_relation_size('{index}')")) / 1048576, 2),
                   "index_buffers_hit": round(index_buffers(table, "drift_sample"), 3),
                   "warm_latency_us": round(latency, 2), "overflow_pages": 0, "overflow_entries": 0}
            if kind == "rmi":
                row["overflow_pages"], row["overflow_entries"] = sql(
                    f"SELECT overflow_pages, overflow_entries FROM rmi_stats('{index}')").split("|")
            out.append(row)
            log(label, kind, row)
        return out

    drift = drift_point("baseline", 0.0)
    inserted = 0
    for fraction in (0.01, 0.05, 0.10, 0.25):
        target = int(rows * fraction)
        count = target - inserted
        for kind in KINDS:
            table = f"lognormal_{kind}"
            sql(f"CREATE TEMP TABLE x (k bigint); "
                f"COPY x FROM PROGRAM 'head -n {target} {new_path} | tail -n {count}'; "
                f"INSERT INTO {table} (k) SELECT k FROM x")
        inserted = target
        drift += drift_point(f"after {round(fraction * 100)}% inserts", fraction)

    reindex_seconds = timed("SET max_parallel_maintenance_workers = 0; REINDEX INDEX lognormal_rmi_idx")
    log("REINDEX took", round(reindex_seconds, 1), "s")
    drift += drift_point("after REINDEX", 0.25)
    for row in drift:
        row["reindex_s"] = round(reindex_seconds, 2)
    write_csv("pg_drift.csv", drift)
    log("done")


if __name__ == "__main__":
    main()
