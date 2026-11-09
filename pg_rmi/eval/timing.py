#!/usr/bin/env python3
"""Re-measures every timing from eval.py, with btree and rmi taking turns.

eval.py measures one index and then the other. If the machine's speed changes
during the run, that order becomes part of the result, so here each round runs
both indexes and alternates which goes first. Medians over rounds are reported.

Run after eval.py, which creates the tables. Updates pg_summary.csv and replaces
pg_range.csv and pg_cold.csv.

    python3 eval/timing.py [--rounds 5]
"""
import argparse
import csv
import os
import statistics
import subprocess

import eval as ev


def server_side_function(table):
    # Static SQL, so plpgsql reuses one cached plan. This times the executor,
    # index and heap access with no client round trip or parsing per lookup.
    ev.sql(f"""
CREATE OR REPLACE FUNCTION eval_lookup_{table}(sample regclass) RETURNS float8 AS $$
DECLARE keys bigint[]; key_val bigint; v int; t0 timestamptz; n int;
BEGIN
    EXECUTE format('SELECT array_agg(k) FROM %s', sample) INTO keys;
    n := array_length(keys, 1);
    t0 := clock_timestamp();
    FOREACH key_val IN ARRAY keys LOOP
        SELECT payload INTO v FROM {table} WHERE k = key_val;
    END LOOP;
    RETURN extract(epoch FROM clock_timestamp() - t0) * 1e9 / n;
END $$ LANGUAGE plpgsql;""")


def order(round_no):
    return ev.KINDS if round_no % 2 == 0 else list(reversed(ev.KINDS))


def med(values):
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seconds", type=int, default=8)
    args = parser.parse_args()
    rounds, seconds = args.rounds, args.seconds

    subprocess.run([ev.CLUSTER, "restart"], check=True)
    build, server_ns, latency, tps, ranges, cold = {}, {}, {}, {}, {}, {}

    # eval.py's drift experiment leaves 25% extra rows in the lognormal tables.
    with open(os.path.join(ev.RESULTS, "pg_summary.csv")) as f:
        rows = int(next(csv.DictReader(f))["rows"])
    ev.log("restoring lognormal tables to", rows, "rows")
    ev.load_dataset("lognormal", rows)

    ev.log("index builds")
    for r in range(3):
        for ds in ev.DATASETS:
            for kind in order(r):
                seconds_taken = ev.timed("SET max_parallel_maintenance_workers = 0; "
                                         f"SET maintenance_work_mem = '2GB'; REINDEX INDEX {ds}_{kind}_idx")
                build.setdefault((ds, kind), []).append(seconds_taken)
    ev.sql("CHECKPOINT")

    for ds in ev.DATASETS:
        for kind in ev.KINDS:
            server_side_function(f"{ds}_{kind}")

    ev.log("warm point lookups")
    for r in range(rounds):
        for ds in ev.DATASETS:
            for kind in order(r):
                table, index, sample = f"{ds}_{kind}", f"{ds}_{kind}_idx", f"{ds}_sample"
                ev.prewarm(table, index)
                ev.sql(f"SELECT eval_lookup_{table}('{sample}')")  # first pass settles the caches
                server_ns.setdefault((ds, kind), []).append(
                    float(ev.sql(f"SELECT eval_lookup_{table}('{sample}')")))
                script = ev.point_script(table, sample)
                latency.setdefault((ds, kind), []).append(ev.pgbench(script, "-c", "1", "-T", str(seconds))[0])
                tps.setdefault((ds, kind), []).append(
                    ev.pgbench(script, "-c", "8", "-j", "8", "-T", str(seconds))[1])
        ev.log("round", r + 1, "done")

    ev.log("range scans")
    for r in range(min(rounds, 3)):
        for ds in ev.DATASETS:
            for width in (100, 10000):
                for kind in order(r):
                    table, index = f"{ds}_{kind}", f"{ds}_{kind}_idx"
                    ev.prewarm(table, index)
                    script = ev.range_script(table, f"{ds}_ranges_{width}")
                    ranges.setdefault((ds, kind, width), []).append(
                        ev.pgbench(script, "-c", "1", "-T", str(seconds))[0])

    ev.log("cold lookups with direct I/O")
    ev.sql("ALTER SYSTEM SET debug_io_direct = 'data'")
    subprocess.run([ev.CLUSTER, "restart"], check=True)
    try:
        for ds in ev.DATASETS:
            ev.sql(f"SELECT pg_prewarm('{ds}_sample'), pg_prewarm('{ds}_sample_pkey')")
        for r in range(rounds):
            for ds in ev.DATASETS:
                for lookups in (500, 5000):
                    for kind in order(r):
                        table, index, sample = f"{ds}_{kind}", f"{ds}_{kind}_idx", f"{ds}_sample"
                        ev.evict(table, index)
                        cold.setdefault((ds, kind, lookups), []).append(
                            ev.pgbench(ev.point_script(table, sample), "-c", "1", "-t", str(lookups))[0])
            ev.log("cold round", r + 1, "done")
    finally:
        ev.sql("ALTER SYSTEM RESET debug_io_direct")
        subprocess.run([ev.CLUSTER, "restart"], check=True)

    path = os.path.join(ev.RESULTS, "pg_summary.csv")
    with open(path) as f:
        summary = list(csv.DictReader(f))
    for row in summary:
        key = (row["dataset"], row["index"])
        row["build_s"] = round(med(build[key]), 2)
        row["server_lookup_us"] = round(med(server_ns[key]) / 1000.0, 3)
        row["warm_latency_us"] = round(med(latency[key]), 1)
        row["warm_tps_8c"] = round(med(tps[key]), 1)
    ev.write_csv("pg_summary.csv", summary)

    ev.write_csv("pg_range.csv", [
        {"dataset": ds, "index": kind, "rows_per_range": width, "latency_us": round(med(v), 1)}
        for (ds, kind, width), v in ranges.items()])
    ev.write_csv("pg_cold.csv", [
        {"dataset": ds, "index": kind, "lookups_after_evict": n, "latency_us": round(med(v), 1),
         "min_us": round(min(v), 1), "max_us": round(max(v), 1)}
        for (ds, kind, n), v in cold.items()])
    ev.log("done")


if __name__ == "__main__":
    main()
