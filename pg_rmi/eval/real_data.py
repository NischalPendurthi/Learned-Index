#!/usr/bin/env python3
"""rmi vs btree on real keys (SOSD), with the two indexes taking turns.

    python3 eval/real_data.py /path/to/fb_200M_uint64 [--rows 10000000] [--rounds 5]

Writes eval/results/pg_real.csv.
"""
import argparse
import statistics

import eval as ev
import timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--name", default="fb")
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    ds = args.name

    ev.build_generator()
    ev.setup_sql()
    results = ev.load_dataset(ds, args.rows, source=f"sosd:{args.path}")
    ev.sql("CHECKPOINT")

    builds, server_ns = {}, {}
    for r in range(3):
        for kind in timing.order(r):
            builds.setdefault(kind, []).append(
                ev.timed("SET max_parallel_maintenance_workers = 0; SET maintenance_work_mem = '2GB'; "
                         f"REINDEX INDEX {ds}_{kind}_idx"))

    for kind in ev.KINDS:
        timing.server_side_function(f"{ds}_{kind}")

    for r in range(args.rounds):
        for kind in timing.order(r):
            table, index, sample = f"{ds}_{kind}", f"{ds}_{kind}_idx", f"{ds}_sample"
            ev.prewarm(table, index)
            ev.sql(f"SELECT eval_lookup_{table}('{sample}')")
            server_ns.setdefault(kind, []).append(float(ev.sql(f"SELECT eval_lookup_{table}('{sample}')")))

    rows = []
    for row in results:
        kind = row["index"]
        table, index = f"{ds}_{kind}", f"{ds}_{kind}_idx"
        ev.prewarm(table, index)
        row["rows"] = ev.sql(f"SELECT count(*) FROM {table}")
        row["build_s"] = round(statistics.median(builds[kind]), 2)
        row["index_buffers_hit"] = round(ev.index_buffers(table, f"{ds}_sample"), 3)
        row["index_buffers_absent"] = round(ev.index_buffers(table, f"{ds}_sample", absent=True), 3)
        row["server_lookup_us"] = round(statistics.median(server_ns[kind]) / 1000.0, 3)
        row["stats"] = (ev.sql(f"SELECT models || ' models, ' || stage1 || ', avg window ' || round(avg_window::numeric, 1) "
                               f"|| ', max window ' || max_window FROM rmi_stats('{index}')")
                        if kind == "rmi" else
                        "height " + ev.sql(f"SELECT level + 1 FROM bt_metap('{index}')"))
        rows.append(row)
        ev.log(row)
    ev.write_csv("pg_real.csv", rows)


if __name__ == "__main__":
    main()
