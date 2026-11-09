#!/usr/bin/env python3
"""Charts for the PostgreSQL evaluation and the hybrid index.

    python3 plots/make_pg_plots.py        # writes plots/5_*.png to 8_*.png

Inputs : pg_rmi/eval/results/*.csv and results_hybrid_100M.csv
Needs  : matplotlib only.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "plots")
RES = os.path.join(ROOT, "pg_rmi", "eval", "results")

# Same identity colours as make_plots.py: blue = RMI, orange = B-Tree.
RMI, BTREE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
COLORS = {"rmi": RMI, "btree": BTREE}
NAMES = {"rmi": "rmi (learned index)", "btree": "btree"}
DATASETS = ["dense", "uniform_sparse", "lognormal"]

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,
})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def legend(fig):
    fig.legend(handles=[Patch(color=BTREE, label=NAMES["btree"]), Patch(color=RMI, label=NAMES["rmi"])],
               loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.03))


def grouped(ax, groups, value, fmt="{:.0f}"):
    """groups: list of (label, {index_kind: number})."""
    width = 0.36
    top = max(v for _, vals in groups for v in vals.values())
    for i, (_, vals) in enumerate(groups):
        for j, kind in enumerate(["btree", "rmi"]):
            x = i + (j - 0.5) * (width + 0.03)
            ax.bar(x, vals[kind], width=width, color=COLORS[kind], edgecolor="white", linewidth=1)
            ax.text(x, vals[kind] + top * 0.015, fmt.format(vals[kind]), ha="center", va="bottom",
                    fontsize=8.5, color=INK_2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, top * 1.14)
    ax.set_title(value)


def by_dataset(rows, field):
    out = []
    for ds in DATASETS:
        vals = {r["index"]: float(r[field]) for r in rows if r["dataset"] == ds}
        out.append((ds, vals))
    return out


def chart_summary():
    rows = read(os.path.join(RES, "pg_summary.csv"))
    n = int(rows[0]["rows"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    grouped(axes[0], by_dataset(rows, "index_buffers_hit"), "Index pages read per lookup", "{:.2f}")
    grouped(axes[1], by_dataset(rows, "index_mb"), "Index size (MB)")
    grouped(axes[2], by_dataset(rows, "server_lookup_us"), "Point lookup, all pages cached (µs, server side)", "{:.2f}")
    legend(fig)
    fig.suptitle(f"rmi vs btree inside PostgreSQL 18, {n // 1_000_000}M rows",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "5_pg_summary.png")


def chart_cold():
    rows = read(os.path.join(RES, "pg_cold.csv"))
    counts = sorted({int(r["lookups_after_evict"]) for r in rows})
    fig, axes = plt.subplots(1, len(counts), figsize=(10, 3.9), sharey=True)
    for ax, count in zip(axes, counts):
        subset = [r for r in rows if int(r["lookups_after_evict"]) == count]
        grouped(ax, by_dataset(subset, "latency_us"), f"Average of the first {count:,} lookups")
    axes[0].set_ylabel("lookup latency (µs)")
    top = max(float(r["latency_us"]) for r in rows) * 1.14
    for ax in axes:
        ax.set_ylim(0, top)
    legend(fig)
    fig.suptitle("Lookup latency starting from an empty cache (direct I/O)",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "6_pg_cold.png")


def chart_drift():
    rows = read(os.path.join(RES, "pg_drift.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fields = [("index_buffers_hit", "Index pages read per lookup"), ("warm_latency_us", "Point lookup, all pages cached (µs)")]
    for ax, (field, title) in zip(axes, fields):
        top = 0.0
        for kind in ["btree", "rmi"]:
            pts = [r for r in rows if r["index"] == kind and "REINDEX" not in r["step"]]
            xs = [float(r["new_key_pct"]) for r in pts]
            ys = [float(r[field]) for r in pts]
            top = max(top, *ys)
            ax.plot(xs, ys, color=COLORS[kind], lw=2, marker="o", markersize=8,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            if field == "index_buffers_hit":  # the lines nearly touch on the other panel
                ax.annotate(NAMES[kind], (xs[-1], ys[-1]), xytext=(0, 9), textcoords="offset points",
                            ha="right", fontsize=8.5, color=INK_2)
        after = next(r for r in rows if r["index"] == "rmi" and "REINDEX" in r["step"])
        ax.scatter([float(after["new_key_pct"])], [float(after[field])], s=90, marker="D", color=RMI,
                   edgecolor="white", linewidth=1.5, zorder=4)
        ax.annotate("rmi after REINDEX", (float(after["new_key_pct"]), float(after[field])), xytext=(-8, -13),
                    textcoords="offset points", ha="right", fontsize=8.5, color=INK_2)
        ax.set_ylim(0, top * 1.2)
        ax.set_xlabel("new keys inserted since build (% of table)")
        ax.set_title(title)
    legend(fig)
    fig.suptitle("Inserts degrade the learned index until it is retrained (lognormal)",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "7_pg_drift.png")


def chart_hybrid():
    path = os.path.join(ROOT, "results_hybrid_100M.csv")
    if not os.path.exists(path):
        print("skipping hybrid chart, no", os.path.relpath(path, ROOT))
        return
    rows = [r for r in read(path) if r["dataset"] == "lognormal"]
    ks = sorted({int(r["k"]) for r in rows})
    thresholds = [-1, 4096, 512, 64]
    labels = ["pure RMI", "t=4096", "t=512", "t=64"]
    fig, axes = plt.subplots(1, len(ks), figsize=(12, 3.9), sharey=True)
    top = max(max(float(r["avg_lookup_ns"]), float(r["worst_lookup_ns"])) for r in rows) * 1.14
    for ax, k in zip(axes, ks):
        pts = {int(r["threshold"]): r for r in rows if int(r["k"]) == k}
        width = 0.36
        for i, t in enumerate(thresholds):
            avg, worst = float(pts[t]["avg_lookup_ns"]), float(pts[t]["worst_lookup_ns"])
            ax.bar(i - 0.2, avg, width=width, color=RMI, edgecolor="white", linewidth=1)
            ax.bar(i + 0.2, worst, width=width, color="#9ec5f4", edgecolor="white", linewidth=1)
            ax.text(i - 0.2, avg + top * 0.012, f"{avg:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)
            ax.text(i + 0.2, worst + top * 0.012, f"{worst:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, top)
        ax.set_title(f"K = {k // 1000}k models")
    axes[0].set_ylabel("lookup time (ns)")
    fig.legend(handles=[Patch(color=RMI, label="all keys (average)"),
                        Patch(color="#9ec5f4", label="keys in the worst model")],
               loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.03))
    fig.suptitle("Hybrid index, 100M lognormal keys",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "8_hybrid.png")


if __name__ == "__main__":
    chart_summary()
    chart_cold()
    chart_drift()
    chart_hybrid()
