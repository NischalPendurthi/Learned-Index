#!/usr/bin/env python3
"""Checkpoint-1 charts from the committed benchmark CSVs.

    python3 plots/make_plots.py        # writes plots/*.png

Inputs : benchmark_results.csv      (RMI vs B-Tree, 10M and 100M)
         rmi/rmi_results.csv        (RMI error widths, 10M / 100M / 200M)
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

# Colour by identity, the same in every chart: blue = RMI, orange = B-Tree.
RMI, RMI_LIGHT = "#2a78d6", "#9ec5f4"
BTREE, BTREE_LIGHT = "#eb6834", "#f6b99f"
LOGNORMAL, UNIFORM = "#6250d6", "#1baf7a"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

DATASETS = ["dense", "uniform_sparse", "lognormal"]
TITLES = {"dense": "dense", "uniform_sparse": "uniform_sparse", "lognormal": "lognormal"}

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
    with open(os.path.join(ROOT, path)) as f:
        return list(csv.DictReader(f))


def short(row):
    """'page=128' -> 'B-Tree\\np=128', 'K=100000' -> 'RMI\\nK=100k'."""
    if row["index_type"] == "BTree":
        return "B-Tree\np=" + row["config"].split("=")[1]
    k = int(row["config"].split("=")[1])
    return "RMI\nK=" + (f"{k // 1000}k" if k >= 1000 else str(k))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


bench = read("benchmark_results.csv")


def rows_for(ds, n):
    return [r for r in bench if r["dataset"] == ds and int(r["num_keys"]) == n]


# --- 1. Lookup time per configuration, small multiples by dataset ----------
def chart_lookup(n, label):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        rows = rows_for(ds, n)
        x = range(len(rows))
        vals = [float(r["lookup_ns"]) for r in rows]
        colors = [RMI if r["index_type"] == "RMI" else BTREE for r in rows]
        ax.bar(x, vals, width=0.72, color=colors, edgecolor="white", linewidth=2)
        for i, v in zip(x, vals):
            ax.text(i, v + 6, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK_2)
        ax.set_xticks(list(x))
        ax.set_xticklabels([short(r) for r in rows], fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_title(TITLES[ds])

        best = min((r for r in rows if r["index_type"] == "RMI"), key=lambda r: float(r["lookup_ns"]))
        ax.text(0.98, 0.97, f"best RMI ({short(best).split(chr(10))[1]}): "
                f"{float(best['speedup_vs_btree128']):.1f}× faster than B-Tree p=128",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK, fontweight="bold")
    axes[0].set_ylabel("lookup time (ns, lower is better)")
    axes[0].set_ylim(0, 460)
    fig.legend(handles=[Patch(color=BTREE, label="B-Tree"), Patch(color=RMI, label="RMI (learned index)")],
               loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.02))
    fig.suptitle(f"Lookup time, N = {label} keys (median of 5 × 1M random lookups)",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, f"1_lookup_time_{label}.png")


# --- 2. Where the time goes: model/traversal vs search (stacked) -----------
def chart_breakdown(n, label):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        rows = [r for r in rows_for(ds, n) if r["index_type"] == "RMI" or r["config"] == "page=128"]
        x = list(range(len(rows)))
        model = [float(r["model_ns"]) for r in rows]
        search = [float(r["search_ns"]) for r in rows]
        dark = [RMI if r["index_type"] == "RMI" else BTREE for r in rows]
        light = [RMI_LIGHT if r["index_type"] == "RMI" else BTREE_LIGHT for r in rows]
        ax.bar(x, model, width=0.62, color=dark, edgecolor="white", linewidth=0.8)
        ax.bar(x, search, width=0.62, bottom=model, color=light, edgecolor="white", linewidth=0.8)
        for i, r in enumerate(rows):
            ax.text(i, model[i] + search[i] + 6, f"{float(r['model_pct']):.0f}% model",
                    ha="center", va="bottom", fontsize=8.5, color=INK_2)
        ax.set_xticks(x)
        ax.set_xticklabels([short(r) for r in rows], fontsize=8.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(TITLES[ds])
    axes[0].set_ylabel("lookup time (ns)")
    axes[0].set_ylim(0, 430)
    fig.legend(handles=[
        Patch(color=BTREE, label="B-Tree: root-to-leaf traversal"),
        Patch(color=BTREE_LIGHT, label="B-Tree: search in leaf page"),
        Patch(color=RMI, label="RMI: run 2 models"),
        Patch(color=RMI_LIGHT, label="RMI: search error window"),
    ], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.suptitle(f"Where lookup time goes, N = {label}: running the model is cheap; memory access dominates",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, f"2_time_breakdown_{label}.png")


# --- 3. Index size vs lookup time ------------------------------------------
def chart_size_speed(n, label):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for r in rows_for(ds, n):
            size_kb = float(r["index_size_bytes"]) / 1024
            ns = float(r["lookup_ns"])
            c = RMI if r["index_type"] == "RMI" else BTREE
            m = "o" if r["index_type"] == "RMI" else "s"
            ax.scatter(size_kb, ns, s=60, color=c, marker=m, edgecolor="white", linewidth=1.5, zorder=3)
            if r["index_type"] == "RMI":
                ax.annotate(short(r).replace("\n", " "), (size_kb, ns), xytext=(6, 4),
                            textcoords="offset points", fontsize=7.5, color=INK_2)
            else:
                # B-Tree points sit 2x apart on the log axis: label each with
                # just its page size, underneath, so the four don't collide.
                ax.annotate(r["config"].split("=")[1], (size_kb, ns), xytext=(0, -14),
                            textcoords="offset points", ha="center", fontsize=7.5, color=INK_2)
        ax.text(5500, 410, "B-Tree, page size:", ha="center", fontsize=7.5, color=INK_2)
        ax.set_xscale("log")
        ax.set_xlim(10, 80000)
        ax.set_title(TITLES[ds])
        ax.set_xlabel("index size (KB, log scale)")
    axes[0].set_ylabel("lookup time (ns)")
    axes[0].set_ylim(0, 430)
    fig.legend(handles=[
        plt.Line2D([], [], marker="s", ls="", color=BTREE, markersize=8, label="B-Tree"),
        plt.Line2D([], [], marker="o", ls="", color=RMI, markersize=8, label="RMI"),
    ], loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.02))
    fig.suptitle(f"Size vs speed, N = {label}: bottom-left is better",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, f"3_size_vs_speed_{label}.png")


# --- 4. RMI error window vs K ----------------------------------------------
def chart_error():
    rows = read("rmi/rmi_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, n, label in zip(axes, [100_000_000, 200_000_000], ["100M", "200M"]):
        for ds, color in [("lognormal", LOGNORMAL), ("uniform_sparse", UNIFORM)]:
            pts = sorted((int(r["k"]), float(r["err_mean"]), float(r["err_max"]))
                         for r in rows if r["dataset"] == ds and int(r["n"]) == n)
            ks = [p[0] for p in pts]
            ax.plot(ks, [p[1] for p in pts], color=color, lw=2, marker="o", markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5)
            # Labelled at K=1k, where the lines are far apart (they meet at 100k).
            ax.text(ks[0] / 1.3, pts[0][1], f"{ds}\nmean", color=INK_2, fontsize=8.5,
                    va="center", ha="right")
            if ds == "lognormal":
                ax.plot(ks, [p[2] for p in pts], color=color, lw=2, ls="--", marker="o", markersize=7,
                        markeredgecolor="white", markeredgewidth=1.5)
                ax.text(ks[0] / 1.3, pts[0][2], "lognormal\nworst model", color=INK_2,
                        fontsize=8.5, va="center", ha="right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(150, 200_000)
        ax.set_xticks([1000, 10000, 100000])
        ax.set_xticklabels(["1k", "10k", "100k"])
        ax.set_xlabel("number of stage-2 models (K)")
        ax.set_title(f"N = {label}   (dense: error = 0 at every K)")
    axes[0].set_ylabel("error window width (positions, log)")
    fig.legend(handles=[
        plt.Line2D([], [], color=LOGNORMAL, lw=2, label="lognormal"),
        plt.Line2D([], [], color=UNIFORM, lw=2, label="uniform_sparse"),
        plt.Line2D([], [], color=MUTED, lw=2, label="mean over models"),
        plt.Line2D([], [], color=MUTED, lw=2, ls="--", label="worst single model"),
    ], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.suptitle("RMI error window: more models shrink the average, but not lognormal's worst bucket",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "4_error_window_vs_K.png")


if __name__ == "__main__":
    for n, label in [(100_000_000, "100M"), (10_000_000, "10M")]:
        chart_lookup(n, label)
        chart_breakdown(n, label)
        chart_size_speed(n, label)
    chart_error()
