"""Figures 2-4 of the paper, drawn from saved results (no GPU, no downloads).

    python -m src.paper_figures --main results/results_full --second results/results_full_llama --out paper_figures

Writes PNG (600 dpi) and TIFF versions. The figures are designed to be readable in greyscale:
line styles, markers and hatching carry the distinctions, not colour.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import NAMES, PIPELINES, TREC_ROW  # noqa: E402

MM = 1 / 25.4
INK, MID, LIGHT = "#000000", "#555555", "#9a9a9a"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "savefig.dpi": 600, "savefig.bbox": "tight",
})


def _save(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.png"))
    fig.savefig(os.path.join(out, f"{name}.tif"), pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return os.path.join(out, f"{name}.png")


def fig2_ablation(main, out):
    ab = pd.read_csv(os.path.join(main, "exp3_ablation", "summary.csv"))
    fig, ax = plt.subplots(figsize=(84 * MM, 58 * MM))
    for (cw, g), ls, mk, col in zip(ab.groupby("chunk_words"), ["-", "--", ":"], ["o", "s", "^"], [INK, MID, LIGHT]):
        g = g.sort_values("k")
        ax.plot(g.k, g.f1, ls=ls, marker=mk, ms=4, lw=1.2, color=col, label=f"{cw}-word passages")
    ax.set_xticks([1, 3, 5, 10])
    ax.set_xlabel("Passages given to the reader (k)")
    ax.set_ylabel("Answer F1 (development set)")
    ax.set_ylim(0.25, 0.47)
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, out, "Fig2_ablation")


def fig3_rank_comparison(main, second, out):
    trec_t = pd.read_csv(os.path.join(main, "exp1_retrieval", "table_retrieval.csv")).set_index("system")
    trec = {p: trec_t.loc[TREC_ROW[p], "ndcg@10"] for p in PIPELINES}
    qs = pd.read_csv(os.path.join(main, "exp2_main", "summary.csv")).set_index("condition")
    cols = [("TREC-COVID\nnDCG@10", trec), ("COVID-QA\nhit@10", qs["hit@k"].to_dict()),
            ("COVID-QA F1\n(Qwen)", qs["f1"].to_dict())]
    if second:
        ls_ = pd.read_csv(os.path.join(second, "exp2_main", "summary.csv")).set_index("condition")
        cols.append(("COVID-QA F1\n(Llama)", ls_["f1"].to_dict()))
    ranks = []
    for _, d in cols:
        order = sorted(PIPELINES, key=lambda s: -d[s])
        ranks.append({s: order.index(s) + 1 for s in PIPELINES})
    fig, ax = plt.subplots(figsize=(174 * MM, 62 * MM))
    styles = {"bm25": ("-", "o", INK), "bge": ("--", "s", INK), "medcpt": (":", "^", MID),
              "hybrid": ("-.", "D", MID), "hybrid_rerank": ("-", "v", LIGHT)}
    for s in PIPELINES:
        ls, mk, col = styles[s]
        y = [r[s] for r in ranks]
        ax.plot(range(len(cols)), y, ls=ls, marker=mk, ms=5, lw=1.3, color=col)
        ax.text(-0.08, y[0], f"{NAMES[s]} ({trec[s]:.3f})", ha="right", va="center", fontsize=7.5)
        ax.text(len(cols) - 1 + 0.08, y[-1], f"{NAMES[s]} ({cols[-1][1][s]:.3f})", ha="left", va="center", fontsize=7.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols])
    ax.set_yticks([])
    ax.set_ylim(5.4, 0.6)
    ax.set_xlim(-1.0, len(cols))
    ax.spines["left"].set_visible(False)
    for x in range(len(cols)):
        ax.axvline(x, color="#dddddd", lw=0.5, zorder=0)
    return _save(fig, out, "Fig3_rank_comparison")


def fig4_retrieved_vs_missed(main, second, out):
    runs = [(main, "Qwen2.5-7B")] + ([(second, "Llama-3.1-8B")] if second else [])
    fig, axes = plt.subplots(1, len(runs), figsize=(174 * MM if second else 90 * MM, 62 * MM), sharey=True, squeeze=False)
    for ax, (run, title) in zip(axes[0], runs):
        t = pd.read_csv(os.path.join(run, "exp2_main", "transfer_vs_oracle.csv")).set_index("condition").loc[PIPELINES]
        eq_col = [c for c in t.columns if str(c).startswith("equivalent")][0]
        x, w = range(len(PIPELINES)), 0.26
        ax.bar([i - w for i in x], t.F1_hit, w, color="white", edgecolor=INK, lw=0.7, label="Retrieved: system")
        ax.bar(list(x), t.oracle_F1_same_q, w, color=LIGHT, edgecolor=INK, lw=0.7, label="Retrieved: oracle, same questions")
        ax.bar([i + w for i in x], t.F1_miss, w, color="white", edgecolor=INK, lw=0.7, hatch="////", label="Missed: system")
        for i, s in enumerate(PIPELINES):
            if not bool(t.loc[s, eq_col]):
                ax.text(i - w, t.F1_hit[s] + 0.03, "n.e.", ha="center", fontsize=6.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels([NAMES[s].replace(" + ", "\n+ ") for s in PIPELINES])
        ax.set_title(title)
        ax.set_ylim(0, 0.75)
        ax.grid(axis="y", lw=0.3, color="#dddddd")
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Answer F1 (test set)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.16))
    return _save(fig, out, "Fig4_retrieved_vs_missed")


def make_all(main, second=None, out="paper_figures"):
    os.makedirs(out, exist_ok=True)
    return [fig2_ablation(main, out), fig3_rank_comparison(main, second, out), fig4_retrieved_vs_missed(main, second, out)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--main", default="results/results_full", help="results folder of the primary reader")
    ap.add_argument("--second", default="results/results_full_llama", help="results folder of the second reader (optional)")
    ap.add_argument("--out", default="paper_figures")
    a = ap.parse_args()
    second = a.second if a.second and os.path.isdir(a.second) else None
    for p in make_all(a.main, second, a.out):
        print("wrote", p)
