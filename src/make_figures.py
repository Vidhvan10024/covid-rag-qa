"""Builds the paper's tables (Markdown + LaTeX) and 300-dpi figures from the results folders.

Usage:  python -m src.make_figures [--root results]
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

# Validated palette (colour-blind-safe categorical pair; blue ramp for the ordered chunk sizes)
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]      # light -> dark = short -> long passages
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def _style(plt):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def fig_retrieval(root, plt):
    p = os.path.join(root, "exp1_retrieval", "table_retrieval.csv")
    if not os.path.exists(p):
        return
    t = pd.read_csv(p).sort_values("ndcg@10")
    t.to_markdown(os.path.join(root, "table1_retrieval.md"), index=False)
    t.to_latex(os.path.join(root, "table1_retrieval.tex"), index=False, float_format="%.4f")
    rc = os.path.join(root, "exp1_retrieval", "reranker_check.csv")
    if os.path.exists(rc):
        pd.read_csv(rc).to_markdown(os.path.join(root, "table1b_reranker_check.md"), index=False)
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(t) + 1))
    ax.barh(t["system"], t["ndcg@10"], color=BLUE, height=0.6)
    for y, (v, s) in enumerate(zip(t["ndcg@10"], t["sig"].fillna(""))):
        ax.text(v + 0.005, y, f"{v:.3f}{s}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("nDCG@10 on TREC-COVID (* = p < .05 vs BM25, Holm)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, min(1.0, t["ndcg@10"].max() * 1.18))
    fig.savefig(os.path.join(root, "fig1_retrieval.png")); plt.close(fig)


def fig_ablation(root, plt):
    p = os.path.join(root, "exp3_ablation", "summary.csv")
    if not os.path.exists(p):
        return
    s = pd.read_csv(p)
    s.to_markdown(os.path.join(root, "table3_ablation.md"), index=False)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for (cw, g), c, m in zip(s.groupby("chunk_words"), RAMP, ["o", "s", "^"]):
        g = g.sort_values("k")
        ax.plot(g["k"], g["f1"], color=c, lw=2, marker=m, ms=6, label=f"{cw} words")
    ax.set_xlabel("Passages given to the reader (k)")
    ax.set_ylabel("Token F1 (dev)")
    ax.set_xticks(sorted(s["k"].unique()))
    ax.legend(title="Passage length", frameon=False, fontsize=8, title_fontsize=8)
    fig.savefig(os.path.join(root, "fig3_ablation.png")); plt.close(fig)


def fig_qa(root, plt):
    p = os.path.join(root, "exp2_main", "summary.csv")
    if not os.path.exists(p):
        return
    s = pd.read_csv(p)
    order = ["closed_book", "bm25", "bge", "medcpt", "hybrid", "hybrid_rerank", "oracle"]
    s["_o"] = s["condition"].map({c: i for i, c in enumerate(order)}).fillna(99)
    s = s.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    s.to_markdown(os.path.join(root, "table2_qa.md"), index=False)
    s.to_latex(os.path.join(root, "table2_qa.tex"), index=False, float_format="%.3f")
    sig = os.path.join(root, "exp2_main", "significance.csv")
    if os.path.exists(sig):
        pd.read_csv(sig).to_markdown(os.path.join(root, "table2b_significance.md"), index=False)

    # Retrieval -> answer transfer: F1 when the gold passage was / was not retrieved
    pr = pd.read_json(os.path.join(root, "exp2_main", "predictions.jsonl"), lines=True)
    tr = pr.dropna(subset=["hit@k"]).groupby(["condition", "hit@k"])["f1"].agg(["mean", "count"]).round(4)
    tr.to_markdown(os.path.join(root, "table4_transfer.md"))
    tv = os.path.join(root, "exp2_main", "transfer_vs_oracle.csv")
    if os.path.exists(tv):
        pd.read_csv(tv).to_markdown(os.path.join(root, "table4b_transfer_vs_oracle.md"), index=False)

    import numpy as np
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, s["f1"], w, color=BLUE, label="Token F1")
    if "faithfulness" in s:
        ax.bar(x + w / 2 + 0.01, s["faithfulness"].fillna(0), w, color=ORANGE, label="Attribution (NLI, answers that cite)")
    ax.set_xticks(x, s["condition"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (COVID-QA test)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.savefig(os.path.join(root, "fig2_qa.png")); plt.close(fig)


# TREC-COVID run name  ->  COVID-QA condition name (same retrieval pipeline on both benchmarks)
SAME_PIPELINE = {"bm25": "bm25", "bge": "bge", "medcpt": "medcpt",
                 "rrf(all)": "hybrid", "rrf(all)>>CE": "hybrid_rerank"}


def table_rank_agreement(root):
    """Does retrieval quality predict answer quality? Kendall's tau between system rankings,
    (a) within COVID-QA (passage hit@k vs answer F1) and (b) across benchmarks
    (TREC-COVID nDCG@10 vs COVID-QA). With 5 systems, tau is descriptive; report it as such."""
    from scipy.stats import kendalltau
    t1 = os.path.join(root, "exp1_retrieval", "table_retrieval.csv")
    t2 = os.path.join(root, "exp2_main", "summary.csv")
    if not (os.path.exists(t1) and os.path.exists(t2)):
        return
    trec = pd.read_csv(t1).set_index("system")
    qa = pd.read_csv(t2).set_index("condition")
    rows = [{"pipeline": qa_name, "TREC nDCG@10": trec.loc[tr, "ndcg@10"],
             "TREC nDCG@10 (judged only)": trec.loc[tr].get("ndcg@10_judged_only", float("nan")),
             "COVID-QA hit@k": qa.loc[qa_name, "hit@k"], "COVID-QA MRR": qa.loc[qa_name, "rr"],
             "COVID-QA F1": qa.loc[qa_name, "f1"]}
            for tr, qa_name in SAME_PIPELINE.items() if tr in trec.index and qa_name in qa.index]
    if len(rows) < 3:
        return
    t = pd.DataFrame(rows).dropna(axis=1, how="all")
    for c in list(t.columns[1:]):
        t[c + " rank"] = t[c].rank(ascending=False, method="min").astype("Int64")
    pairs = [("COVID-QA hit@k", "COVID-QA F1"), ("TREC nDCG@10", "COVID-QA F1"),
             ("TREC nDCG@10", "COVID-QA hit@k"), ("TREC nDCG@10 (judged only)", "COVID-QA F1")]
    taus = []
    for a, b in pairs:
        if a in t and b in t and t[a].notna().all():
            r = kendalltau(t[a], t[b])
            taus.append({"ranking A": a, "ranking B": b, "Kendall tau": round(r.statistic, 3),
                         "p (exact)": round(r.pvalue, 4), "n systems": len(t)})
    with open(os.path.join(root, "table5_rank_agreement.md"), "w") as f:
        f.write(t.round(4).to_markdown(index=False) + "\n\n" + pd.DataFrame(taus).to_markdown(index=False) + "\n")
    t.to_csv(os.path.join(root, "rank_agreement.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    root = ap.parse_args().root
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig_retrieval(root, plt); fig_ablation(root, plt); fig_qa(root, plt)
    table_rank_agreement(root)
    print(f"[figures] written to {root}/")


if __name__ == "__main__":
    main()
