"""Analyses reported in the paper that are computed from saved outputs and annotations, not from new runs.

Everything here runs on a CPU in a few seconds:
  - rank agreement between benchmarks (Kendall's tau)            -> Section 5.4, Fig. 3
  - human validation of answer attribution (Cohen's kappa, AUC)  -> Section 5.5, Table 5
  - error analysis category shares with Wilson intervals         -> Section 5.6, Table 6

Bootstrap intervals use 10,000 resamples with seed 42, the same convention as the rest of the code.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score, roc_auc_score

PIPELINES = ["bm25", "bge", "medcpt", "hybrid", "hybrid_rerank"]
CONDITIONS = ["oracle"] + PIPELINES
NAMES = {"oracle": "Oracle", "bm25": "BM25", "bge": "BGE", "medcpt": "MedCPT",
         "hybrid": "Hybrid", "hybrid_rerank": "Hybrid + rerank", "closed_book": "Closed-book"}
# TREC-COVID rows that correspond to the five QA pipelines ("hybrid" is RRF of all three retrievers)
TREC_ROW = {"bm25": "bm25", "bge": "bge", "medcpt": "medcpt", "hybrid": "rrf(all)", "hybrid_rerank": "rrf(all)>>CE"}
ERROR_CATEGORIES = ["Wrongly declined", "Retrieval miss", "Extraction error", "Wrong passage used",
                    "Correct but scored wrong", "Gold answer problem", "Other"]


# ----------------------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def kappa_with_ci(a, b, n_boot: int = 10_000, seed: int = 42) -> tuple[float, float, float]:
    """Cohen's kappa with a percentile bootstrap 95% interval (items resampled with replacement)."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i])) < 2 and len(set(b[i])) < 2:
            continue  # kappa is undefined when both raters use a single label
        boots.append(cohen_kappa_score(a[i], b[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return cohen_kappa_score(a, b), lo, hi


# ----------------------------------------------------------------------------- Section 5.4
def rank_agreement(results_main: str, results_second: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores and ranks of the five pipelines on both benchmarks, and Kendall's tau between the rankings."""
    trec = pd.read_csv(os.path.join(results_main, "exp1_retrieval", "table_retrieval.csv")).set_index("system")
    qa = pd.read_csv(os.path.join(results_main, "exp2_main", "summary.csv")).set_index("condition")
    cols = {
        "TREC nDCG@10": [trec.loc[TREC_ROW[p], "ndcg@10"] for p in PIPELINES],
        "TREC nDCG@10 (judged only)": [trec.loc[TREC_ROW[p], "ndcg@10_judged_only"] for p in PIPELINES],
        "COVID-QA hit@10": [qa.loc[p, "hit@k"] for p in PIPELINES],
        "F1 (primary reader)": [qa.loc[p, "f1"] for p in PIPELINES],
    }
    if results_second:
        qa2 = pd.read_csv(os.path.join(results_second, "exp2_main", "summary.csv")).set_index("condition")
        cols["F1 (second reader)"] = [qa2.loc[p, "f1"] for p in PIPELINES]
    scores = pd.DataFrame(cols, index=[NAMES[p] for p in PIPELINES])
    pairs = [("COVID-QA hit@10", "F1 (primary reader)"), ("TREC nDCG@10", "F1 (primary reader)"),
             ("TREC nDCG@10", "COVID-QA hit@10"), ("TREC nDCG@10 (judged only)", "F1 (primary reader)")]
    if results_second:
        pairs += [("COVID-QA hit@10", "F1 (second reader)"), ("TREC nDCG@10", "F1 (second reader)"),
                  ("F1 (primary reader)", "F1 (second reader)")]
    rows = []
    for x, y in pairs:
        tau, p = kendalltau(scores[x], scores[y])
        rows.append({"ranking A": x, "ranking B": y, "Kendall tau": round(tau, 2), "p (exact)": round(p, 3)})
    return scores.round(4), pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Section 5.5
def human_validation(path: str, threshold: float = 0.5) -> dict:
    """Agreement between the two annotators and validity of the automatic NLI attribution score."""
    df = pd.read_csv(path)
    a = (df.label_A == "Yes").astype(int).values
    b = (df.label_B == "Yes").astype(int).values
    final = (df.final_label == "Yes").astype(int).values
    nli = df.nli_score.values
    pred = (nli >= threshold).astype(int)

    k_ab, lo_ab, hi_ab = kappa_with_ci(a, b)
    k_nli, lo_nli, hi_nli = kappa_with_ci(final, pred)
    per_condition = (
        df.assign(A=a, B=b, final=final)
        .groupby("condition")[["final", "A", "B"]]
        .agg(["sum", "mean"])
    )
    table = pd.DataFrame({
        "supported (final)": [f"{int(per_condition.loc[c, ('final', 'sum')])}/{int((df.condition == c).sum())}"
                              f" ({per_condition.loc[c, ('final', 'mean')]:.2f})" for c in CONDITIONS],
        "annotator A": [round(per_condition.loc[c, ("A", "mean")], 2) for c in CONDITIONS],
        "annotator B": [round(per_condition.loc[c, ("B", "mean")], 2) for c in CONDITIONS],
    }, index=[NAMES[c] for c in CONDITIONS])
    table.loc["All sampled answers"] = [f"{final.sum()}/{len(df)} ({final.mean():.2f})", round(a.mean(), 2), round(b.mean(), 2)]
    return {
        "n_items": len(df),
        "annotators_agree": int((a == b).sum()),
        "kappa_annotators": (round(float(k_ab), 2), round(float(lo_ab), 2), round(float(hi_ab), 2)),
        "supported_final": int(final.sum()),
        "nli_agreement": round(float((pred == final).mean()), 3),
        "kappa_nli": (round(float(k_nli), 2), round(float(lo_nli), 2), round(float(hi_nli), 2)),
        "nli_auc": round(float(roc_auc_score(final, nli)), 2),
        "nli_mean_supported": round(float(nli[final == 1].mean()), 3),
        "nli_mean_unsupported": round(float(nli[final == 0].mean()), 3),
        "supported_but_below_threshold": int(((final == 1) & (pred == 0)).sum()),
        "unsupported_but_above_threshold": int(((final == 0) & (pred == 1)).sum()),
        "table": table,
    }


# ----------------------------------------------------------------------------- Section 5.6
def error_table(best_path: str, bge_path: str) -> pd.DataFrame:
    """Counts and Wilson 95% intervals of the error categories (Table 6)."""
    best, bge = pd.read_csv(best_path), pd.read_csv(bge_path)

    def cell(df, cat):
        k, n = int((df.category == cat).sum()), len(df)
        if k == 0:
            return "0"
        lo, hi = wilson(k, n)
        return f"{k} ({k / n:.0%}) [{lo:.0%}–{hi:.0%}]"

    rows = {cat: [cell(best, cat), cell(bge, cat)] for cat in ERROR_CATEGORIES}
    k = int((bge.distractor_present == "Yes").sum())
    rows["Distractor passage present"] = ["–", f"{k} ({k / len(bge):.0%})"]
    return pd.DataFrame(rows, index=[f"Best pipeline (hybrid + rerank), n = {len(best)}",
                                     f"BGE with answer retrieved, oracle correct, n = {len(bge)}"]).T
