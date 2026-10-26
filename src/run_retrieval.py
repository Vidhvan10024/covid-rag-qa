"""Experiment 1 — Retrieval effectiveness on TREC-COVID.

RQ1: Do biomedical dense encoders, hybrid fusion and cross-encoder re-ranking
     significantly outperform BM25 for COVID-19 literature retrieval?

Usage:
    python -m src.run_retrieval --config configs/default.yaml
    python -m src.run_retrieval --smoke            # offline, toy data, stub models
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import yaml
from ranx import Qrels, Run, evaluate

from . import data as D
from . import warn_if_gpu_busy
from .metrics import holm, randomization_test
from .retrievers import build_all, build_reranker, rrf_fuse


def judged_at_k(run, qrels, k=10):
    """Fraction of top-k documents that were judged — flags pool bias against new systems."""
    vals = []
    for q, docs in run.items():
        top = [d for d, _ in sorted(docs.items(), key=lambda x: -x[1])[:k]]
        vals.append(np.mean([d in qrels.get(q, {}) for d in top]) if top else 0.0)
    return float(np.mean(vals))


def capped_recall(run, qrels, k=100):
    """R_cap@k = |relevant in top-k| / min(k, |relevant|). BEIR reports this for TREC-COVID because
    topics have hundreds of relevant documents, which makes plain recall@100 look tiny."""
    vals = []
    for q, rel in qrels.items():
        relevant = {d for d, r in rel.items() if r > 0}
        if not relevant:
            continue
        top = [d for d, _ in sorted(run.get(q, {}).items(), key=lambda x: -x[1])[:k]]
        vals.append(len(relevant.intersection(top)) / min(k, len(relevant)))
    return float(np.mean(vals))


# Pre-specified retrieval hypotheses: (label, system A, system B, direction claimed)
HYPOTHESES = [("H1a: MedCPT > BGE", "bge", "medcpt"),
              ("H1b: RRF(all) > best single retriever (BGE)", "bge", "rrf(all)"),
              ("H1c: re-ranking improves RRF(all)", "rrf(all)", "rrf(all)>>CE")]


def hypothesis_tests(per_query, n_perm, seed, alpha):
    rows, raw = [], {}
    for label, a, b in HYPOTHESES:
        if a in per_query and b in per_query:
            d = float(np.mean(per_query[b]) - np.mean(per_query[a]))
            raw[label] = randomization_test(per_query[a], per_query[b], n=n_perm, seed=seed)
            rows.append({"hypothesis": label, "A": a, "B": b, "nDCG@10 A": round(float(np.mean(per_query[a])), 4),
                         "nDCG@10 B": round(float(np.mean(per_query[b])), 4), "Δ (B-A)": round(d, 4)})
    adj = holm(raw)
    for r in rows:
        p = adj[r["hypothesis"]]
        r["p (Holm)"] = round(p, 4)
        r["outcome"] = ("supported" if (p < alpha and r["Δ (B-A)"] > 0) else
                        "refuted (significant, opposite direction)" if (p < alpha and r["Δ (B-A)"] < 0) else
                        "not supported (no significant difference)")
    return pd.DataFrame(rows)


def condensed(run, qrels):
    """Judged-only ranking (Sakai, 2007): drop unjudged documents before scoring, so systems
    that surface relevant-but-never-judged papers are not penalised by the 2020 pools."""
    return {q: {d: s for d, s in docs.items() if d in qrels.get(q, {})} or {"__none__": 0.0}
            for q, docs in run.items()}


def pairwise_accuracy(run, qrels, candidates=None):
    """Fraction of judged document pairs (different relevance grades) that the run orders
    correctly. 0.5 = random; clearly below 0.5 = scores inverted (a bug signature)."""
    ok = tot = 0
    for q, docs in run.items():
        pool = candidates.get(q, docs) if candidates else docs
        judged = [(d, qrels[q][d], docs[d]) for d in pool if d in docs and d in qrels.get(q, {})]
        for i in range(len(judged)):
            for j in range(i + 1, len(judged)):
                (_, ri, si), (_, rj, sj) = judged[i], judged[j]
                if ri == rj:
                    continue
                tot += 1
                ok += (si > sj) == (ri > rj)
    return (ok / tot if tot else float("nan")), tot


def reranker_check(runs, reranked, qrels, depth):
    """Compare each re-ranked run with its first stage on the SAME top-`depth` candidates."""
    rows = []
    for rr_name, base_name in reranked.items():
        cand = {q: [d for d, _ in sorted(docs.items(), key=lambda x: -x[1])[:depth]]
                for q, docs in runs[base_name].items()}
        acc_b, n = pairwise_accuracy(runs[base_name], qrels, cand)
        acc_r, _ = pairwise_accuracy(runs[rr_name], qrels, cand)
        rows.append({"reranked": rr_name, "first_stage": base_name, "judged_pairs": n,
                     "pairwise_acc_first_stage": round(acc_b, 4), "pairwise_acc_reranker": round(acc_r, 4),
                     "judged@10_first_stage": round(judged_at_k(runs[base_name], qrels), 4),
                     "judged@10_reranker": round(judged_at_k(runs[rr_name], qrels), 4),
                     "verdict": ("SUSPECT: below chance, check score direction" if acc_r < 0.5 else
                                 "orders judged docs better than first stage" if acc_r > acc_b + 1e-9 else
                                 "orders judged docs the same as first stage" if abs(acc_r - acc_b) <= 1e-9 else
                                 "orders judged docs worse than first stage")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if not args.smoke:
        warn_if_gpu_busy()
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    out = os.path.join(cfg["output_dir"] + ("_smoke" if args.smoke else ""), "exp1_retrieval")
    os.makedirs(out, exist_ok=True)

    corpus, queries, qrels = D.toy_trec() if args.smoke else D.load_trec_covid(cfg["trec_covid"], cfg["cache_dir"])
    docs = [(d, v["title"], v["text"]) for d, v in corpus.items()]
    texts = {d: f"{t} {x}".strip() for d, t, x in docs}
    depth = cfg["retrievers"]["hybrid"]["depth"]
    rrf_k = cfg["retrievers"]["hybrid"]["rrf_k"]

    # ---- first-stage retrievers --------------------------------------------------
    retrievers = build_all(cfg, docs, args.smoke, cache_dir=os.path.join(cfg["cache_dir"], "trec_emb"))
    runs = {name: r.search(queries, depth) for name, r in retrievers.items()}

    # ---- hybrids (combinations fixed in advance; nothing is chosen on test data) ----
    dense = [n for n in runs if n != "bm25"]
    for n in dense:
        runs[f"rrf(bm25+{n})"] = rrf_fuse([runs["bm25"], runs[n]], rrf_k, depth)
    runs["rrf(all)"] = rrf_fuse([runs[n] for n in ["bm25"] + dense], rrf_k, depth)

    # ---- re-ranking --------------------------------------------------------------
    reranker = build_reranker(cfg, args.smoke)
    rcache = os.path.join(cfg["cache_dir"], "trec_rerank"); os.makedirs(rcache, exist_ok=True)
    tag = f"{len(corpus)}"
    runs["bm25>>CE"] = reranker.rerank(runs["bm25"], queries, texts,
                                       cache_path=os.path.join(rcache, f"bm25_{tag}.json"))
    runs["rrf(all)>>CE"] = reranker.rerank(runs["rrf(all)"], queries, texts,
                                           cache_path=os.path.join(rcache, f"rrf_all_{tag}.json"))

    # ---- evaluation --------------------------------------------------------------
    metrics = cfg["retrieval_eval"]["metrics"]
    qrels_r = Qrels.from_dict(qrels)
    rows, per_query = [], {}
    for name, rd in runs.items():
        run = Run.from_dict(rd, name=name)
        scores = evaluate(qrels_r, run, metrics)
        scores = scores if isinstance(scores, dict) else {metrics[0]: scores}
        per_query[name] = evaluate(qrels_r, run, "ndcg@10", return_mean=False)
        cond = evaluate(qrels_r, Run.from_dict(condensed(rd, qrels), name=name), "ndcg@10")
        rows.append({"system": name, **{m: round(float(v), 4) for m, v in scores.items()},
                     "R_cap@100": round(capped_recall(rd, qrels), 4),
                     "ndcg@10_judged_only": round(float(cond), 4),
                     "judged@10": round(judged_at_k(rd, qrels), 4)})
        run.save(os.path.join(out, f"run_{name.replace('>>', '_').replace('+', '_')}.trec"), kind="trec")
    table = pd.DataFrame(rows)

    # ---- significance vs BM25 (paired randomisation, Holm-corrected) -------------
    base = per_query["bm25"]
    n_perm = cfg["retrieval_eval"].get("permutations", 10000)
    raw_p = {n: randomization_test(base, v, n=n_perm, seed=cfg["seed"])
             for n, v in per_query.items() if n != "bm25"}
    adj_p = holm(raw_p)
    table["p_vs_bm25(ndcg@10)"] = table["system"].map(lambda n: round(adj_p.get(n, np.nan), 4))
    alpha = cfg["retrieval_eval"]["alpha"]
    table["sig"] = table["p_vs_bm25(ndcg@10)"].map(lambda p: "*" if p < alpha else "")

    table.to_csv(os.path.join(out, "table_retrieval.csv"), index=False)
    rchk = reranker_check(runs, {"bm25>>CE": "bm25", "rrf(all)>>CE": "rrf(all)"}, qrels,
                          cfg["retrievers"]["reranker"]["depth"])
    rchk.to_csv(os.path.join(out, "reranker_check.csv"), index=False)
    hyp = hypothesis_tests(per_query, n_perm, cfg["seed"], alpha)
    hyp.to_csv(os.path.join(out, "hypothesis_tests.csv"), index=False)
    pd.DataFrame(per_query).to_csv(os.path.join(out, "per_query_ndcg10.csv"))
    json.dump({"n_queries": len(queries), "n_docs": len(corpus)}, open(os.path.join(out, "meta.json"), "w"))
    print("\n=== Experiment 1: TREC-COVID retrieval ===")
    print(table.to_string(index=False))
    print(f"\n* = significantly different from BM25 (Holm-adjusted p < {alpha})")
    print("\n=== Pre-specified hypotheses H1a-H1c (paired randomisation, Holm) ===")
    print(hyp.to_string(index=False))
    print("\n=== Re-ranker check (same candidates, judged pairs only) ===")
    print(rchk.to_string(index=False))


if __name__ == "__main__":
    main()
