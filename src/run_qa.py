"""Experiments 2 & 3 — End-to-end question answering on COVID-QA.

RQ2: Does retrieval-augmented generation improve answer accuracy and faithfulness
     over a closed-book LLM, and do retrieval gains (Exp. 1) carry over to answers?
RQ3: How sensitive is RAG to passage length and the number of passages k?

Protocol (leakage-safe):
  stage 'ablation' -> grid over chunk size x k on the DEV documents only; writes best.json
  stage 'main'     -> all conditions on the held-out TEST documents with the chosen setting

Usage:
    python -m src.run_qa --stage ablation
    python -m src.run_qa --stage main
    python -m src.run_qa --stage all --smoke
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re

import numpy as np
import pandas as pd
import yaml

from . import data as D
from .ckpt import load_json, save_json
from . import warn_if_gpu_busy
from .metrics import (NLIFaithfulness, StubFaithfulness, containment, exact_match, holm,
                      mcnemar, paired_bootstrap, token_f1)
from .reader import HFReader, StubReader, build_prompt, parse_answer
from .retrievers import build_all, build_reranker, rrf_fuse


FAITH_VERSION = "sentence-windows-cited-only-v2"  # bump when the scorer changes


def free_gpu():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# Retrieval over chunked COVID-QA articles
# ----------------------------------------------------------------------------
def retrieve_all(cfg, examples, articles, chunk_words, conditions, smoke, depth):
    passages = D.chunk_articles(articles, chunk_words, max(1, chunk_words // 2))
    pmap = {p.pid: p for p in passages}
    docs = [(p.pid, p.title, p.text) for p in passages]
    queries = {e.qid: e.question for e in examples}
    needed = set(conditions) - {"closed_book", "oracle"}
    runs = {}
    if needed:
        rets = build_all(cfg, docs, smoke, cache_dir=os.path.join(cfg["cache_dir"], f"qa_emb_{chunk_words}"))
        for name, r in rets.items():
            runs[name] = r.search(queries, depth)
        free_gpu()
        rrf_k = cfg["retrievers"]["hybrid"]["rrf_k"]
        runs["hybrid"] = rrf_fuse([runs[n] for n in rets], rrf_k, depth)
        if "hybrid_rerank" in needed:
            texts = {p.pid: p.text for p in passages}
            rpath = os.path.join(cfg["cache_dir"], f"qa_rerank_cw{chunk_words}_{len(examples)}q.json")
            os.makedirs(cfg["cache_dir"], exist_ok=True)
            runs["hybrid_rerank"] = build_reranker(cfg, smoke).rerank(runs["hybrid"], queries, texts,
                                                                       cache_path=rpath)
            free_gpu()
    return passages, pmap, runs


def contexts_for(cond, ex, runs, passages, pmap, k):
    """Return the list of passage ids given to the reader for one question."""
    if cond == "closed_book":
        return []
    if cond == "oracle":
        gold = [p.pid for p in passages if p.doc_id == ex.doc_id and D.answer_bearing(p, ex)]
        return gold[:1]
    ranked = sorted(runs[cond][ex.qid].items(), key=lambda x: -x[1])
    return [pid for pid, _ in ranked[:k]]


# ----------------------------------------------------------------------------
# Scoring one condition
# ----------------------------------------------------------------------------
def score(cond, examples, ctx_ids, raw_outputs, pmap, faith, faith_cache=None):
    rows = []
    for ex, ids, raw in zip(examples, ctx_ids, raw_outputs):
        ans, cites, abst = parse_answer(raw)
        gold_flags = [D.answer_bearing(pmap[i], ex) for i in ids]
        valid_cites = [c for c in cites if 1 <= c <= len(ids)]
        rows.append({
            "qid": ex.qid, "condition": cond, "question": ex.question, "gold": ex.answers[0],
            "raw": raw, "pred": ans, "abstained": float(abst),
            "em": 0.0 if abst else exact_match(ans, ex.answers),
            "f1": 0.0 if abst else token_f1(ans, ex.answers),
            "contains": 0.0 if abst else containment(ans, ex.answers),
            "hit@k": float(any(gold_flags)) if ids else np.nan,
            "rr": next((1.0 / (r + 1) for r, g in enumerate(gold_flags) if g), 0.0) if ids else np.nan,
            "cite_precision": (np.mean([gold_flags[c - 1] for c in valid_cites]) if valid_cites else np.nan),
            "cited": (float(bool(valid_cites)) if (ids and not abst) else np.nan),
            "ctx": ids, "cited_ids": [ids[c - 1] for c in dict.fromkeys(valid_cites)],
        })
    if faith is not None:
        # Attribution is scored on the passages the answer cites, and only for answers that cite.
        # Whether an answer cites at all is reported separately ("cited"): mixing the two made the
        # oracle (one passage, so the reader often skips the "[1]") look unfaithful in the pilot.
        ev = [[pmap[i].text for i in r["cited_ids"]] for r in rows]
        qs, ps = [r["question"] for r in rows], [r["pred"] for r in rows]
        key = hashlib.md5(json.dumps([qs, ps, ev, type(faith).__name__, FAITH_VERSION]).encode()).hexdigest()
        fpath = os.path.join(faith_cache, f"faith_{key}.json") if faith_cache else None
        cached = load_json(fpath) if fpath else None
        if cached is not None:
            fs = cached
        else:
            fs = faith.score(qs, ps, ev)
            if fpath:
                save_json(fs, fpath)
        for r, f in zip(rows, fs):
            ok = r["ctx"] and not r["abstained"] and r["cited_ids"]
            r["faithfulness"] = f if ok else np.nan
    return rows


def summarise(df: pd.DataFrame, keys):
    cols = ["em", "f1", "contains", "abstained", "hit@k", "rr", "cited", "cite_precision", "faithfulness"]
    cols = [c for c in cols if c in df]
    g = df.groupby(keys)[cols].mean().reset_index()
    g["n"] = df.groupby(keys).size().values
    return g.round(4)


# ----------------------------------------------------------------------------
def run_stage(cfg, stage, smoke):
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    qcfg = cfg["qa"]
    root = cfg["output_dir"] + ("_smoke" if smoke else "")
    examples, articles = D.toy_qa() if smoke else D.load_covid_qa(cfg["covid_qa"])
    dev, test = (examples, examples) if smoke else D.grouped_split(examples, cfg["covid_qa"]["dev_fraction"], cfg["seed"])
    if qcfg.get("max_questions"):
        rng = random.Random(cfg["seed"])
        test = rng.sample(test, min(len(test), qcfg["max_questions"]))
        dev = rng.sample(dev, min(len(dev), max(50, qcfg["max_questions"] // 4)))
    depth = max(max(qcfg["ablation_k"]), cfg["retrievers"]["reranker"]["depth"])

    if stage == "ablation":
        out = os.path.join(root, "exp3_ablation"); os.makedirs(out, exist_ok=True)
        jobs = []  # (chunk_words, k, examples, ctx_ids, pmap)
        for cw in qcfg["ablation_chunk_words"]:
            passages, pmap, runs = retrieve_all(cfg, dev, articles, cw, ["hybrid"], smoke, depth)
            for k in qcfg["ablation_k"]:
                jobs.append((cw, k, [contexts_for("hybrid", e, runs, passages, pmap, k) for e in dev], pmap))
        conds, main_ex = [f"hybrid|cw={cw}|k={k}" for cw, k, _, _ in jobs], dev
    else:
        out = os.path.join(root, "exp2_main"); os.makedirs(out, exist_ok=True)
        best_path = os.path.join(root, "exp3_ablation", "best.json")
        best = json.load(open(best_path)) if os.path.exists(best_path) else \
            {"chunk_words": cfg["covid_qa"]["chunk_words"], "k": qcfg["top_k"]}
        print(f"[main] using chunk_words={best['chunk_words']}, k={best['k']}")
        passages, pmap, runs = retrieve_all(cfg, test, articles, best["chunk_words"], qcfg["conditions"], smoke, depth)
        jobs = [(best["chunk_words"], best["k"],
                 [contexts_for(c, e, runs, passages, pmap, best["k"]) for e in test], pmap)
                for c in qcfg["conditions"]]
        conds, main_ex = qcfg["conditions"], test

    # ---- generation (reader loaded once, after all retrieval is done) -----------
    reader, reader_name = None, ("stub" if smoke else qcfg["reader_model"])
    # Each condition's answers are saved as soon as they finish, so a disconnected
    # session resumes where it stopped instead of starting over.
    ckpt_dir = os.path.join(out, "generations"); os.makedirs(ckpt_dir, exist_ok=True)
    qids = [e.qid for e in main_ex]
    raw = {}
    save_every = qcfg.get("save_every", 64)
    for cond, (cw, k, ctx, pmap) in zip(conds, jobs):
        ck = os.path.join(ckpt_dir, re.sub(r"[^A-Za-z0-9_.=-]", "_", f"{cond}__cw{cw}_k{k}") + ".json")
        prompts = [build_prompt(e.question, [pmap[i].text for i in ids]) for e, ids in zip(main_ex, ctx)]
        # A checkpoint is reused only if it was made from exactly these prompts (same questions,
        # same retrieved passages, same prompt wording) and the same reader model.
        pk = hashlib.md5(json.dumps(prompts).encode()).hexdigest()
        valid = lambda c: bool(c) and c.get("qids") == qids and c.get("reader") == reader_name and c.get("prompts_md5") == pk
        saved = load_json(ck)
        if valid(saved):
            raw[cond] = saved["outputs"]
            print(f"[gen] {cond}: resumed from checkpoint")
            continue
        if saved:
            print(f"[gen] {cond}: saved answers were made from different passages or settings -> regenerating")
        # partial progress: answers are saved every `save_every` questions
        part = load_json(ck + ".partial")
        done = part["outputs"] if valid(part) else []
        if done:
            print(f"[gen] {cond}: resuming at question {len(done) + 1} of {len(prompts)}")
        for s0 in range(len(done), len(prompts), save_every):
            if reader is None:
                reader = StubReader() if smoke else HFReader(qcfg["reader_model"], qcfg["load_in_4bit"],
                                                             qcfg["max_new_tokens"], qcfg["batch_size"], cfg["seed"])
            done.extend(reader.generate(prompts[s0:s0 + save_every]))
            save_json({"qids": qids, "reader": reader_name, "prompts_md5": pk, "outputs": done}, ck + ".partial")
            print(f"[gen] {cond}: {len(done)}/{len(prompts)} answers saved")
        raw[cond] = done
        save_json({"qids": qids, "reader": reader_name, "prompts_md5": pk, "outputs": done}, ck)
        try:
            os.remove(ck + ".partial")
        except OSError:
            pass
        print(f"[gen] {cond}: complete")
    del reader; free_gpu()

    # ---- scoring ------------------------------------------------------------------
    faith = StubFaithfulness() if smoke else NLIFaithfulness(qcfg["nli_model"])
    rows = []
    for cond, (cw, k, ctx, pmap) in zip(conds, jobs):
        for r in score(cond, main_ex, ctx, raw[cond], pmap, faith, faith_cache=ckpt_dir):
            r.update({"chunk_words": cw, "k": k}); rows.append(r)
    df = pd.DataFrame(rows)
    df.to_json(os.path.join(out, "predictions.jsonl"), orient="records", lines=True)
    summ = summarise(df, ["condition"] if stage == "main" else ["chunk_words", "k"])
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)

    if stage == "ablation":
        top = summ.sort_values(["f1", "k"], ascending=[False, True]).iloc[0]
        json.dump({"chunk_words": int(top["chunk_words"]), "k": int(top["k"])}, open(os.path.join(out, "best.json"), "w"))
        print("\n=== Experiment 3: ablation on DEV ===\n" + summ.to_string(index=False))
        print(f"selected: chunk_words={int(top['chunk_words'])}, k={int(top['k'])}")
        return

    # ---- significance: every condition vs RAG-BM25 baseline, Holm-corrected ------
    base = "bm25" if "bm25" in conds else conds[0]
    piv = {m: df.pivot(index="qid", columns="condition", values=m) for m in ["f1", "contains", "faithfulness"]}
    sig, pf1, pc = [], {}, {}
    for c in conds:
        if c == base:
            continue
        bs = paired_bootstrap(piv["f1"][base], piv["f1"][c], qcfg["bootstrap_samples"], cfg["seed"])
        pf1[c], pc[c] = bs["p"], mcnemar(piv["contains"][base], piv["contains"][c])
        sig.append({"condition": c, "vs": base, "ΔF1": round(bs["diff"], 4),
                    "F1 95% CI": f"[{bs['ci_low']:.3f}, {bs['ci_high']:.3f}]"})
    af, ac = holm(pf1), holm(pc)
    for s in sig:
        s["p_F1 (Holm)"] = round(af[s["condition"]], 4)
        s["p_contains McNemar (Holm)"] = round(ac[s["condition"]], 4)
    sig = pd.DataFrame(sig)
    sig.to_csv(os.path.join(out, "significance.csv"), index=False)
    print("\n=== Experiment 2: COVID-QA end-to-end (TEST) ===\n" + summ.to_string(index=False))
    print("\n" + sig.to_string(index=False))

    rag = [c for c in conds if c not in ("closed_book", "oracle")]
    if "oracle" in conds:
        transfer_vs_oracle(df, rag, qcfg, cfg["seed"], out)
        faithfulness_sanity(summ, rag, out)


def transfer_vs_oracle(df, rag, qcfg, seed, out):
    """Is the remaining error a reading failure? On the questions where condition c DID retrieve
    an answer-bearing passage, compare c's F1 with the oracle's F1 on the SAME questions.
    Equivalence = two one-sided tests (TOST): 90% bootstrap CI of the difference inside
    +/- the margin fixed before the full run. Non-significance alone is NOT evidence of equivalence."""
    margin = qcfg.get("equivalence_margin_f1", 0.05)
    f1 = df.pivot(index="qid", columns="condition", values="f1")
    hit = df.pivot(index="qid", columns="condition", values="hit@k")
    rows, pv = [], {}
    for c in rag:
        q_hit = hit.index[hit[c] == 1.0]
        q_miss = hit.index[hit[c] == 0.0]
        if len(q_hit) < 2:
            continue
        bs = paired_bootstrap(f1.loc[q_hit, "oracle"], f1.loc[q_hit, c], qcfg["bootstrap_samples"], seed)
        pv[c] = bs["p"]
        rows.append({"condition": c, "n_hit": len(q_hit), "F1_hit": round(f1.loc[q_hit, c].mean(), 4),
                     "oracle_F1_same_q": round(f1.loc[q_hit, "oracle"].mean(), 4),
                     "Δ(c-oracle)": round(bs["diff"], 4),
                     "95% CI": f"[{bs['ci_low']:.3f}, {bs['ci_high']:.3f}]",
                     "90% CI (TOST)": f"[{bs['ci90_low']:.3f}, {bs['ci90_high']:.3f}]",
                     f"equivalent (±{margin})": bool(-margin < bs["ci90_low"] and bs["ci90_high"] < margin),
                     "n_miss": len(q_miss), "F1_miss": round(f1.loc[q_miss, c].mean(), 4) if len(q_miss) else np.nan})
    if not rows:
        return
    adj = holm(pv)
    for r in rows:
        r["p_diff (Holm)"] = round(adj[r["condition"]], 4)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "transfer_vs_oracle.csv"), index=False)
    print("\n=== Retrieval-to-answer transfer: condition vs oracle on the same questions ===")
    print(t.to_string(index=False))


def faithfulness_sanity(summ, rag, out):
    """The oracle sees the gold passage by construction, so a valid attribution metric should
    not score any RAG condition above it (among answers that cite). Prints PASS / FAIL."""
    if "faithfulness" not in summ:
        return
    f = summ.set_index("condition")["faithfulness"]
    if "oracle" not in f or pd.isna(f["oracle"]):
        return
    above = {c: float(f[c]) for c in rag if c in f and f[c] > f["oracle"] + 1e-9}
    res = {"oracle_faithfulness": float(f["oracle"]), "rag_conditions_above_oracle": above,
           "passed": not above}
    json.dump(res, open(os.path.join(out, "sanity_faithfulness.json"), "w"), indent=2)
    if above:
        print(f"\n[sanity] FAIL: faithfulness of {list(above)} exceeds the oracle ({f['oracle']:.4f}). "
              "Do not report the automatic faithfulness metric; use the human validation instead.")
    else:
        print(f"\n[sanity] PASS: oracle faithfulness ({f['oracle']:.4f}) >= every RAG condition.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stage", choices=["ablation", "main", "all"], default="all")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if not a.smoke:
        warn_if_gpu_busy()
    for st in (["ablation", "main"] if a.stage == "all" else [a.stage]):
        run_stage(cfg, st, a.smoke)


if __name__ == "__main__":
    main()
