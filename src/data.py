"""Data loading, passage chunking and leakage-safe splitting.

TREC-COVID  : BEIR release (171,332 CORD-19 abstracts, 50 topics, graded qrels)
COVID-QA    : Möller et al. (2020) — 2,019 expert QA pairs over 147 CORD-19 papers
"""
from __future__ import annotations

import io
import json
import os
import random
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ----------------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------------
@dataclass
class Passage:
    pid: str
    doc_id: str
    title: str
    text: str
    char_start: int = 0
    char_end: int = 0


@dataclass
class QAExample:
    qid: str
    question: str
    answers: List[str]
    answer_starts: List[int]
    doc_id: str
    is_impossible: bool = False
    meta: Dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# TREC-COVID (BEIR)
# ----------------------------------------------------------------------------
def load_trec_covid(cfg: dict, cache_dir: str):
    """Returns corpus {doc_id: {title, text}}, queries {qid: text}, qrels {qid: {doc_id: rel}}."""
    root = os.path.join(cache_dir, "trec-covid")
    if not os.path.exists(os.path.join(root, "qrels", "test.tsv")):
        os.makedirs(cache_dir, exist_ok=True)
        try:
            print(f"[data] downloading {cfg['beir_url']}")
            with urllib.request.urlopen(cfg["beir_url"], timeout=120) as r:
                zipfile.ZipFile(io.BytesIO(r.read())).extractall(cache_dir)
        except Exception as e:  # mirror unavailable -> identical copy on Hugging Face
            print(f"[data] BEIR mirror failed ({e}); using the Hugging Face copy instead")
            _trec_from_hf(root)

    corpus = {}
    with open(os.path.join(root, "corpus.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if cfg.get("max_docs") and i >= cfg["max_docs"]:
                break
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d.get("title", ""), "text": d.get("text", "")}

    queries = {}
    with open(os.path.join(root, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    qrels: Dict[str, Dict[str, int]] = {}
    with open(os.path.join(root, "qrels", "test.tsv"), encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            qid, did, rel = line.rstrip("\n").split("\t")
            if did in corpus:  # respect max_docs subsampling
                qrels.setdefault(qid, {})[did] = max(0, int(rel))  # -1 ("excluded") -> 0
    queries = {q: t for q, t in queries.items() if q in qrels}
    print(f"[data] TREC-COVID: {len(corpus):,} docs, {len(queries)} queries")
    return corpus, queries, qrels


def _trec_from_hf(root: str):
    """Writes BEIR-format files from the Hugging Face copies of TREC-COVID."""
    from datasets import load_dataset
    os.makedirs(os.path.join(root, "qrels"), exist_ok=True)

    def first_split(ds):
        return ds[next(iter(ds))]

    with open(os.path.join(root, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for r in first_split(load_dataset("BeIR/trec-covid", "corpus")):
            f.write(json.dumps({"_id": str(r["_id"]), "title": r["title"], "text": r["text"]}) + "\n")
    with open(os.path.join(root, "queries.jsonl"), "w", encoding="utf-8") as f:
        for r in first_split(load_dataset("BeIR/trec-covid", "queries")):
            f.write(json.dumps({"_id": str(r["_id"]), "text": r["text"]}) + "\n")
    with open(os.path.join(root, "qrels", "test.tsv"), "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for r in load_dataset("BeIR/trec-covid-qrels", split="test"):
            f.write(f"{r['query-id']}\t{r['corpus-id']}\t{r['score']}\n")


# ----------------------------------------------------------------------------
# COVID-QA
# ----------------------------------------------------------------------------
def load_covid_qa(cfg: dict):
    """Returns (examples, articles {doc_id: full_text})."""
    from datasets import load_dataset

    ds = load_dataset(cfg["hf_id"], split="train")
    examples, articles = [], {}
    for i, row in enumerate(ds):
        ans = row["answers"] if "answers" in row else row["answer"]
        texts = ans["text"] if isinstance(ans["text"], list) else [ans["text"]]
        starts = ans["answer_start"] if isinstance(ans["answer_start"], list) else [ans["answer_start"]]
        doc_id = str(row["document_id"])
        articles.setdefault(doc_id, row["context"])
        examples.append(QAExample(
            qid=f"q{i}", question=row["question"].strip(),
            answers=[t.strip() for t in texts], answer_starts=list(starts),
            doc_id=doc_id, is_impossible=bool(row.get("is_impossible", False)),
        ))
    examples = [e for e in examples if not e.is_impossible and e.answers and e.answers[0]]
    print(f"[data] COVID-QA: {len(examples):,} answerable questions, {len(articles)} articles")
    return examples, articles


def chunk_articles(articles: Dict[str, str], chunk_words: int, stride: int) -> List[Passage]:
    """Sliding-window word chunking that keeps character offsets (needed for the oracle condition)."""
    passages = []
    for doc_id, text in articles.items():
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        if not spans:
            continue
        title = text[: spans[min(len(spans), 20) - 1][1]].split("\n")[0][:200]
        for j, s in enumerate(range(0, len(spans), stride)):
            w = spans[s: s + chunk_words]
            cs, ce = w[0][0], w[-1][1]
            passages.append(Passage(f"{doc_id}#{j}", doc_id, title, text[cs:ce], cs, ce))
            if s + chunk_words >= len(spans):
                break
    return passages


def answer_bearing(p: Passage, ex: QAExample) -> bool:
    """A passage is 'gold' if it comes from the source article AND covers a gold answer span."""
    if p.doc_id != ex.doc_id:
        return False
    for a, st in zip(ex.answers, ex.answer_starts):
        if st >= 0 and p.char_start <= st and st + len(a) <= p.char_end:
            return True
    return any(a.lower() in p.text.lower() for a in ex.answers)


def grouped_split(examples: List[QAExample], dev_fraction: float, seed: int):
    """Document-grouped dev/test split: no article appears in both (prevents leakage)."""
    docs = sorted({e.doc_id for e in examples})
    random.Random(seed).shuffle(docs)
    dev_docs = set(docs[: max(1, int(len(docs) * dev_fraction))])
    dev = [e for e in examples if e.doc_id in dev_docs]
    test = [e for e in examples if e.doc_id not in dev_docs]
    return dev, test


# ----------------------------------------------------------------------------
# Tiny synthetic data for --smoke runs (no downloads, no GPU)
# ----------------------------------------------------------------------------
_TOY = [
    ("d1", "Incubation period of SARS-CoV-2",
     "The median incubation period of SARS-CoV-2 was estimated at 5.1 days, and 97.5% of patients develop symptoms within 11.5 days."),
    ("d2", "ACE2 receptor binding",
     "SARS-CoV-2 enters human cells by binding its spike protein to the angiotensin-converting enzyme 2 (ACE2) receptor."),
    ("d3", "Transmission through aerosols",
     "Airborne transmission via aerosols contributes to the spread of COVID-19 in poorly ventilated indoor spaces."),
    ("d4", "Remdesivir clinical trial",
     "Remdesivir shortened the time to recovery in adults hospitalized with COVID-19 in a randomized controlled trial."),
    ("d5", "Hydroxychloroquine outcomes",
     "Hydroxychloroquine did not reduce mortality among hospitalized patients with COVID-19 in the RECOVERY trial."),
    ("d6", "Mask effectiveness",
     "Face masks reduce the emission of respiratory droplets and lower community transmission of the coronavirus."),
    ("d7", "Origin of the virus",
     "Phylogenetic analysis suggests SARS-CoV-2 is closely related to bat coronavirus RaTG13, supporting a zoonotic origin."),
    ("d8", "Loss of smell",
     "Anosmia, a sudden loss of smell, is a frequently reported early symptom of COVID-19 infection."),
    ("d9", "Influenza seasonality",
     "Seasonal influenza peaks in winter months in temperate climates and is caused by influenza A and B viruses."),
    ("d10", "Dexamethasone mortality",
     "Dexamethasone reduced 28-day mortality among patients receiving invasive mechanical ventilation."),
]


def toy_trec():
    corpus = {d: {"title": t, "text": x} for d, t, x in _TOY}
    queries = {"1": "what is the incubation period of the coronavirus",
               "2": "how does the virus enter human cells",
               "3": "does hydroxychloroquine help covid patients",
               "4": "what drugs reduce covid-19 mortality",
               "5": "where did the coronavirus come from"}
    qrels = {"1": {"d1": 2}, "2": {"d2": 2}, "3": {"d5": 2, "d4": 1},
             "4": {"d10": 2, "d4": 1, "d5": 1}, "5": {"d7": 2}}
    return corpus, queries, qrels


def toy_qa():
    articles = {d: f"{t}\n{x}" for d, t, x in _TOY}
    qa = [("What is the median incubation period of SARS-CoV-2?", "5.1 days", "d1"),
          ("Which receptor does the spike protein bind?", "angiotensin-converting enzyme 2 (ACE2) receptor", "d2"),
          ("Which drug shortened time to recovery?", "Remdesivir", "d4"),
          ("What early symptom is frequently reported?", "Anosmia", "d8"),
          ("Which bat coronavirus is SARS-CoV-2 related to?", "RaTG13", "d7"),
          ("What reduced 28-day mortality in ventilated patients?", "Dexamethasone", "d10"),
          ("What contributes to spread in poorly ventilated spaces?", "aerosols", "d3"),
          ("What do face masks reduce?", "the emission of respiratory droplets", "d6")]
    ex = [QAExample(str(i), q, [a], [articles[d].find(a)], d) for i, (q, a, d) in enumerate(qa)]
    return ex, articles
