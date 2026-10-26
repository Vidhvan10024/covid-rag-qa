"""Retrieval components: BM25, dense bi-encoders, hybrid fusion, cross-encoder re-ranking.

All retrievers share one interface:
    index(docs: List[Doc])                       Doc = (doc_id, title, text)
    search(queries: Dict[qid, str], k) -> Dict[qid, Dict[doc_id, score]]
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from .ckpt import load_json, load_npy, save_json, save_npy

Doc = Tuple[str, str, str]
RunDict = Dict[str, Dict[str, float]]


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Lexical baseline
# ----------------------------------------------------------------------------
class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = k1, b

    def _tok(self, texts):
        import bm25s
        try:
            import Stemmer  # PyStemmer (optional)
            stemmer = Stemmer.Stemmer("english")
        except ImportError:
            stemmer = None
        return bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)

    def index(self, docs: List[Doc]):
        import bm25s
        self.ids = [d[0] for d in docs]
        self.model = bm25s.BM25(k1=self.k1, b=self.b)
        self.model.index(self._tok([f"{t} {x}" for _, t, x in docs]), show_progress=False)
        return self

    def search(self, queries: Dict[str, str], k: int) -> RunDict:
        qids = list(queries)
        k = min(k, len(self.ids))
        idx, sc = self.model.retrieve(self._tok([queries[q] for q in qids]), k=k,
                                      show_progress=False, backend_selection="numpy")
        return {q: {self.ids[int(i)]: float(s) for i, s in zip(idx[r], sc[r])} for r, q in enumerate(qids)}


# ----------------------------------------------------------------------------
# Dense bi-encoder (BGE, MedCPT, E5 …) — exact inner-product search
# ----------------------------------------------------------------------------
class DenseRetriever:
    def __init__(self, name: str, query_model: str, doc_model: str, pooling="cls",
                 query_prefix="", doc_prefix="", title_pair=False,
                 max_len_query=64, max_len_doc=512, batch_size=64, cache_dir=None, shard_size=10000,
                 normalize=True, **_):
        self.name, self.pooling = name, pooling
        self.qm_name, self.dm_name = query_model, doc_model
        self.qp, self.dp, self.title_pair = query_prefix, doc_prefix, title_pair
        self.lq, self.ld, self.bs, self.cache_dir = max_len_query, max_len_doc, batch_size, cache_dir
        self.shard = shard_size
        self.normalize = normalize   # BGE: cosine (normalised); MedCPT: raw dot product, as trained

    def _load(self, name):
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(_device()).eval()
        if _device() == "cuda":
            model = model.half()
        return tok, model

    def _encode(self, tok, model, texts, max_len, pairs=None):
        import torch
        out = []
        n = len(pairs[0]) if pairs is not None else len(texts)
        for i in range(0, n, self.bs):
            if pairs is not None:
                batch = tok(pairs[0][i:i + self.bs], pairs[1][i:i + self.bs], truncation="longest_first",
                            max_length=max_len, padding=True, return_tensors="pt")
            else:
                batch = tok(texts[i:i + self.bs], truncation=True, max_length=max_len,
                            padding=True, return_tensors="pt")
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                h = model(**batch).last_hidden_state
            if self.pooling == "mean":
                m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
                e = (h * m).sum(1) / m.sum(1)
            else:
                e = h[:, 0]
            e = e.float()
            if self.normalize:
                e = torch.nn.functional.normalize(e, dim=-1)
            out.append(e.cpu().numpy())
        return np.vstack(out).astype(np.float32)

    def index(self, docs: List[Doc]):
        self.ids = [d[0] for d in docs]
        cache = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            import hashlib
            sig = hashlib.md5(("|".join(self.ids[:1000]) + self.dm_name
                               + ("" if self.normalize else "|dot")).encode()).hexdigest()[:10]
            cache = os.path.join(self.cache_dir, f"emb_{self.name}_{len(docs)}_{sig}.npy")
            emb = load_npy(cache)
            if emb is not None and emb.shape[0] == len(docs):
                self.emb = emb
                print(f"[{self.name}] loaded cached embeddings {self.emb.shape}")
                return self
        # Encode in shards; each finished shard is saved, so a disconnect loses at most one shard.
        tok, model = None, None
        parts = []
        for si, s0 in enumerate(range(0, len(docs), self.shard)):
            chunk = docs[s0:s0 + self.shard]
            part = f"{cache[:-4]}_part{si:03d}.npy" if cache else None
            arr = load_npy(part) if part else None
            if arr is None or arr.shape[0] != len(chunk):
                if model is None:
                    tok, model = self._load(self.dm_name)
                if self.title_pair:
                    arr = self._encode(tok, model, None, self.ld,
                                       pairs=([t for _, t, _ in chunk], [x for _, _, x in chunk]))
                else:
                    arr = self._encode(tok, model, [f"{self.dp}{t} {x}".strip() for _, t, x in chunk], self.ld)
                if part:
                    save_npy(arr, part)
            print(f"[{self.name}] encoded {min(s0 + self.shard, len(docs)):,}/{len(docs):,} documents")
            parts.append(arr)
        del model
        self.emb = np.vstack(parts).astype(np.float32)
        if cache:
            save_npy(self.emb, cache)
            for si in range(len(parts)):
                try:
                    os.remove(f"{cache[:-4]}_part{si:03d}.npy")
                except OSError:
                    pass
        return self

    def search(self, queries: Dict[str, str], k: int) -> RunDict:
        tok, model = self._load(self.qm_name)
        qids = list(queries)
        qe = self._encode(tok, model, [f"{self.qp}{queries[q]}" for q in qids], self.lq)
        return _topk_ip(qids, qe, self.emb, self.ids, k)


def _topk_ip(qids, qe, de, ids, k, block=50_000) -> RunDict:
    """Exact top-k inner product, blocked over the corpus to bound memory."""
    k = min(k, len(ids))
    best_s = np.full((len(qids), 0), -np.inf, dtype=np.float32)
    best_i = np.zeros((len(qids), 0), dtype=np.int64)
    for s in range(0, len(ids), block):
        sc = qe @ de[s:s + block].T
        kk = min(k, sc.shape[1])
        part = np.argpartition(-sc, kk - 1, axis=1)[:, :kk]
        best_s = np.hstack([best_s, np.take_along_axis(sc, part, 1)])
        best_i = np.hstack([best_i, part + s])
        top = np.argsort(-best_s, axis=1)[:, :k]
        best_s, best_i = np.take_along_axis(best_s, top, 1), np.take_along_axis(best_i, top, 1)
    return {q: {ids[i]: float(v) for i, v in zip(best_i[r], best_s[r])} for r, q in enumerate(qids)}


class TfidfStubRetriever:
    """Offline stand-in for a dense encoder, used only by --smoke tests."""

    def __init__(self, name: str, **_):
        self.name = name

    def index(self, docs: List[Doc]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.ids = [d[0] for d in docs]
        self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2) if "med" in self.name else (1, 1))
        self.emb = self.vec.fit_transform([f"{t} {x}" for _, t, x in docs])
        return self

    def search(self, queries, k):
        qids = list(queries)
        qe = self.vec.transform([queries[q] for q in qids])
        return _topk_ip(qids, qe.toarray().astype(np.float32),
                        self.emb.toarray().astype(np.float32), self.ids, k)


# ----------------------------------------------------------------------------
# Hybrid fusion
# ----------------------------------------------------------------------------
def rrf_fuse(runs: List[RunDict], k: int = 60, depth: int = 1000) -> RunDict:
    """Reciprocal Rank Fusion (Cormack et al., 2009) — parameter-free w.r.t. score scales."""
    fused: RunDict = {}
    for q in set().union(*runs):
        acc: Dict[str, float] = {}
        for run in runs:
            ranked = sorted(run.get(q, {}).items(), key=lambda x: -x[1])[:depth]
            for r, (d, _) in enumerate(ranked, 1):
                acc[d] = acc.get(d, 0.0) + 1.0 / (k + r)
        fused[q] = dict(sorted(acc.items(), key=lambda x: -x[1])[:depth])
    return fused


# ----------------------------------------------------------------------------
# Cross-encoder re-ranking
# ----------------------------------------------------------------------------
class CrossEncoderReranker:
    def __init__(self, model: str, depth: int = 100, max_len: int = 512, batch_size: int = 32, **_):
        self.model_name, self.depth, self.max_len, self.bs = model, depth, max_len, batch_size

    def rerank(self, run: RunDict, queries: Dict[str, str], texts: Dict[str, str],
               cache_path: str | None = None, save_every: int = 100) -> RunDict:
        """Re-scores the top-`depth` candidates of each query. With cache_path, progress is saved
        every `save_every` queries and reused on the next run if the candidates are identical."""
        import torch
        from tqdm.auto import tqdm
        saved = (load_json(cache_path) or {}) if cache_path else {}
        out: RunDict = {}
        todo = []
        for q, docs in run.items():
            cand = [d for d, _ in sorted(docs.items(), key=lambda x: -x[1])[:self.depth]]
            prev = saved.get(q)
            if prev and prev["cand"] == cand and prev.get("query") == queries[q]:
                out[q] = dict(sorted(zip(cand, prev["scores"]), key=lambda x: -x[1]))
            else:
                todo.append((q, cand))
        if not todo:
            print(f"[rerank] all {len(out)} queries resumed from checkpoint")
            return out
        if out:
            print(f"[rerank] resumed {len(out)} queries, {len(todo)} to go")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(_device()).eval()
        if _device() == "cuda":
            model = model.half()
        for n, (q, cand) in enumerate(tqdm(todo, desc="re-ranking", leave=False), 1):
            scores = []
            for i in range(0, len(cand), self.bs):
                b = cand[i:i + self.bs]
                enc = tok([queries[q]] * len(b), [texts[d] for d in b], truncation=True,
                          max_length=self.max_len, padding=True, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    logits = model(**enc).logits
                scores.extend((logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]).float().cpu().tolist())
            out[q] = dict(sorted(zip(cand, scores), key=lambda x: -x[1]))
            saved[q] = {"query": queries[q], "cand": cand, "scores": scores}
            if cache_path and (n % save_every == 0 or n == len(todo)):
                save_json(saved, cache_path)
        return out


class StubReranker(CrossEncoderReranker):
    """Offline stand-in (token-overlap scoring), used only by --smoke tests."""

    def __init__(self, depth: int = 100, **_):
        self.depth = depth

    def rerank(self, run, queries, texts, cache_path=None, save_every=100):
        out = {}
        for q, docs in run.items():
            qt = set(queries[q].lower().split())
            cand = [d for d, _ in sorted(docs.items(), key=lambda x: -x[1])[:self.depth]]
            sc = {d: len(qt & set(texts[d].lower().split())) + docs[d] * 1e-3 for d in cand}
            out[q] = dict(sorted(sc.items(), key=lambda x: -x[1]))
        return out


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def build_all(cfg: dict, docs: List[Doc], smoke: bool, cache_dir: str | None = None):
    """Returns {name: fitted retriever} for bm25 + every dense model in the config."""
    rcfg = cfg["retrievers"]
    out = {"bm25": BM25Retriever(**rcfg["bm25"]).index(docs)}
    for name, dcfg in rcfg["dense"].items():
        r = (TfidfStubRetriever(name) if smoke else
             DenseRetriever(name, batch_size=rcfg.get("batch_size", 64), cache_dir=cache_dir, **dcfg))
        out[name] = r.index(docs)
    return out


def build_reranker(cfg: dict, smoke: bool):
    rr = cfg["retrievers"]["reranker"]
    return StubReranker(**rr) if smoke else CrossEncoderReranker(**rr)
