"""Answer-quality metrics, faithfulness scoring and paired significance tests."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------
# SQuAD-style answer metrics (Rajpurkar et al., 2016)
# ----------------------------------------------------------------------------
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: Sequence[str]) -> float:
    return float(max(normalize(pred) == normalize(g) for g in golds))


def token_f1(pred: str, golds: Sequence[str]) -> float:
    def f1(p, g):
        pt, gt = normalize(p).split(), normalize(g).split()
        common = Counter(pt) & Counter(gt)
        same = sum(common.values())
        if not pt or not gt:
            return float(pt == gt)
        if same == 0:
            return 0.0
        pr, rc = same / len(pt), same / len(gt)
        return 2 * pr * rc / (pr + rc)
    return max(f1(pred, g) for g in golds)


def containment(pred: str, golds: Sequence[str]) -> float:
    """Lenient 'answer recall': gold span appears inside the (possibly verbose) prediction or vice-versa."""
    p = normalize(pred)
    return float(bool(p) and any(normalize(g) in p or (len(p) > 3 and p in normalize(g)) for g in golds))


# ----------------------------------------------------------------------------
# Faithfulness / attribution via NLI
# ----------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")


def sentence_windows(text: str, max_window: int = 2, max_words: int = 80) -> List[str]:
    """Single sentences plus consecutive pairs, each kept short.

    NLI models are trained on sentence-length premises; long premises depress the
    entailment probability regardless of content. Short windows remove that confound,
    and 2-sentence windows still catch answers whose support spans a sentence break.
    """
    sents = [x.strip() for x in _SENT_SPLIT.split(text) if x.strip()]
    wins = []
    for w in range(1, max_window + 1):
        for i in range(0, max(1, len(sents) - w + 1)):
            chunk = " ".join(sents[i:i + w])
            if chunk:
                wins.append(" ".join(chunk.split()[:max_words]))
    return list(dict.fromkeys(wins))


class NLIFaithfulness:
    """Attribution score: max P(entailment) of the claim 'the answer to Q is A' over
    sentence windows of the passages the answer CITES (ALCE-style; Gao et al., 2023).

    - Sentence windows remove the premise-length confound.
    - Scoring only cited passages removes the max-over-k confound (more passages shown
      would otherwise mean more chances of a high score).
    Answers without a valid citation are not scored here (the caller reports the citation
    rate separately). Validate against human labels (Cohen's kappa) before reporting, and
    check oracle >= every RAG condition.
    """

    def __init__(self, model: str, batch_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        lab = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.ent = lab.get("entailment", 1)
        self.bs = batch_size

    def score(self, questions, answers, evidence: List[List[str]]) -> List[float]:
        import torch
        from tqdm.auto import tqdm
        pairs, owner = [], []
        for i, (q, a, ev) in enumerate(zip(questions, answers, evidence)):
            hyp = f"The answer to the question '{q}' is: {a}."
            for p in ev or []:
                for w in sentence_windows(p):
                    pairs.append((w, hyp)); owner.append(i)
        best = [0.0] * len(questions)
        for s in tqdm(range(0, len(pairs), self.bs), desc="faithfulness", leave=False):
            b = pairs[s:s + self.bs]
            enc = self.tok([x for x, _ in b], [y for _, y in b], truncation="only_first",
                           max_length=256, padding=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                pr = self.model(**enc).logits.float().softmax(-1)[:, self.ent].cpu().tolist()
            for j, v in enumerate(pr):
                o = owner[s + j]
                best[o] = max(best[o], v)
        return best


class StubFaithfulness:
    """Offline stand-in: 1.0 if the normalised answer string occurs in the evidence."""

    def score(self, questions, answers, evidence):
        return [float(bool(normalize(a)) and any(normalize(a) in normalize(p) for p in (ev or [])))
                for a, ev in zip(answers, evidence)]


# ----------------------------------------------------------------------------
# Significance testing
# ----------------------------------------------------------------------------
def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 10000, seed: int = 42) -> Dict:
    """Mean difference (b - a), 95% percentile CI, two-sided bootstrap p-value (Efron & Tibshirani)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(1)
    centred = boot - d.mean()
    p = float((np.abs(centred) >= abs(d.mean())).mean())
    return {"diff": float(d.mean()), "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)), "p": max(p, 1.0 / n),
            "ci90_low": float(np.percentile(boot, 5)), "ci90_high": float(np.percentile(boot, 95))}


def randomization_test(a: Sequence[float], b: Sequence[float], n: int = 10000, seed: int = 42) -> float:
    """Two-sided paired randomisation (sign-flip) test — the IR standard (Smucker et al., 2007)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return float(((np.abs((signs * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n + 1))


def mcnemar(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact McNemar test on paired binary outcomes."""
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error)."""
    items = sorted(pvals.items(), key=lambda x: x[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj
