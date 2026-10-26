"""LLM reader: turns (question, retrieved passages) into a short, cited answer."""
from __future__ import annotations

import re
from typing import List, Tuple

ABSTAIN = "insufficient evidence"

SYSTEM = ("You are a biomedical question-answering assistant. Answer strictly from the "
          "provided COVID-19 literature excerpts; never use outside knowledge.")

RAG_TEMPLATE = """Excerpts:
{context}

Question: {question}

Instructions: Reply with the shortest exact phrase from the excerpts that answers the question, \
followed by the number(s) of the supporting excerpt(s) in square brackets, e.g. "5.1 days [2]". \
If the excerpts do not contain the answer, reply exactly: {abstain}"""

CLOSED_TEMPLATE = """Question: {question}

Instructions: Reply with the shortest phrase that answers the question. \
If you do not know, reply exactly: {abstain}"""


def build_prompt(question: str, passages: List[str] | None) -> str:
    if not passages:
        return CLOSED_TEMPLATE.format(question=question, abstain=ABSTAIN)
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    return RAG_TEMPLATE.format(context=ctx, question=question, abstain=ABSTAIN)


def parse_answer(raw: str) -> Tuple[str, List[int], bool]:
    """-> (answer text without citations, cited passage numbers, abstained?)"""
    raw = raw.strip().split("\n")[0]
    cites = [int(c) for grp in re.findall(r"\[([\d,\s]+)\]", raw) for c in re.findall(r"\d+", grp)]
    ans = re.sub(r"\[[\d,\s]+\]", "", raw).strip().strip('"').strip()
    ans = re.sub(r"^(answer|final answer)\s*:\s*", "", ans, flags=re.I).rstrip(".").strip()
    return ans, cites, ABSTAIN in ans.lower()


class HFReader:
    def __init__(self, model: str, load_in_4bit: bool = True, max_new_tokens: int = 64,
                 batch_size: int = 8, seed: int = 42):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        set_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dtype_key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        kw = {dtype_key: torch.float16, "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
        self.max_new, self.bs = max_new_tokens, batch_size
        self.name = model

    def _batch(self, prompts: List[str]) -> List[str]:
        import torch
        chats = [self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True) for p in prompts]
        # The chat template already inserts any begin-of-text token (Llama does; Qwen has none),
        # so the tokenizer must not add another one.
        enc = self.tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                      temperature=None, top_p=None, top_k=None,
                                      pad_token_id=self.tok.pad_token_id)
        n = enc["input_ids"].shape[1]
        return [self.tok.decode(g, skip_special_tokens=True) for g in gen[:, n:]]

    def generate(self, prompts: List[str]) -> List[str]:
        """Batched greedy decoding; halves the batch automatically on GPU out-of-memory."""
        import torch
        from tqdm.auto import tqdm
        outs, i, bs = [], 0, self.bs
        bar = tqdm(total=len(prompts), desc="generating", leave=False)
        while i < len(prompts):
            try:
                outs.extend(self._batch(prompts[i:i + bs]))
                bar.update(min(bs, len(prompts) - i))
                i += bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(f"[reader] GPU out of memory -> batch size {bs}")
        bar.close()
        return outs


class StubReader:
    """Offline extractive stand-in for --smoke tests: picks the best-overlapping clause."""
    name = "stub-reader"

    def generate(self, prompts: List[str]) -> List[str]:
        outs = []
        for p in prompts:
            q = re.search(r"Question: (.*)", p).group(1).lower()
            qt = set(re.findall(r"\w+", q))
            ex = re.findall(r"\[(\d+)\] (.*?)(?=\n\n\[\d+\] |\n\nQuestion:)", p, flags=re.S)
            if not ex:
                outs.append(ABSTAIN)
                continue
            best = max(((len(qt & set(re.findall(r"\w+", c.lower()))), n, c)
                        for n, t in ex for c in re.split(r"[,.;]", t)), key=lambda x: x[0])
            outs.append(f"{best[2].strip()} [{best[1]}]" if best[0] > 0 else ABSTAIN)
        return outs
