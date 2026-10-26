"""Integration test of the REAL model code paths (not the --smoke stand-ins).

Builds tiny random models with the same architectures/interfaces as the real ones
(BERT encoders incl. MedCPT-style title/abstract pairs, a 1-logit cross-encoder,
a 3-label NLI model, a Qwen2 chat LLM), points the config at them, and runs
Experiments 1, 3 and 2 on the toy data. Scores are meaningless; the test only
proves every code path executes and writes its outputs.

    python -m tests.test_real_paths
"""
import copy
import os
import shutil
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import src  # noqa: E402,F401  (sets JAX/TF env vars first)
from src import data as D  # noqa: E402


def build_tiny_models(d):
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizerFast
    torch.manual_seed(0)
    words = sorted({w.lower().strip(".,;:?()%\"'[]") for _, t, x in D._TOY for w in (t + " " + x).split()}
                   | set("what is the which does of a an to in reply with excerpts question answer insufficient "
                         "evidence instructions shortest exact phrase from that supporting number s square "
                         "brackets e g if do not contain exactly you are biomedical assistant system user".split()))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [w for w in words if w] + list("0123456789[]:.,?\"'")
    os.makedirs(d, exist_ok=True)
    vf = os.path.join(d, "vocab.txt")
    open(vf, "w").write("\n".join(dict.fromkeys(vocab)))
    small = dict(vocab_size=len(dict.fromkeys(vocab)), hidden_size=32, num_hidden_layers=2,
                 num_attention_heads=2, intermediate_size=64, max_position_embeddings=512)

    def save(model, name, chat=False):
        p = os.path.join(d, name)
        tok = BertTokenizerFast(vocab_file=vf, do_lower_case=True)
        if chat:
            tok.chat_template = ("{% for m in messages %}<{{ m['role'] }}> {{ m['content'] }}\n{% endfor %}"
                                 "{% if add_generation_prompt %}<assistant>{% endif %}")
        model.save_pretrained(p); tok.save_pretrained(p)
        return p

    paths = {
        "enc": save(BertModel(BertConfig(**small)), "enc"),
        "ce": save(BertForSequenceClassification(BertConfig(**small, num_labels=1)), "ce"),
        "nli": save(BertForSequenceClassification(BertConfig(
            **small, num_labels=3, id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
            label2id={"contradiction": 0, "entailment": 1, "neutral": 2})), "nli"),
        "llm": build_tiny_llama(os.path.join(d, "llm"), small),
    }
    return paths


def build_tiny_llama(path, small):
    """Tiny Llama-style chat model with a REAL trained BPE tokenizer that adds a begin-of-text
    token, and a chat template that also starts with it (as Llama 3 does). Used to check the
    reader feeds exactly one begin token and non-empty inputs."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    corpus = [x for _, t, x in D._TOY] + [t for _, t, _ in D._TOY] + [
        "Question: Instructions: Reply with the shortest exact phrase from the excerpts insufficient evidence",
        "system user assistant You are a biomedical question-answering assistant Excerpts"]
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.train_from_iterator(corpus * 5, trainers.BpeTrainer(
        vocab_size=600, special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tk.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", tk.token_to_id("<s>"))])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    tok.chat_template = ("{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=None))
    model.save_pretrained(path); tok.save_pretrained(path)
    return path


def main():
    tmp = tempfile.mkdtemp(prefix="covidqa_test_")
    try:
        m = build_tiny_models(os.path.join(tmp, "models"))
        cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml")))
        cfg["output_dir"] = os.path.join(tmp, "results")
        cfg["cache_dir"] = os.path.join(tmp, "cache")
        for name in cfg["retrievers"]["dense"]:
            cfg["retrievers"]["dense"][name].update(query_model=m["enc"], doc_model=m["enc"])
        cfg["retrievers"]["reranker"]["model"] = m["ce"]
        cfg["qa"].update(reader_model=m["llm"], load_in_4bit=False, nli_model=m["nli"],
                         max_new_tokens=8, bootstrap_samples=200, ablation_k=[1, 3], ablation_chunk_words=[100, 200])
        cfg_path = os.path.join(tmp, "cfg.yaml")
        yaml.safe_dump(cfg, open(cfg_path, "w"))

        # reader input check: exactly one begin token and real content (Llama-style tokenizer)
        from src.reader import HFReader, build_prompt, SYSTEM
        rd = HFReader(m["llm"], load_in_4bit=False, max_new_tokens=4, batch_size=2)
        chat = rd.tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": build_prompt("What is IFITM?", ["IFITM is a protein."])}],
                                          tokenize=False, add_generation_prompt=True)
        ids = rd.tok(chat, add_special_tokens=False)["input_ids"]
        assert ids.count(rd.tok.bos_token_id) == 1 and len(ids) > 20, (ids.count(rd.tok.bos_token_id), len(ids))
        assert len(rd.generate([build_prompt("What is IFITM?", None)] * 3)) == 3
        print(f"reader input check OK: 1 begin token, {len(ids)} tokens")
        del rd

        # real loaders need the internet; swap in the toy data but keep every model path real
        D.load_trec_covid = lambda *_a, **_k: D.toy_trec()
        D.load_covid_qa = lambda *_a, **_k: D.toy_qa()

        from src import run_qa, run_retrieval
        sys.argv = ["run_retrieval", "--config", cfg_path]
        run_retrieval.main()
        run_qa.run_stage(copy.deepcopy(cfg), "ablation", smoke=False)
        run_qa.run_stage(copy.deepcopy(cfg), "main", smoke=False)
        run_qa.run_stage(copy.deepcopy(cfg), "main", smoke=False)   # second run must resume from checkpoints

        res = cfg["output_dir"]
        expected = ["exp1_retrieval/table_retrieval.csv", "exp1_retrieval/per_query_ndcg10.csv",
                    "exp3_ablation/summary.csv", "exp3_ablation/best.json",
                    "exp2_main/summary.csv", "exp2_main/significance.csv", "exp2_main/predictions.jsonl"]
        missing = [f for f in expected if not os.path.exists(os.path.join(res, f))]
        assert not missing, f"missing outputs: {missing}"
        emb = os.listdir(os.path.join(cfg["cache_dir"], "trec_emb"))
        assert len(emb) == 2, f"expected 2 cached embedding files, got {emb}"

        # the random reader rarely cites, so exercise the NLI attribution scorer directly
        from src.metrics import NLIFaithfulness, sentence_windows
        wins = sentence_windows("The median incubation period was 5.1 days. Most patients had fever. "
                                "Symptoms appeared within 11.5 days.")
        assert len(wins) == 5, wins            # 3 single sentences + 2 consecutive pairs
        f = NLIFaithfulness(m["nli"]).score(["What is the incubation period?", "q2"], ["5.1 days", "x"],
                                            [[D._TOY[0][2]], []])
        assert 0.0 <= f[0] <= 1.0 and f[1] == 0.0, f

        from src import make_figures
        sys.argv = ["make_figures", "--root", res]
        make_figures.main()
        print("\nREAL-PATH TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
