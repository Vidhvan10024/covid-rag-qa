# Does Better Retrieval Mean Better Answers?

Code, model outputs and annotations for the paper

> Mayurkumar Marolia and Hardik Joshi. *Does Better Retrieval Mean Better Answers? Selecting Retrievers for Retrieval-Augmented Question Answering over a COVID-19 Scholarly Collection.* Submitted to the International Journal on Digital Libraries, 2026.

We compare five retrieval pipelines (BM25, the general-domain dense retriever BGE, the biomedical dense retriever MedCPT, reciprocal rank fusion of all three, and fusion followed by cross-encoder re-ranking) in two settings: ad hoc retrieval on TREC-COVID, and as the retrieval stage of a question-answering system that answers the COVID-QA questions from the full set of 147 COVID-QA articles. Answers are written by two open-weight readers, Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, both run in 4-bit on a single 16 GB GPU.

The main finding is that the ranking of the pipelines on TREC-COVID tells you nothing about the quality of the answers they support (Kendall's τ = 0.00), whereas the share of questions for which a pipeline retrieves the answer-bearing passage predicts answer quality exactly (τ = 1.00, for both readers).

## What is here

```
src/                  experiment code (retrieval, reader, metrics, statistics, figures, analysis)
configs/default.yaml  every setting used in the paper
notebooks/
  01_run_experiments.ipynb       reruns all experiments on Google Colab (GPU)
  02_analysis_and_figures.ipynb  reproduces the paper's tables, statistics and Figs 2-4 (CPU only)
results/
  results_full/         primary reader (Qwen2.5-7B-Instruct): Experiments 1, 2 and 3
  results_full_llama/   second reader (Llama-3.1-8B-Instruct): Experiment 2
annotations/          human validation of answer attribution and the error analysis (see annotations/README.md)
scripts/              recreates the annotation sheets from the predictions
paper_figures/        Figs 2-4 as they appear in the paper (PNG and TIFF)
tests/                tests of the real model code paths with tiny models
run_all.sh            runs everything from the command line on a GPU machine
```

## Checking the numbers in the paper (no GPU needed)

Every table and figure in the results section can be rebuilt from the files in `results/` and `annotations/` in a couple of minutes on an ordinary laptop or in Colab without a GPU:

```bash
pip install -r requirements.txt
python -m src.make_figures --root results/results_full        # retrieval and QA tables
python -m src.paper_figures                                    # Figs 2-4
```

or open `notebooks/02_analysis_and_figures.ipynb`, which walks through Tables 2-7 and Figs 2-4 one at a time, including the inter-annotator agreement and the validity check of the automatic attribution score.

## Rerunning the experiments

**On Google Colab.** Open `notebooks/01_run_experiments.ipynb` in Colab, select a T4 GPU and run the cells from the top. The notebook explains each step. A pilot run on reduced data takes about an hour; the full run takes several hours per reader and usually needs more than one Colab session. Progress is saved to Google Drive, and an interrupted step continues where it stopped.

**On your own machine** (one GPU with at least 16 GB of memory):

```bash
pip install -r requirements.txt
python -m src.run_retrieval --smoke && python -m src.run_qa --smoke --stage all   # 1-minute check, no downloads
bash run_all.sh                                                                     # full run
```

For a pilot, set `trec_covid.max_docs: 20000` and `qa.max_questions: 200` in a copy of the config and pass it to `run_all.sh`. To run the second reader, set `qa.reader_model: meta-llama/Llama-3.1-8B-Instruct` and use the passage setting chosen by the primary reader (`results/results_full/exp3_ablation/best.json`: 100-word passages, k = 10).

### Data and models

Everything is downloaded automatically on first use.

| Resource | Source | Notes |
|---|---|---|
| TREC-COVID (BEIR release) | [BEIR](https://github.com/beir-cellar/beir) | 171,332 documents, 50 topics |
| COVID-QA | [deepset/covid_qa_deepset](https://huggingface.co/datasets/deepset/covid_qa_deepset) | 2,019 questions, 147 articles |
| BM25 | [bm25s](https://github.com/xhluca/bm25s) | k1 = 0.9, b = 0.4 |
| BGE | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | |
| MedCPT | [ncbi/MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder), [Article-Encoder](https://huggingface.co/ncbi/MedCPT-Article-Encoder), [Cross-Encoder](https://huggingface.co/ncbi/MedCPT-Cross-Encoder) | |
| Readers | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Llama is gated: accept its licence on Hugging Face and log in with a token |
| Attribution scorer | [cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) | |

### Versions we used

All runs used a Tesla T4 (16 GB) on Google Colab. The exact package lists are in `results/*/environment.txt`. The main ones:

| Package | Primary reader run | Second reader run |
|---|---|---|
| torch | 2.11.0 (CUDA 12.8) | 2.11.0 (CUDA 13.0) |
| transformers | 5.16.1 | 5.18.0 |
| bitsandbytes | 0.50.2 | 0.50.2 |
| bm25s | 0.3.11 | 0.3.13 |
| ranx | 0.3.21 | 0.3.21 |

Colab updated its packages between the two runs. Retrieval results were identical in both. Decoding is greedy, so on the same hardware and library versions the answers are deterministic; on a different GPU or with other library versions, expect small differences in individual answers.

## Output files

`results/results_full/exp2_main/predictions.jsonl` has one line per question and condition with the raw model output, the parsed answer, the ids of the passages shown and cited, and all per-question scores. `summary.csv`, `significance.csv` and `transfer_vs_oracle.csv` in the same folder are the aggregated results behind Tables 3 and 4. `exp1_retrieval/` holds the TREC run files for every system and the per-topic nDCG@10 used in the significance tests.

## Citation

If you use this code or the annotations, please cite the paper (citation details will be added on publication) and this repository; see `CITATION.cff`. Archived release: [ZENODO DOI].

## Licence

Code: MIT (see `LICENSE`). Annotations and model outputs: CC BY 4.0 (see `DATA_LICENSE.md`). The datasets and models used keep their own licences.

## Acknowledgements

We thank Jaykumar Jasani (Government Science College, Chikhli) for independently annotating the answer-attribution sample.
