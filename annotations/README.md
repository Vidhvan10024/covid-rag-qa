# Annotations

Two kinds of human judgement were collected for the paper. Both use answers of the primary reader (Qwen2.5-7B-Instruct) on the COVID-QA test articles, drawn with a fixed seed by `scripts/make_annotation_sheets.py`.

## Attribution of answers (Section 4.7 and Table 5 of the paper)

`faithfulness_validation.csv`: 102 answers, 17 from each of the six conditions that give the reader evidence (oracle, BM25, BGE, MedCPT, hybrid, hybrid + rerank). Only answers that cite at least one passage were eligible.

| Column | Meaning |
|---|---|
| item | item id as shown to the annotators (F001-F102, in random order) |
| qid, condition | question id in COVID-QA and the system that produced the answer (hidden from the annotators) |
| question, answer, cited_passages | what the annotators saw |
| label_A, note_A | annotator A's judgement (Yes/No) and comment |
| label_B, note_B | annotator B's judgement and comment, made independently |
| final_label | the label used for the supported rates: the common label, or the label agreed in discussion for the five disagreements |
| nli_score | the automatic attribution score (maximum NLI entailment over sentence windows of the cited passages) |
| f1, contains | answer accuracy against the gold answer |

`faithfulness_disagreements.csv` lists the five items on which the annotators disagreed and how each was settled.

Annotator A is the first author. Annotator B is a colleague who did not take part in the experiments. Both saw the question, the answer and the cited passages, but not the system or the automatic score.

### Guidelines given to the annotators

Decide whether the passage(s) the AI cited **support** its answer.

- **Yes**: the passages state the answer, or it follows directly from what they say. Wording may differ.
- **No**: the passages do not contain the answer, contradict it, or the answer adds claims that are not in them.

Rules:

1. Judge only against the passages shown. Do not use your own knowledge, and do not judge whether the answer is medically true.
2. Do not look anything up and do not discuss items with the other annotator until both of you have finished.
3. If the answer is only partly supported (one part is in the passage, another part is not), label No and add a note.
4. Use the notes column for any comment, especially when unsure.

Disagreements were settled afterwards by the two annotators together: each explained their label, the original rule was applied again, and one final label was agreed. The original labels were kept unchanged for computing agreement. Three of the five disagreements concerned answers that were only partly supported (rule 3) and one a cited passage that breaks off before the supporting statement (rule 1); in these four, annotator B had accepted the answer and annotator A had labelled it No. The fifth concerned a one-character transcription slip in the answer. The reason for each final label is given in `faithfulness_disagreements.csv`.

## Error analysis (Section 4.8 and Table 6)

`error_analysis_best_pipeline.csv`: 50 randomly sampled wrong answers (answer does not contain the gold span) of the best pipeline, hybrid + rerank.

`error_analysis_bge_distractors.csv`: 25 of the 157 test questions on which BGE retrieved the answer-bearing passage and answered wrongly while the oracle answered correctly. `distractor_present` records whether the other passages contained topically similar but non-answering text.

One main cause was chosen per answer:

| Category | Meaning |
|---|---|
| Retrieval miss | the answer is in none of the ten passages shown |
| Wrong passage used | the answer was in a shown passage, but the reader answered from a different one |
| Extraction error | right passage, wrong or incomplete phrase |
| Correct but scored wrong | the answer is right (reworded, longer, or a valid alternative) but does not contain the gold wording |
| Wrongly declined | the reader replied "insufficient evidence" although a shown passage contained the answer |
| Gold answer problem | the gold answer is wrong or incomplete, or the question is ambiguous |
| Other | none of the above |

If two categories applied, the earlier one in this list was chosen and the other mentioned in the notes. The error analysis was done by annotator A.

## Licence

CC BY 4.0. The questions, gold answers and passages come from COVID-QA (Möller et al., 2020) and the CORD-19 articles it is built on, and remain subject to their original terms.
