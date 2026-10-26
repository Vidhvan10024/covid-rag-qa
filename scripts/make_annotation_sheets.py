"""Recreates the blank annotation workbooks used for the human validation and the error analysis.

The samples are drawn with a fixed seed from the primary reader's test-set predictions, so running this
script again gives exactly the items that the annotators labelled (the labelled versions are in
annotations/*.csv). It writes three workbooks: one per annotator, one for the error analysis, and a key
that links each item to its condition and automatic score (the key was kept away from the annotators).

    python scripts/make_annotation_sheets.py --predictions results/results_full/exp2_main/predictions.jsonl --out annotation_sheets

COVID-QA is loaded from Hugging Face (deepset/covid_qa_deepset), as in the experiments. If you have the
original COVID-QA.json from github.com/deepset-ai/COVID-QA you can pass it with --covidqa-json instead.
"""
import argparse
import json
import os
import random
import sys

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import data as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--predictions", default="results/results_full/exp2_main/predictions.jsonl")
ap.add_argument("--covidqa-json", default=None)
ap.add_argument("--out", default="annotation_sheets")
args = ap.parse_args()

if args.covidqa_json:
    raw = json.load(open(args.covidqa_json))
    EX, articles, i = {}, {}, 0
    for art in raw["data"]:
        for para in art["paragraphs"]:
            did = str(para["document_id"])
            articles.setdefault(did, para["context"])
            for qa in para["qas"]:
                EX[f"q{i}"] = D.QAExample(f"q{i}", qa["question"].strip(), [a["text"].strip() for a in qa["answers"]],
                                          [a["answer_start"] for a in qa["answers"]], did, qa.get("is_impossible", False))
                i += 1
else:
    examples, articles = D.load_covid_qa({"hf_id": "deepset/covid_qa_deepset"})
    EX = {e.qid: e for e in examples}
PMAP = {p.pid: p for p in D.chunk_articles(articles, 100, 50)}

OUT = args.out
os.makedirs(OUT, exist_ok=True)
SEED = 42
pred = pd.read_json(args.predictions, lines=True)

F = "Arial"
HDR = PatternFill("solid", fgColor="1F3864")
INPUT = PatternFill("solid", fgColor="FFF2CC")   # cells the annotator fills in
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")


def header(ws, cols, widths):
    for j, (c, w) in enumerate(zip(cols, widths), 1):
        cell = ws.cell(row=1, column=j, value=c)
        cell.font = Font(name=F, bold=True, color="FFFFFF")
        cell.fill = HDR
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = BOX
        ws.column_dimensions[cell.column_letter].width = w
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 32


def put(ws, r, c, v, fill=None, bold=False):
    cell = ws.cell(row=r, column=c, value=v)
    cell.font = Font(name=F, bold=bold)
    cell.alignment = WRAP
    cell.border = BOX
    if fill:
        cell.fill = fill
    return cell


def instructions(ws, title, lines, example=None):
    ws.column_dimensions["A"].width = 120
    ws["A1"] = title
    ws["A1"].font = Font(name=F, bold=True, size=14)
    r = 3
    for ln in lines:
        c = ws.cell(row=r, column=1, value=ln)
        c.font = Font(name=F, bold=ln.isupper() and len(ln) < 40)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        r += 1
    if example:
        r += 1
        ws.cell(row=r, column=1, value="EXAMPLE (for format only; not one of your items)").font = Font(name=F, bold=True)
        for ln in example:
            r += 1
            c = ws.cell(row=r, column=1, value=ln)
            c.font = Font(name=F, italic=True)
            c.alignment = Alignment(wrap_text=True)


def passage_text(pid):
    return PMAP[pid].text


# ---------------------------------------------------------------------------
# 1. Faithfulness validation: 17 citing answers from each of 6 conditions = 102
# ---------------------------------------------------------------------------
conds = ["oracle", "bm25", "bge", "medcpt", "hybrid", "hybrid_rerank"]
elig = pred[pred.condition.isin(conds) & (pred.abstained == 0) & (pred.cited_ids.map(len) > 0)]
sample = pd.concat([elig[elig.condition == c].sample(17, random_state=SEED) for c in conds])
sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
sample["item"] = [f"F{i + 1:03d}" for i in range(len(sample))]


def cited_block(ids):
    return "\n\n".join(f"[Passage {n}] {passage_text(p)}" for n, p in enumerate(ids, 1))


FAITH_LINES = [
    "WHAT YOU ARE DOING",
    "Each row shows a question about a COVID-19 research paper, an answer written by an AI system, and the passage(s) the AI said it used. "
    "Decide whether the passage(s) SUPPORT the answer.",
    "",
    "HOW TO LABEL (column E, yellow cells)",
    "Yes = the passage(s) state the answer, or it follows directly from what they say. Wording may differ.",
    "No = the passage(s) do not contain the answer, contradict it, or the answer adds claims that are not in them.",
    "",
    "RULES",
    "1. Judge ONLY against the passage(s) shown. Do not use your own knowledge, and do not judge whether the answer is medically true.",
    "2. Do not look anything up and do not discuss items with the other annotator until both of you have finished.",
    "3. If the answer is only partly supported (one part is in the passage, another part is not), label No and add a note.",
    "4. Use column F for any comment, especially when unsure.",
    "",
    "Work through all items in order. The progress cell at the top right of the Items sheet counts how many you have labelled.",
]
FAITH_EXAMPLE = [
    "Question: What is the median incubation period of SARS-CoV-2?",
    "Answer: 5.1 days",
    "Cited passage: \"... the median incubation period was estimated to be 5.1 days (95% CI 4.5 to 5.8) ...\"",
    "Supported?: Yes      Notes: (blank)",
]


def faith_book(annotator):
    wb = Workbook()
    ins = wb.active
    ins.title = "Instructions"
    instructions(ins, f"Faithfulness annotation: Annotator {annotator}", FAITH_LINES, FAITH_EXAMPLE)
    ws = wb.create_sheet("Items")
    header(ws, ["Item", "Question", "AI answer", "Passage(s) the AI cited", "Supported? (Yes/No)", "Notes"],
           [8, 34, 28, 95, 14, 30])
    dv = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
    ws.add_data_validation(dv)
    for i, r in enumerate(sample.itertuples(), start=2):
        put(ws, i, 1, r.item, bold=True)
        put(ws, i, 2, r.question)
        put(ws, i, 3, r.pred)
        put(ws, i, 4, cited_block(r.cited_ids))
        put(ws, i, 5, None, fill=INPUT)
        put(ws, i, 6, None, fill=INPUT)
        dv.add(ws.cell(row=i, column=5))
    last = len(sample) + 1
    ws["H1"] = "Labelled"
    ws["H1"].font = Font(name=F, bold=True)
    ws["I1"] = f'=COUNTIF(E2:E{last},"Yes")+COUNTIF(E2:E{last},"No")'
    ws["J1"] = f"=I1/{len(sample)}"
    ws["J1"].number_format = "0%"
    for c in ("I1", "J1"):
        ws[c].font = Font(name=F, bold=True)
    path = os.path.join(OUT, f"faithfulness_annotator_{annotator}.xlsx")
    wb.save(path)
    return path


fa, fb = faith_book("A"), faith_book("B")

# ---------------------------------------------------------------------------
# 2. Error analysis
#    Sheet "Errors": 50 random wrong answers from the best system (hybrid_rerank)
#    Sheet "BGE_distractors": 25 cases where BGE retrieved the gold passage but answered
#    wrong while the oracle answered right on the same question
# ---------------------------------------------------------------------------
best = pred[pred.condition == "hybrid_rerank"]
wrong = best[best.contains == 0].sample(50, random_state=SEED).reset_index(drop=True)
oracle_ok = set(pred[(pred.condition == "oracle") & (pred.contains == 1)].qid)
bge = pred[pred.condition == "bge"]
bge_cases = bge[(bge["hit@k"] == 1) & (bge.contains == 0) & bge.qid.isin(oracle_ok)]
n_bge_pool = len(bge_cases)
bge_cases = bge_cases.sample(min(25, n_bge_pool), random_state=SEED).reset_index(drop=True)

CATS = ["Retrieval miss", "Wrong passage used", "Extraction error", "Correct but scored wrong",
        "Wrongly declined", "Gold answer problem", "Other"]
CAT_HELP = {
    "Retrieval miss": "The answer is not in any of the 10 passages shown.",
    "Wrong passage used": "The answer was in a shown passage, but the AI answered from a different passage.",
    "Extraction error": "The AI used the right passage but picked the wrong or incomplete phrase.",
    "Correct but scored wrong": "The AI's answer is right (e.g. reworded, longer, or a valid alternative), but it does not contain the gold wording.",
    "Wrongly declined": "The AI said 'insufficient evidence' although the answer was in a shown passage.",
    "Gold answer problem": "The official (gold) answer is wrong, incomplete, or the question is ambiguous.",
    "Other": "None of the above. Describe in Notes.",
}


def gold_info(qid, ids):
    ex = EX[qid]
    ranks = [n for n, p in enumerate(ids, 1) if D.answer_bearing(PMAP[p], ex)]
    return ("Yes, passage " + ", ".join(map(str, ranks))) if ranks else "No"


def all_block(ids):
    return "\n\n".join(f"[{n}] {passage_text(p)}" for n, p in enumerate(ids, 1))


def cited_nums(ids, cited):
    nums = [str(ids.index(c) + 1) for c in cited if c in ids]
    return ", ".join(nums) if nums else "none"


ERR_LINES = [
    "WHAT YOU ARE DOING",
    "Each row is a question the AI system answered WRONGLY (its answer does not contain the official answer). "
    "Pick the main reason for the error in column I (yellow cells).",
    "",
    "CATEGORIES (choose one per row)",
] + [f"{c}: {h}" for c, h in CAT_HELP.items()] + [
    "",
    "HOW TO WORK",
    "1. Read the question, the official answer and the AI answer.",
    "2. Column E says whether a passage containing the official answer was shown to the AI, and which number it was. "
    "Column G lists all 10 passages in the order the AI saw them.",
    "3. Choose the single best category. If two apply, choose the earlier one in the list above and mention the other in Notes.",
    "",
    "The Summary sheet counts the categories automatically.",
    f"Sheet 'Errors': 50 randomly chosen wrong answers from the best system (hybrid + re-ranking).",
    f"Sheet 'BGE_distractors': {len(bge_cases)} cases (from {n_bge_pool}) where the BGE system WAS shown the right passage "
    "but still answered wrongly, while the oracle answered correctly. For these, note in column J whether the other passages "
    "looked similar but irrelevant (a distractor), which would explain the error.",
]
ERR_EXAMPLE = [
    "Question: Which receptor does the spike protein bind?   Official answer: ACE2   AI answer: TMPRSS2 [3]",
    "Right passage shown?: Yes, passage 1   Passages AI cited: 3",
    "Error category: Wrong passage used   Notes: passage 3 is about TMPRSS2 priming, a near-miss distractor",
]


def error_sheet(wb, name, df, distractor_col):
    ws = wb.create_sheet(name)
    cols = ["Item", "Question", "Official answer", "AI answer", "Right passage shown?", "Passages AI cited",
            "All passages shown to the AI (in order)", "Shown passage containing the official answer", "Error category",
            "Distractor present? / Notes" if distractor_col else "Notes"]
    header(ws, cols, [8, 30, 22, 24, 14, 10, 90, 50, 22, 30])
    dv = DataValidation(type="list", formula1='"' + ",".join(CATS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    prefix = "E" if name == "Errors" else "B"
    for i, r in enumerate(df.itertuples(), start=2):
        ex = EX[r.qid]
        gold_p = [PMAP[p].text for p in r.ctx if D.answer_bearing(PMAP[p], ex)]
        put(ws, i, 1, f"{prefix}{i - 1:03d}", bold=True)
        put(ws, i, 2, r.question)
        put(ws, i, 3, r.gold)
        put(ws, i, 4, r.raw)
        put(ws, i, 5, gold_info(r.qid, r.ctx))
        put(ws, i, 6, cited_nums(r.ctx, r.cited_ids))
        put(ws, i, 7, all_block(r.ctx))
        put(ws, i, 8, gold_p[0] if gold_p else "(not shown)")
        put(ws, i, 9, None, fill=INPUT)
        put(ws, i, 10, None, fill=INPUT)
        dv.add(ws.cell(row=i, column=9))
    return ws, len(df) + 1


wb = Workbook()
ins = wb.active
ins.title = "Instructions"
instructions(ins, "Error analysis: hybrid + re-ranking (and BGE distractor check)", ERR_LINES, ERR_EXAMPLE)
e_ws, e_last = error_sheet(wb, "Errors", wrong, False)
b_ws, b_last = error_sheet(wb, "BGE_distractors", bge_cases, True)

sm = wb.create_sheet("Summary")
header(sm, ["Error category", "Errors: count", "Errors: share", "BGE_distractors: count", "BGE_distractors: share"],
       [28, 14, 14, 22, 22])
for k, c in enumerate(CATS, start=2):
    put(sm, k, 1, c)
    put(sm, k, 2, f'=COUNTIF(Errors!$I$2:$I${e_last},A{k})')
    put(sm, k, 3, f'=IF(B${len(CATS) + 2}=0,0,B{k}/B${len(CATS) + 2})').number_format = "0%"
    put(sm, k, 4, f'=COUNTIF(BGE_distractors!$I$2:$I${b_last},A{k})')
    put(sm, k, 5, f'=IF(D${len(CATS) + 2}=0,0,D{k}/D${len(CATS) + 2})').number_format = "0%"
t = len(CATS) + 2
put(sm, t, 1, "Labelled (total)", bold=True)
put(sm, t, 2, f"=SUM(B2:B{t - 1})", bold=True)
put(sm, t, 3, f"=B{t}/{len(wrong)}", bold=True).number_format = "0%"
put(sm, t, 4, f"=SUM(D2:D{t - 1})", bold=True)
put(sm, t, 5, f"=D{t}/{len(bge_cases)}", bold=True).number_format = "0%"
put(sm, t + 2, 1, f"Share columns divide by the labelled total; the last row shows progress against "
                  f"{len(wrong)} and {len(bge_cases)} items.")
sm.column_dimensions["A"].width = 28
err_path = os.path.join(OUT, "error_analysis.xlsx")
wb.save(err_path)

# ---------------------------------------------------------------------------
# 3. Answer key (kept apart from the annotation files)
# ---------------------------------------------------------------------------
key = sample[["item", "qid", "condition", "faithfulness", "f1", "contains"]].rename(
    columns={"faithfulness": "nli_faithfulness"})
wb = Workbook()
ws = wb.active
ws.title = "Key"
header(ws, list(key.columns), [8, 8, 16, 18, 8, 10])
for i, row in enumerate(key.itertuples(index=False), start=2):
    for j, v in enumerate(row, 1):
        put(ws, i, j, float(v) if isinstance(v, float) else v)
ws.cell(row=len(key) + 3, column=1,
        value="Do not show this sheet to either annotator. It links each item to its system and automatic score, "
              "and is used afterwards to compute agreement.").font = Font(name=F, italic=True)
key_path = os.path.join(OUT, "KEY_do_not_share.xlsx")
wb.save(key_path)

print("faithfulness items:", len(sample), sample.condition.value_counts().to_dict())
print("errors:", len(wrong), "| gold shown in", int((wrong['hit@k'] == 1).sum()), "of them")
print("BGE distractor pool:", n_bge_pool, "-> sampled", len(bge_cases))
for p in (fa, fb, err_path, key_path):
    print(p)
