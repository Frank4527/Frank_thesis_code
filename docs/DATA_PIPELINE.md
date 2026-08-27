# Data pipeline — bank statements & quarterly reports

## ⚠️ The data is PRIVATE
The source documents are **proprietary data from Evolution AI**. The raw documents
and their field **values** (names, amounts, dates, account details) are confidential
and are **not** committed to git or shared. Only the images/manifests on the private
server are used (`/home/frank/from_martin`, `/home/frank/data`). This file documents
**only the format/schema at each pipeline stage**, so the pipeline is reproducible
with equivalent data without exposing any private content.

The pipeline: **raw docs → converter → manifest → dataloader → training input/target.**

---

## Stage 1 — Raw data (private, from Evolution AI)
- Location: `from_martin/{raw_output/, processed_output/}`, one entry per document,
  keyed by a content hash (e.g. `00095913a686…`).
- Each entry = a **page image** + its **ground-truth field annotations** in
  Evolution's internal export format.

## Stage 2 — Converter: raw → unified manifest  (`data_prep/build_*.py`)
Scripts: `build_thesis_datasets.py`, `build_quarterly_dataset.py`,
`build_company_manifest.py`, `detect_language_split.py`.
- Read the raw annotations and **normalise them into one common schema**.
- **Split BY DOCUMENT** (a document's pages never straddle train/val/test — no leakage),
  with a **fixed seed** for reproducibility.
- Language split (de/en/po) via `detect_language_split.py`.
- Output: one `manifest.jsonl` per split under
  `data/Thesis_datasets/<Dataset>/<train|validation|test>/manifest.jsonl`.

## Stage 3 — Manifest / dataloader format  (what `load_documents()` returns)
Each **line of `manifest.jsonl` = one document**, a JSON object:
```jsonc
{
  "image":    "<absolute path to the page image>",   // str
  "gt_parse": { ...nested field tree... },            // dict  ← the extraction TARGET
  "bank":     "<source/institution tag>",             // str  (metadata)
  "language": "de" | "en" | "po"                      // str  (metadata)
}
```
- `gt_parse` is a **nested key→value field tree**; repeated records (e.g. transaction
  rows) are represented as **lists of sub-objects**. Depth is typically 1–2 levels.
- `Metrics.document_extraction.load_documents(manifest)` reads the file and returns a
  **list of these row dicts** (image paths made absolute). This is the "format the
  dataloader gives."

## Stage 4 — Training format  (what we feed DoRA)
Built by `Metrics.document_extraction.extract_messages` + `Training_Dora.qa_data.mask_and_batch`:
- **PROMPT (input):** one chat *user* turn = `[ page image ]` + the fixed text
  instruction `"Extract all fields from this document as JSON."`, run through the
  processor's chat template (image tokens + text).
- **TARGET (label):** `serialize_gt(gt_parse)` = `json.dumps(gt_parse)` — the field
  tree rendered as a **JSON string** — followed by the EOS token.
- **Full sequence to the model** = `PROMPT + TARGET + EOS`. **Loss is computed on the
  TARGET tokens only** (prompt tokens masked to `-100`), so the model learns to emit
  the `gt_parse` JSON given the image.

---

## Flow summary
| stage | code | format |
|---|---|---|
| 1. raw | `from_martin/` (Evolution) | image + internal annotations (private) |
| 2. convert | `data_prep/build_*.py` | → `manifest.jsonl`, split-by-doc, seeded |
| 3. load | `document_extraction.load_documents` | list of `{image, gt_parse, bank, language}` |
| 4. train | `qa_data.mask_and_batch` | `(image + "extract as JSON")` → `json.dumps(gt_parse)`; loss on target only |

Evaluation uses the **same** Stage-3/4 machinery (same `extract_messages`), but scores the
model's generated JSON against `gt_parse` with `Metrics.field_f1` (exact key+value field-F1).
