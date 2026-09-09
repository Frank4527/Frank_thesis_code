"""OCRBench and MMBench evaluation for the pretrained-ability-recovery study.

New file. Nothing in the existing pipeline is modified; this only reads the model
and the datasets. Generation reuses the same pattern as
Identification_and_Reversion/generate_and_score.py (the Qwen3-VL M-RoPE handling and
left-padding are load-bearing, so they are kept identical).

Scoring:
  OCRBench  -- official rule: a sample is correct if any acceptable answer appears in
               the prediction, case-insensitively. Score is the count out of 1000.
  MMBench   -- multiple choice. The answer letter is extracted with a cascade of
               increasingly lenient patterns; if none matches, the sample counts as a
               PARSE FAILURE and is scored wrong, but is also reported separately so
               "answered in the wrong format" can be told apart from "answered wrongly".
"""
from __future__ import annotations
import ast, re, time
import torch

OPTIONS = ["A", "B", "C", "D"]   # official MMBench is A-D


# --------------------------------------------------------------------------- prompts
def ocrbench_prompt(row):
    return row["question"]


def mmbench_prompt(row, force_letter=False):
    """Only offer options that actually exist -- some rows have C/D as 'nan'."""
    opts = [(k, row[k]) for k in OPTIONS
            if row.get(k) is not None and str(row[k]).strip().lower() not in ("", "nan", "none")]
    lines = []
    hint = row.get("hint")
    if hint and str(hint).strip().lower() not in ("", "nan", "none"):
        lines.append(str(hint).strip())
    lines.append(str(row["question"]).strip())
    for k, v in opts:
        lines.append(f"{k}. {v}")
    lines.append("Answer with the option's letter from the given choices directly."
                 if not force_letter else
                 "Reply with a single letter and nothing else.")
    return "\n".join(lines), [k for k, _ in opts]


# --------------------------------------------------------------------------- generation
# same cap the DocVQA path uses (Training_Dora/qa_data.MAX_PIXELS). Without it a
# large benchmark image expands to tens of thousands of vision tokens and the batch
# OOMs -- which is exactly how the first run died.
MAX_PIXELS = 1048576


@torch.no_grad()
def generate(model, processor, items, batch_size=4, max_new_tokens=32, device="cuda:0",
             log_every=0, max_pixels=MAX_PIXELS):
    """items: list of (PIL image, prompt string). Returns list of decoded strings."""
    from qwen_vl_utils import process_vision_info
    model.eval()
    gen_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    prev = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    out = []
    try:
        for s in range(0, len(items), batch_size):
            chunk = items[s:s + batch_size]
            texts, convs = [], []
            for img, prompt in chunk:
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": img, "max_pixels": max_pixels},
                    {"type": "text", "text": prompt}]}]
                texts.append(processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True))
                convs.append(msgs)
            images, _ = process_vision_info(convs)
            inputs = processor(text=texts, images=images, padding=True,
                               return_tensors="pt").to(device)
            inputs.pop("mm_token_type_ids", None)
            gen = gen_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, use_cache=True)
            trimmed = gen[:, inputs["input_ids"].shape[1]:]
            out.extend(p.strip() for p in
                       processor.batch_decode(trimmed, skip_special_tokens=True))
            if log_every and (s // batch_size) % log_every == 0:
                print(f"      {min(s+batch_size, len(items))}/{len(items)}", flush=True)
    finally:
        processor.tokenizer.padding_side = prev
    return out


# --------------------------------------------------------------------------- scoring
# Scoring is delegated to scoring_official.py, which is a faithful port of the
# upstream implementations (MultimodalOCR/OCRBench/example.py and
# VLMEvalKit/vlmeval/utils/matching_util.py). Do not "improve" those rules: the point
# of matching them is comparability with published numbers.
from Pretrained_Ability.scoring_official import (
    ocrbench_correct, can_infer, can_infer_option, can_infer_text, OCRBENCH_COMPONENTS)


def _golds(ans):
    """OCRBench answers arrive as a stringified list, or occasionally a bare string."""
    if isinstance(ans, list):
        return [str(a) for a in ans]
    s = str(ans)
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            return [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
        except Exception:
            pass
    return [s]


def score_ocrbench(preds, rows):
    """Official rule; reported out of 1000 with the five component subtotals."""
    by_type, empty, correct = {}, 0, 0
    for p, r in zip(preds, rows):
        if not str(p or "").strip():
            empty += 1
        hit = ocrbench_correct(p, _golds(r["answer"]), r.get("dataset", ""))
        correct += hit
        t = r.get("question_type", "?")
        a, b = by_type.get(t, (0, 0))
        by_type[t] = (a + hit, b + 1)
    comps = {}
    for name, (types, total) in OCRBENCH_COMPONENTS.items():
        got = sum(by_type.get(t, (0, 0))[0] for t in types)
        n = sum(by_type.get(t, (0, 0))[1] for t in types)
        comps[name] = {"score": got, "max": total, "n_seen": n}
    return {
        "score": correct,                  # OCRBench total, out of 1000
        "n": len(preds),
        "accuracy": correct / max(1, len(preds)),
        "empty_output_rate": empty / max(1, len(preds)),
        "components": comps,
        "per_type": {k: {"correct": v[0], "n": v[1], "acc": v[0] / v[1]}
                     for k, v in sorted(by_type.items())},
    }


def score_mmbench(preds, rows):
    """Official can_infer cascade. A prediction can_infer cannot resolve is a PARSE
    FAILURE: scored wrong, but counted separately so format loss can be told apart
    from ability loss."""
    correct = unparsed = refusal = 0
    by_cat, routes = {}, {}
    for p, r in zip(preds, rows):
        _, valid = mmbench_prompt(r)
        choices = {k: r[k] for k in valid}
        # can_infer == can_infer_option else can_infer_text; resolve the two halves
        # separately so the extraction ROUTE can be reported. Scoring is unchanged.
        opt = can_infer_option(str(p), dict(choices))
        got = opt if opt else can_infer_text(str(p), dict(choices))
        route = "option" if opt else ("text" if got else "unparsed")
        if got == "Z":
            route = "refusal"
        routes[route] = routes.get(route, 0) + 1
        if got == "Z":
            refusal += 1
        if got is False or got == "Z":
            unparsed += 1
            hit = False
        else:
            hit = (str(got).upper() == str(r["answer"]).strip().upper())
        correct += hit
        cat = r.get("L2-category") or r.get("category") or "?"
        a, b, u = by_cat.get(cat, (0, 0, 0))
        by_cat[cat] = (a + hit, b + 1, u + (got is False or got == "Z"))
    n = max(1, len(preds))
    return {
        "accuracy": correct / n,
        "correct": correct,
        "n": len(preds),
        "parse_fail_rate": unparsed / n,
        "parse_fail_n": unparsed,
        "refusal_n": refusal,
        "extraction_route": routes,
        "eval_mode": "vanilla (single pass, not circular)",
        "per_ability": {k: {"acc": v[0] / v[1], "n": v[1], "parse_fail": v[2] / v[1]}
                        for k, v in sorted(by_cat.items())},
    }


# --------------------------------------------------------------------------- runners
def run_ocrbench(model, processor, ds, batch_size=4, device="cuda:0", limit=None):
    rows = list(ds)[:limit] if limit else list(ds)
    items = [(r["image"], ocrbench_prompt(r)) for r in rows]
    t0 = time.time()
    preds = generate(model, processor, items, batch_size=batch_size,
                     max_new_tokens=32, device=device, log_every=20)
    m = score_ocrbench(preds, rows)
    m["seconds"] = round(time.time() - t0, 1)
    return m, preds


def run_mmbench(model, processor, ds, batch_size=4, device="cuda:0", limit=None,
                force_letter=False):
    rows = list(ds)[:limit] if limit else list(ds)
    items = []
    for r in rows:
        prompt, _ = mmbench_prompt(r, force_letter=force_letter)
        items.append((r["image"], prompt))
    t0 = time.time()
    preds = generate(model, processor, items, batch_size=batch_size,
                     max_new_tokens=16, device=device, log_every=40)
    m = score_mmbench(preds, rows)
    m["seconds"] = round(time.time() - t0, 1)
    m["force_letter_prompt"] = force_letter
    return m, preds
