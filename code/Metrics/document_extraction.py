"""
document_extraction.py — structured document extraction: training target, prompt, metrics.

The TRAINING TARGET is the full gt_parse field tree (image -> nested fields), which is
where the real headroom is: zero-shot the model reads values fine (value-F1 ~0.78)
but cannot produce the target schema (schema-F1 ~0). Fine-tuning teaches the schema, a
large real ID gain — exactly what DocVQA lacked.

Two metrics, deliberately separate:
  schema-F1 : field F1 that REQUIRES the right key (the field name) AND value.
              This is the task the model must learn; ~0 before fine-tuning.
  value-F1  : value-only F1 (keys ignored). Reading ability; ~0.78 before tuning.

The prompt is FIXED and shared by training and eval, so the model is trained on
exactly the format it is tested on.
"""
from __future__ import annotations

import json
import os
import re

import torch

from Metrics.text_metrics import anls
from Training_Dora.qa_data import MAX_PIXELS, mask_and_batch, raw_messages

# generic on purpose: the model is NOT told the schema, so schema-F1 starts ~0
# and rises only once fine-tuning teaches the field structure. The default is
# a generic "document" instruction (model verified invariant to the noun: bank
# field-F1 0.715 receipt vs 0.728 document). Override via EXTRACT_INSTRUCTION env.
EXTRACT_INSTRUCTION = os.environ.get(
    "EXTRACT_INSTRUCTION", "Extract all fields from this document as JSON.")


def serialize_gt(gt_parse):
    """Canonical JSON target string the model is trained to reproduce."""
    return json.dumps(gt_parse, ensure_ascii=False)


def extract_messages(image_path, max_pixels=MAX_PIXELS):
    return raw_messages(image_path, EXTRACT_INSTRUCTION, max_pixels)


class ExtractionCollator:
    """Document training batch: target = the serialized gt_parse field tree."""

    def __init__(self, processor, max_pixels=MAX_PIXELS):
        self.processor = processor
        self.max_pixels = max_pixels

    def __call__(self, rows):
        items = [(extract_messages(r["image"], self.max_pixels),
                  serialize_gt(r["gt_parse"])) for r in rows]
        return mask_and_batch(self.processor, items, self.max_pixels)


def load_documents(gt_path, limit=None):
    """gt_parse.jsonl -> rows with absolute image path + gt_parse dict."""
    base = os.path.dirname(os.path.abspath(gt_path))
    img_dir = os.path.join(base, "images")
    rows = []
    with open(gt_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            img = r["image"]
            if not os.path.isabs(img):
                # manifests store "images/x.png"; some store just "x.png"
                cand = os.path.join(base, img)
                r["image"] = cand if os.path.exists(cand) else os.path.join(img_dir,
                                                                            os.path.basename(img))
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def parse_json(text):
    """Grab the outermost {...} from the model output. Unparseable -> {} (scored as
    all false negatives, the honest outcome)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return {}


def value_f1(pred_dict, gold_dict, threshold=0.5):
    """Reading diagnostic: value-only F1 (keys ignored), ANLS value match. Reflects
    whether the model READ the numbers, independent of whether it learned the schema."""
    from Metrics.field_f1 import flatten_fields
    pv = [v for _, v in flatten_fields(pred_dict)]
    gv = [v for _, v in flatten_fields(gold_dict)]
    used = [False] * len(pv)
    tp = 0
    for g in gv:
        for j, p in enumerate(pv):
            if not used[j] and anls(p, g, threshold) >= threshold:
                used[j] = True
                tp += 1
                break
    return tp, len(pv), len(gv)


@torch.no_grad()
def generate_extractions(model, processor, rows, batch_size=8, max_new_tokens=512,
                         max_pixels=MAX_PIXELS, device="cuda:0"):
    """Raw JSON strings the model produces for each document."""
    from qwen_vl_utils import process_vision_info

    model.eval()
    gen_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    prev = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    outs = []
    try:
        for s in range(0, len(rows), batch_size):
            chunk = rows[s:s + batch_size]
            texts, convs = [], []
            for r in chunk:
                msgs = extract_messages(r["image"], max_pixels)
                texts.append(processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True))
                convs.append(msgs)
            images, _ = process_vision_info(convs)
            inputs = processor(text=texts, images=images, padding=True,
                               return_tensors="pt").to(device)
            inputs.pop("mm_token_type_ids", None)
            _eos = [processor.tokenizer.eos_token_id]
            _im = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if isinstance(_im, int) and _im >= 0 and _im not in _eos:
                _eos.append(_im)
            gen = gen_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, use_cache=True,
                                     eos_token_id=_eos,
                                     pad_token_id=processor.tokenizer.pad_token_id)
            trimmed = gen[:, inputs["input_ids"].shape[1]:]
            outs.extend(processor.batch_decode(trimmed, skip_special_tokens=True))
    finally:
        processor.tokenizer.padding_side = prev
    return outs


def evaluate_extraction(model, processor, rows, batch_size=8, max_new_tokens=512,
                        max_pixels=MAX_PIXELS, device="cuda:0"):
    """Official field-F1 + nTED (from field_f1) and value-F1 diagnostic."""
    from Metrics.field_f1 import corpus_field_f1, corpus_nted

    outs = generate_extractions(model, processor, rows, batch_size, max_new_tokens,
                                max_pixels, device)
    preds = [parse_json(o) for o in outs]
    golds = [r.get("gt_parse", {}) for r in rows]

    tp = n_pred = n_gold = 0
    for p, g in zip(preds, golds):
        t, npd, ngd = value_f1(p, g)
        tp += t; n_pred += npd; n_gold += ngd
    vprec = tp / n_pred if n_pred else 0.0
    vrec = tp / n_gold if n_gold else 0.0
    vf1 = 2 * vprec * vrec / (vprec + vrec) if (vprec + vrec) else 0.0

    return {"field_f1": corpus_field_f1(preds, golds),
            "nted": corpus_nted(preds, golds),
            "value_f1": vf1, "n": len(rows)}
