"""
eval_lib.py — batched generation + ANLS scoring against an ALREADY-LOADED model.

eval_baseline.py reloads the model and generates one sample at a time, which is
fine for a single baseline but far too slow for the reversion matrix (4 versions
x {ID, OOD} x an alpha sweep). Here the 8B is loaded once and the caller mutates
its weights between evaluations.
"""
from __future__ import annotations

import json
import os

import torch

from Metrics.text_metrics import corpus_anls, corpus_exact_match
from Training_Dora.qa_data import MAX_PIXELS, build_messages


@torch.no_grad()
def generate_answers(model, processor, rows, batch_size=8, max_new_tokens=32,
                     max_pixels=MAX_PIXELS, device="cuda:0", log_every=0):
    from qwen_vl_utils import process_vision_info

    model.eval()

    # Generate through the underlying Qwen model, not PEFT's wrapper. Qwen3-VL
    # REQUIRES mm_token_type_ids (for M-RoPE) but PeftModel.generate rejects it as
    # an unknown kwarg. get_base_model() returns the same module tree with the LoRA
    # layers still injected, so the adapter stays active.
    gen_model = model.get_base_model() if hasattr(model, "get_base_model") else model

    # left padding: generation must continue from the true end of each prompt
    prev_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    preds = []
    try:
        for s in range(0, len(rows), batch_size):
            chunk = rows[s:s + batch_size]
            texts, convs = [], []
            for r in chunk:
                msgs = build_messages(r["image"], r["question"], max_pixels)
                texts.append(processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True))
                convs.append(msgs)
            # flat image list over the batch (see DocVQACollator)
            images, _ = process_vision_info(convs)
            inputs = processor(text=texts, images=images, padding=True,
                               return_tensors="pt").to(device)
            # a training-only field; generate() handles M-RoPE itself. Verified to give
            # identical output whether passed or dropped (on an unpatched forward).
            inputs.pop("mm_token_type_ids", None)
            gen = gen_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, use_cache=True)
            trimmed = gen[:, inputs["input_ids"].shape[1]:]
            preds.extend(p.strip() for p in
                         processor.batch_decode(trimmed, skip_special_tokens=True))
            if log_every and (s // batch_size) % log_every == 0:
                print(f"    {min(s+batch_size, len(rows))}/{len(rows)}", flush=True)
    finally:
        processor.tokenizer.padding_side = prev_side

    return preds


def score(preds, rows):
    golds = [r["answers"] if isinstance(r["answers"], list) else [r["answers"]]
             for r in rows]
    return {"anls": corpus_anls(preds, golds),
            "exact_match": corpus_exact_match(preds, golds),
            "n": len(preds)}


def evaluate(model, processor, rows, **kw):
    preds = generate_answers(model, processor, rows, **kw)
    return score(preds, rows), preds


def save_run(out_dir, metrics, rows=None, preds=None, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(metrics)
    if extra:
        payload.update(extra)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)
    if rows is not None and preds is not None:
        with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
            for r, p in zip(rows, preds):
                f.write(json.dumps({"image": r["image"], "question": r["question"],
                                    "pred": p, "gold": r["answers"]},
                                   ensure_ascii=False) + "\n")
    return out_dir
