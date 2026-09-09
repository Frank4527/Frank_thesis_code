"""
vlm_data.py — shared data/prompt plumbing for Qwen3-VL DocVQA experiments.

One place defines the prompt, so training and evaluation can never drift apart:
the training text is literally the eval prompt + the answer. If these disagree
the model is trained on one format and tested on another, which silently
destroys the score — so it is built once, here.

Loss is computed on the ANSWER TOKENS ONLY (prompt tokens are masked to -100).
"""
from __future__ import annotations

import json
import os

import torch

# DocVQA convention: terse extractive answer. Must match eval_baseline.py.
INSTRUCTION = "Answer the question using a single word or phrase."
MAX_PIXELS = 1048576


def load_manifest(path, limit=None):
    rows = []
    base = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not os.path.isabs(r["image"]):
                r["image"] = os.path.join(base, r["image"])
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def raw_messages(image_path, user_text, max_pixels=MAX_PIXELS):
    """One image + a verbatim user text (no instruction appended)."""
    return [{"role": "user", "content": [
        {"type": "image", "image": "file://" + image_path, "max_pixels": max_pixels},
        {"type": "text", "text": user_text},
    ]}]


def build_messages(image_path, question, max_pixels=MAX_PIXELS):
    """QA user turn — identical at train and eval time (question + instruction)."""
    return raw_messages(image_path, question + "\n" + INSTRUCTION, max_pixels)


def prompt_text(processor, image_path, question, max_pixels=MAX_PIXELS):
    """Chat-templated prompt ending in the assistant header (generation-ready)."""
    return processor.apply_chat_template(
        build_messages(image_path, question, max_pixels),
        tokenize=False, add_generation_prompt=True)


def load_images(processor, rows, max_pixels=MAX_PIXELS):
    from qwen_vl_utils import process_vision_info
    msgs = [build_messages(r["image"], r["question"], max_pixels) for r in rows]
    imgs, _ = process_vision_info([m for m in msgs])
    return imgs


def mask_and_batch(processor, items, max_pixels=MAX_PIXELS):
    """Encode (messages, target) pairs into a batch with TARGET-only loss.

    Shared by every training task: the full text is prompt + target + eos, so the
    prompt is a guaranteed prefix and the target occupies the last k real tokens.
    Masking by attention-mask position is correct for either padding side.
    `items` is a list of (messages, target_text).
    """
    from qwen_vl_utils import process_vision_info

    tok = processor.tokenizer
    eos = tok.eos_token or "<|im_end|>"
    texts, convs, k_lens = [], [], []
    for msgs, target in items:
        p = processor.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
        suffix = target + eos
        texts.append(p + suffix)
        convs.append(msgs)
        k_lens.append(len(tok(suffix, add_special_tokens=False).input_ids))

    # one call over the whole batch -> FLAT image list, so pixel_values and
    # image_grid_thw stay consistent (a nested list silently desyncs them)
    images, _ = process_vision_info(convs)
    batch = processor(text=texts, images=images, padding=True, return_tensors="pt")

    am, ids = batch["attention_mask"], batch["input_ids"]
    labels = torch.full_like(ids, -100)
    for i, k in enumerate(k_lens):
        real = am[i].nonzero(as_tuple=True)[0]
        if k > 0 and len(real):
            labels[i, real[-k:]] = ids[i, real[-k:]]
    batch["labels"] = labels
    return batch


class DocVQACollator:
    """QA training batch with answer-only loss (target = the gold answer)."""

    def __init__(self, processor, max_pixels=MAX_PIXELS):
        self.processor = processor
        self.max_pixels = max_pixels

    def __call__(self, rows):
        items = []
        for r in rows:
            ans = r["answers"][0] if isinstance(r["answers"], list) else r["answers"]
            items.append((build_messages(r["image"], r["question"], self.max_pixels),
                          ans))
        return mask_and_batch(self.processor, items, self.max_pixels)
