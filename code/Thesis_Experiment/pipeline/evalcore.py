"""Shared evaluation core: load a model, apply a reversion, score datasets.

Every eval in every experiment goes through here, so the prompt, the batch size,
the token cap and the adapter-application path cannot drift apart between scripts
(which is exactly what happened between eval_gen.py and sweep_b2q.py in the first run).

The prompt is taken from the dataset registry, never hardcoded.
"""
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import Metrics.document_extraction as _de
from Metrics.document_extraction import evaluate_extraction, load_documents
from Identification_and_Reversion.dora_merge import DoRAAdapter, wiseft, VERSION_FNS
from Identification_and_Reversion.dora_reversion import (revert_both, revert_direction,
                                                         revert_magnitude)
from Thesis_Experiment.datasets import manifest, prompt

DEFAULT_CAP = 4096
DEFAULT_BATCH = 16
MAX_PIXELS = 589824


def load(base, device="cuda:0"):
    proc = AutoProcessor.from_pretrained(base)
    model = Qwen3VLForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16).to(device)
    model.eval()
    return model, proc


def attach(model, adapter, device="cuda:0"):
    """Load the DoRA adapter next to the model (weights are written by apply_reversion)."""
    return DoRAAdapter.load(adapter, model, device) if adapter else None


def apply_reversion(ad, model, alpha_mag=1.0, beta_dir=1.0):
    """alpha/beta = 1 keeps the trained component, 0 reverts it to the base model.

    Dispatch to the SINGLE-component function whenever only one dial is moved.
    revert_both is mathematically identical there, but it renormalises a blend that
    is already unit length, and those fp32 rounding differences survive the cast to
    bf16 (~8 mantissa bits) as ~2e-4 weight differences -- enough for greedy decoding
    to emit a different digit and change an exact-match field. Using the specific
    function keeps results bit-identical to the original drivers.
    """
    if ad is None:
        return
    if alpha_mag == 1.0 and beta_dir == 1.0:
        # CANONICAL full-DoRA weights: m_ft * D_ft, exactly what PEFT computes.
        # Do NOT reach this point via revert_both/revert_direction: at dial=1 they
        # renormalise an already-unit vector, and that fp32 no-op survives the bf16
        # cast as ~2e-4 weight noise. In the first run this made the SAME model score
        # quarterly 0.4518 via the direction sweep and 0.4307 via the magnitude sweep.
        ad.apply(model, VERSION_FNS["full"])
    elif alpha_mag == 1.0:
        ad.apply(model, lambda l: revert_direction(l, beta_dir))  # direction only
    elif beta_dir == 1.0:
        ad.apply(model, lambda l: revert_magnitude(l, alpha_mag))  # magnitude only
    else:
        ad.apply(model, lambda l: revert_both(l, alpha_mag, beta_dir))


def apply_wiseft(ad, model, coeff):
    """WiSE-FT baseline: interpolate the WHOLE merged weight toward the base,
    with no magnitude/direction split. coeff=1 keeps the fine-tuned weight,
    coeff=0 returns the base. This is the dial our selective reversion competes with."""
    if ad is not None:
        ad.apply(model, wiseft(coeff))


def score(model, proc, dataset, split, device="cuda:0",
          cap=DEFAULT_CAP, batch=DEFAULT_BATCH, limit=None):
    """Score one dataset with ITS OWN registry prompt. Returns metrics + wall seconds."""
    rows = load_documents(manifest(dataset, split))
    if limit:
        rows = rows[:limit]
    _de.EXTRACT_INSTRUCTION = prompt(dataset)
    t0 = time.time()
    m = evaluate_extraction(model, proc, rows, batch_size=batch, max_new_tokens=cap,
                            max_pixels=MAX_PIXELS, device=device)
    secs = time.time() - t0
    return {
        "field_f1": round(m["field_f1"], 4),
        "value_f1": round(m["value_f1"], 4) if m.get("value_f1") is not None else None,
        "n": len(rows),
        "eval_seconds": round(secs, 1),
        "seconds_per_doc": round(secs / max(1, len(rows)), 2),
        "prompt": prompt(dataset),
    }
