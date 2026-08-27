"""
dora_merge.py — use dora_reversion.py's tensor algebra on a real PEFT adapter.

dora_reversion.py is GPU-free and unit-tested. This module is the bridge to a live
Qwen3-VL model: it reads a trained DoRA adapter, keeps a copy of the pretrained
weights, and writes a reconstructed weight back into the model in place. It doesn't
decide anything about the method -- it's just the plumbing.

Why writing the merged weight straight in is valid (PEFT 0.19 DoRA):
    weight_norm = ||W0 + s*B@A||   (dim=1, per output unit)
    forward     = (m_ft / weight_norm) * (W0 + s*B@A) @ x
so the effective weight is exactly m_ft * D_ft, a full-rank matrix I can drop into
nn.Linear.weight with no PEFT wrapper at inference. That's the same decomposition
DoRALayer uses, so every version / reversion is just a different (magnitude,
direction) written into the same slot.
"""
from __future__ import annotations

import json
import os

import torch

from Identification_and_Reversion.dora_reversion import DoRALayer, build_version

# the four diagnostic versions, as functions of a DoRALayer
VERSION_FNS = {
    "base":     lambda l: build_version(l, False, False),   # == W0
    "mag_only": lambda l: build_version(l, True,  False),
    "dir_only": lambda l: build_version(l, False, True),
    "full":     lambda l: build_version(l, True,  True),    # == the DoRA model
}


def wiseft(coeff):
    """WiSE-FT: interpolate the WHOLE merged weight, with no magnitude/direction
    split. The baseline my selective reversion has to beat."""
    return lambda l: (1.0 - coeff) * l.W0 + coeff * l.effective_weight()


class DoRAAdapter:
    """A trained DoRA adapter bound to a live model, with W0 kept safe.

    Usage:
        ad = DoRAAdapter.load(adapter_dir, model, device)
        ad.apply(model, VERSION_FNS["full"])      # model now == the DoRA model
        ad.apply(model, lambda l: revert_magnitude(l, 0.5))
    """

    def __init__(self, layers, scaling, W0, device):
        self.layers = layers      # {module_path: {"A","B","m"}}
        self.scaling = scaling
        self.W0 = W0              # {module_path: pretrained weight, on CPU}
        self.device = device

    # -- construction --------------------------------------------------------

    @classmethod
    def load(cls, adapter_dir, model, device):
        from safetensors.torch import load_file

        cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
        if not cfg.get("use_dora"):
            raise SystemExit("adapter was not trained with use_dora=True — "
                             "there is no magnitude to revert")
        r, alpha = cfg["r"], cfg["lora_alpha"]
        scaling = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r

        sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
        layers = {}
        # match markers WITHOUT a trailing dot: lora_A/B keys end in ".lora_A.weight"
        # but the magnitude key is a bare tensor ending in ".lora_magnitude_vector"
        # (no ".weight"), so a trailing-dot match would silently drop every magnitude.
        for key, val in sd.items():
            for marker, field in ((".lora_A", "A"), (".lora_B", "B"),
                                  (".lora_magnitude_vector", "m")):
                if marker in key:
                    path = key.split(marker)[0].removeprefix("base_model.model.")
                    layers.setdefault(path, {})[field] = val
                    break

        incomplete = [p for p, d in layers.items() if {"A", "B", "m"} - set(d)]
        if incomplete:
            raise SystemExit(f"{len(incomplete)} layers missing A/B/m, "
                             f"e.g. {incomplete[:3]}")

        # keep W0: apply() overwrites the live weights, and every version is defined
        # relative to W0, so losing it would be unrecoverable
        W0 = {p: model.get_submodule(p).weight.detach().to("cpu").clone()
              for p in layers}
        gb = sum(w.numel() * w.element_size() for w in W0.values()) / 1e9

        print(f"adapter: {len(layers)} DoRA layers, r={r}, alpha={alpha}, "
              f"scaling={scaling:.4f}; W0 snapshot {gb:.1f} GB (CPU)")
        return cls(layers, scaling, W0, device)

    # -- use -----------------------------------------------------------------

    def layer(self, path):
        """One adapted layer as a DoRALayer, in fp32. fp32 matters: bf16 only
        keeps ~3 decimal digits, and decompose/blend/recompose in bf16 loses
        enough precision to move the metric on its own."""
        d = self.layers[path]
        return DoRALayer(
            W0=self.W0[path].to(self.device, torch.float32),
            A=d["A"].to(self.device, torch.float32),
            B=d["B"].to(self.device, torch.float32),
            scaling=self.scaling,
            m_ft=d["m"].to(self.device, torch.float32).reshape(-1, 1),
            dim=1,
        )

    def apply(self, model, fn):
        """Write fn(DoRALayer) -> W into every adapted layer of the live model."""
        for path in self.layers:
            layer = self.layer(path)
            W = fn(layer)
            mod = model.get_submodule(path)
            mod.weight.data.copy_(W.to(mod.weight.dtype))
            del layer, W
        torch.cuda.empty_cache()

    def apply_selective(self, model, fn_selected, paths, fn_other=None):
        """Like apply(), but fn_selected is used only on layers listed in `paths`;
        every other adapted layer gets fn_other, defaulting to the canonical full
        DoRA weights.

        fn_other MUST default to VERSION_FNS["full"] rather than to a reversion at
        dial 1. The two are algebraically identical, but revert_direction(1.0)
        renormalises an already-unit vector and that fp32 no-op survives the bf16
        cast as ~2e-4 of weight noise -- enough to change a greedily decoded token.
        Untouched layers therefore have to go through the same path they take in
        every other evaluation, or a "reverted nothing" model would not reproduce
        the stage-2 model it is supposed to equal.
        """
        fn_other = fn_other or VERSION_FNS["full"]
        sel = set(paths)
        unknown = sel - set(self.layers)
        if unknown:
            raise SystemExit(f"{len(unknown)} selected paths are not adapted layers, "
                             f"e.g. {sorted(unknown)[:3]}")
        n = 0
        for path in self.layers:
            layer = self.layer(path)
            hit = path in sel
            W = (fn_selected if hit else fn_other)(layer)
            n += hit
            mod = model.get_submodule(path)
            mod.weight.data.copy_(W.to(mod.weight.dtype))
            del layer, W
        torch.cuda.empty_cache()
        return n


def verify_reconstruction(model, adapter_dir, adapter, processor, rows,
                          evaluate_fn, device, tol=0.95):
    """Check that my 'full' merge reproduces the real PEFT DoRA model.

    This is the load-bearing check: every version / reversion is built with the
    same algebra as 'full', so if 'full' doesn't match the real adapter, nothing
    downstream means anything. Raises if the two disagree on more than (1 - tol).
    """
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(model, adapter_dir)
    peft_model.eval()
    _, preds_peft = evaluate_fn(peft_model, processor, rows)
    model = peft_model.unload()      # strip wrappers; base weights were never touched
    del peft_model
    torch.cuda.empty_cache()

    adapter.apply(model, VERSION_FNS["full"])
    _, preds_full = evaluate_fn(model, processor, rows)

    agree = sum(a == b for a, b in zip(preds_peft, preds_full))
    print(f"reconstruction: {agree}/{len(rows)} predictions identical to PEFT")
    if agree < tol * len(rows):
        for a, b in zip(preds_peft, preds_full):
            if a != b:
                print(f"    peft={a!r}  ours={b!r}")
        raise SystemExit(
            "RECONSTRUCTION MISMATCH: the 'full' weight does not reproduce the real "
            "PEFT DoRA model. Every reversion result would be meaningless. Stop.")
    return {"n": len(rows), "agree": agree,
            "peft_preds": preds_peft, "full_preds": preds_full}
