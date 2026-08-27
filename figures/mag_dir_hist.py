"""Histogram of the magnitude change vs the direction update induced by DoRA
fine-tuning, per output unit, in raw weight-norm units (nothing normalised).

magnitude change : m_ft - ||W0||          (trained magnitude minus base magnitude)
direction update : ||s * B @ A||_row       (raw low-rank update, NOT the unit direction)

Both are one number per output unit; pooled over all 252 adapted layers.
"""
import os
os.environ.setdefault("EXTRACT_INSTRUCTION", "Extract all fields from this document as JSON.")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import Qwen3VLForConditionalGeneration
from Identification_and_Reversion.dora_merge import DoRAAdapter
from Identification_and_Reversion.dora_reversion import column_norm

import sys
_TARGETS = {
 "de2po": ("Bankstatements_Lang2Lang","merged/base_de_8b_seed0",
           "adapters/stage2/stage2_de2po_8b_seed0/adapter_best", r"de$\rightarrow$po"),
 "bank2quarterly": ("Bankstatements_Bank2Quarterly","merged/base_bank_full_8b_seed0",
           "adapters/stage2/stage2_quarterly_8b_seed0/adapter_last", r"bank$\rightarrow$quarterly"),
}
KEY = sys.argv[1] if len(sys.argv) > 1 else "de2po"
_exp,_b,_a,LABEL = _TARGETS[KEY]
SEQ = f"/home/frank/runs/thesis_experiment_runs/{_exp}"
BASE = f"{SEQ}/{_b}"
ADAPTER = f"{SEQ}/{_a}"
dev = "cuda:0"
os.makedirs(f"{SEQ}/results/stage2/figures", exist_ok=True)

model = Qwen3VLForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16).to(dev); model.eval()
ad = DoRAAdapter.load(ADAPTER, model, dev)

dm_all, ddir_all = [], []
for path in ad.layers:
    L = ad.layer(path)                       # fp32 on GPU
    m0   = column_norm(L.W0).flatten()       # ||W0|| per output unit  (base magnitude)
    m_ft = L.m_ft.flatten()                  # trained magnitude vector
    dm   = (m_ft - m0)                        # magnitude CHANGE per unit (raw)
    ddir = L.delta.norm(dim=1)               # ||s*B@A|| per output unit (raw direction update)
    dm_all.append(dm.detach().cpu().numpy())
    ddir_all.append(ddir.detach().cpu().numpy())
    del L
dm  = np.concatenate(dm_all)
ddir = np.concatenate(ddir_all)
np.savez(f"{SEQ}/results/stage2/figures/mag_dir_change_{KEY}.npz", dm=dm, ddir=ddir)

def stat(x):
    a = np.abs(x)
    return f"median={np.median(a):.4f} mean={a.mean():.4f} p95={np.percentile(a,95):.4f} max={a.max():.4f}"
print("n output units :", dm.size)
print("MAGNITUDE  |m_ft - ||W0|||   :", stat(dm))
print("DIRECTION  ||s*B@A||_row     :", stat(ddir))

# ---- two histograms, raw units, SHARED x-axis so they are directly comparable ----
xmax = float(np.percentile(ddir, 99))
bins = np.linspace(0, xmax, 80)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True, sharey=True)
ax[0].hist(np.abs(dm), bins=bins, color="#2563eb")
ax[0].axvline(np.median(np.abs(dm)), color="k", ls="--", lw=1,
              label=f"median = {np.median(np.abs(dm)):.4f}")
ax[0].set_title(r"Magnitude change  $|\,m_{ft}-\|W_0\|\,|$", color="#2563eb")
ax[1].hist(ddir, bins=bins, color="#ef4444")
ax[1].axvline(np.median(ddir), color="k", ls="--", lw=1,
              label=f"median = {np.median(ddir):.4f}")
ax[1].set_title(r"Direction update  $\|s\,B A\|_{row}$  (un-normalised)", color="#ef4444")
for a in ax:
    a.set_xlabel("change, raw weight-norm units"); a.set_yscale("log")
    a.grid(alpha=.25); a.legend()
ax[0].set_ylabel("number of output units (log)")
fig.suptitle(f"{LABEL} stage-2 DoRA: per-output-unit change, magnitude vs direction "
             f"({dm.size:,} units over 252 layers)", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
p = f"{SEQ}/results/stage2/figures/mag_dir_hist_{KEY}.png"
fig.savefig(p, dpi=140); print("saved", p)
