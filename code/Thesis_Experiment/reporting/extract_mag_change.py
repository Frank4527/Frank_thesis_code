"""Per-output-row magnitude and direction change for all six runs.

Computation lifted verbatim from plots/mag_dir_hist.py:
    m0   = column_norm(W0)      ||W0|| per output row  (base magnitude)
    m_ft = L.m_ft               the separately trained magnitude
    dm   = m_ft - m0            raw magnitude change
    ddir = ||s*B@A||_row        raw direction update

Additionally saves m0, which mag_dir_hist.py does not, because the percentage
form (dm/m0*100) used in the report figure needs it.
"""
import os, sys
os.environ.setdefault("EXTRACT_INSTRUCTION", "Extract all fields from this document as JSON.")
import numpy as np, torch
from transformers import Qwen3VLForConditionalGeneration
from Identification_and_Reversion.dora_merge import DoRAAdapter
from Identification_and_Reversion.dora_reversion import column_norm

R = "/home/frank/runs/thesis_experiment_runs"
RUNS = [
    ("b2q_seed1", "Bankstatement2Quaterly_rep_rerun",       "base_bank_full_8b_seed0", "stage2_quarterly_8b_seed0"),
    ("b2q_seed2", "Bankstatement2Quaterly_rep_rerun_seed2", "base_bank_full_8b_seed0", "stage2_quarterly_8b_seed0"),
    ("b2q_seed3", "Bankstatement2Quaterly_rep_rerun_seed3", "base_bank_full_8b_seed0", "stage2_quarterly_8b_seed0"),
    ("q2b_seed1", "Quaterly_rep2Bankstatement_rerun",       "base_quarterly_8b_seed0", "stage2_bank_full_8b_seed0"),
    ("q2b_seed2", "Quaterly_rep2Bankstatement_rerun_seed2", "base_quarterly_8b_seed0", "stage2_bank_full_8b_seed0"),
    ("q2b_seed3", "Quaterly_rep2Bankstatement_rerun_seed3", "base_quarterly_8b_seed0", "stage2_bank_full_8b_seed0"),
]
OUT = "/home/frank/runs/thesis_experiment_runs/_magchange"
os.makedirs(OUT, exist_ok=True)
dev = "cuda:0"

for key, exp, base, adp in RUNS:
    dst = f"{OUT}/mag_dir_change_{key}.npz"
    if os.path.exists(dst):
        print(f"  SKIP {key} (exists)"); continue
    BASE = f"{R}/{exp}/merged/{base}"
    ADAPTER = f"{R}/{exp}/adapters/stage2/{adp}/adapter_last"
    if not (os.path.isdir(BASE) and os.path.isdir(ADAPTER)):
        print(f"  MISSING inputs for {key}"); continue
    print(f"  {key}: loading {os.path.basename(BASE)} + {adp}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16).to(dev)
    model.eval()
    ad = DoRAAdapter.load(ADAPTER, model, dev)

    dm_all, ddir_all, m0_all = [], [], []
    for path in ad.layers:
        L = ad.layer(path)
        m0 = column_norm(L.W0).flatten()
        m_ft = L.m_ft.flatten()
        dm_all.append((m_ft - m0).detach().cpu().numpy())
        ddir_all.append(L.delta.norm(dim=1).detach().cpu().numpy())
        m0_all.append(m0.detach().cpu().numpy())
        del L
    dm = np.concatenate(dm_all); ddir = np.concatenate(ddir_all); m0 = np.concatenate(m0_all)
    np.savez(dst, dm=dm, ddir=ddir, m0=m0)
    rel = dm / m0 * 100.0
    print(f"    rows={dm.size:,}  layers={len(ad.layers)}  "
          f"median |mag %|={np.median(np.abs(rel)):.4f}%  "
          f"within +-1% = {100*np.mean(np.abs(rel)<=1):.1f}%", flush=True)
    del ad, model
    torch.cuda.empty_cache()

print("done")
