"""Per-output-row drift of BOTH DoRA components, kept per layer.

For every adapted layer i and output row j:
    m0   = ||W0||           base magnitude of the row
    dm   = m_ft - m0        magnitude change            (signed)
    ddir = ||s * (B@A)||    direction update            (norm, >= 0)
    rho  = ddir / m0        direction update relative to the row it sits on

extract_mag_change.py computes the same three quantities but concatenates them
across layers, so layer identity is lost. This keeps it. Two uses:

  1. the direction/magnitude comparison histogram -- same numbers as the
     existing figure, now with the direction side available per layer too
  2. layer-targeted reversion, which needs a per-layer score to rank on:
     the median rho over the rows of each layer

Runs on CPU by default so it can share the box with GPU evaluation.

    python layer_drift.py                    # all six runs
    python layer_drift.py --runs b2q_seed1   # one
    python layer_drift.py --check            # verify vs the existing _magchange npz
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="all")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="/home/frank/runs/thesis_experiment_runs/_layerdrift")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--check", action="store_true", help="compare against _magchange/*.npz")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

if args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""       # never touch a GPU by accident
os.environ.setdefault("EXTRACT_INSTRUCTION", "Extract all fields from this document as JSON.")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np, torch
torch.set_num_threads(args.threads)
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
if args.runs != "all":
    keep = set(args.runs.split(","))
    RUNS = [r for r in RUNS if r[0] in keep]
os.makedirs(args.out, exist_ok=True)


def describe(path):
    """Split a module path into the fields the reversion will want to select on."""
    parts = path.split(".")
    block = next((int(parts[i + 1]) for i, p in enumerate(parts)
                  if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit()), -1)
    return {"block": block,
            "proj": parts[-1],                                   # q_proj, down_proj, ...
            "tower": "visual" if "visual" in parts else "language"}


for key, exp, base, adp in RUNS:
    dst = f"{args.out}/layer_drift_{key}.npz"
    if os.path.exists(dst) and not args.force:
        print(f"  SKIP {key} (exists)"); continue
    BASE = f"{R}/{exp}/merged/{base}"
    ADAPTER = f"{R}/{exp}/adapters/stage2/{adp}/adapter_last"
    if not (os.path.isdir(BASE) and os.path.isdir(ADAPTER)):
        print(f"  MISSING inputs for {key}"); continue

    print(f"  {key}: loading {os.path.basename(BASE)} on {args.device}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16)
    if args.device != "cpu":
        model = model.to(args.device)
    model.eval()
    ad = DoRAAdapter.load(ADAPTER, model, args.device)

    names, rows_per_layer, summary = [], [], []
    dm_all, ddir_all, m0_all, dperp_all, dpar_all = [], [], [], [], []
    for path in ad.layers:                      # dict order == safetensors order, stable
        L = ad.layer(path)
        m0 = column_norm(L.W0).flatten()
        dm = (L.m_ft.flatten() - m0)
        ddir = L.delta.norm(dim=1)

        # sBA is the update to the WHOLE weight, so part of it points along W0 and
        # only changes the row norm -- which DoRA renormalises away, so it never
        # reaches the direction. Split it: the perpendicular part is the only part
        # that turns D, and ||perp|| / ||W0|| is tan(theta) for the rotation angle.
        u = L.W0 / m0.reshape(-1, 1)
        par = (L.delta * u).sum(dim=1)                      # signed component along W0
        dperp = (L.delta - par.reshape(-1, 1) * u).norm(dim=1)
        rho = (ddir / m0).cpu().numpy()

        names.append(path)
        rows_per_layer.append(int(m0.numel()))
        m0_all.append(m0.detach().cpu().numpy())
        dm_all.append(dm.detach().cpu().numpy())
        ddir_all.append(ddir.detach().cpu().numpy())
        dperp_all.append(dperp.detach().cpu().numpy())
        dpar_all.append(par.detach().cpu().numpy())
        rel_mag = np.abs(dm_all[-1]) / m0_all[-1]
        rho_perp = dperp_all[-1] / m0_all[-1]
        summary.append({"layer": path, **describe(path), "n_rows": rows_per_layer[-1],
                        "rho_median": float(np.median(rho)),
                        "rho_mean": float(rho.mean()),
                        "rho_p90": float(np.percentile(rho, 90)),
                        "rho_perp_median": float(np.median(rho_perp)),
                        "angle_deg_median": float(np.degrees(np.arctan(np.median(rho_perp)))),
                        "mag_abs_median": float(np.median(rel_mag))})
        del L

    m0 = np.concatenate(m0_all); dm = np.concatenate(dm_all); ddir = np.concatenate(ddir_all)
    dperp = np.concatenate(dperp_all); dpar = np.concatenate(dpar_all)
    row_layer = np.repeat(np.arange(len(names), dtype=np.int32), rows_per_layer)
    np.savez(dst, m0=m0, dm=dm, ddir=ddir, dperp=dperp, dpar=dpar, row_layer=row_layer,
             layer_names=np.array(names), rows_per_layer=np.array(rows_per_layer))
    json.dump({"run": key, "experiment": exp, "adapter": ADAPTER,
               "n_layers": len(names), "n_rows": int(m0.size), "layers": summary},
              open(f"{args.out}/layer_drift_{key}.json", "w"), indent=2)

    rho_all, rho_perp = ddir / m0, dperp / m0
    print(f"    {len(names)} layers, {m0.size:,} rows | "
          f"median rho {np.median(rho_all) * 100:.4f}%  "
          f"perp {np.median(rho_perp) * 100:.4f}% "
          f"({100 * np.median(dperp / ddir):.2f}% of the update is perpendicular, "
          f"i.e. a rotation of {np.degrees(np.arctan(np.median(rho_perp))):.3f} deg)  "
          f"median |dm|/m0 {np.median(np.abs(dm) / m0) * 100:.4f}%  "
          f"ratio {np.median(rho_all) / np.median(np.abs(dm) / m0):.1f}x", flush=True)

    if args.check:
        old = f"{R}/_magchange/mag_dir_change_{key}.npz"
        if os.path.exists(old):
            o = np.load(old)
            # dm = m_ft - ||W0|| is a difference of two nearly equal fp32 numbers, so
            # its RELATIVE error is meaningless; compare absolute error against the
            # size of the effect being reported instead.
            err = max(float(np.abs(o[k] - v).max()) for k, v in
                      (("m0", m0), ("dm", dm), ("ddir", ddir)))
            eff = float(np.median(np.abs(dm)))
            print(f"    CHECK vs _magchange (GPU run): max abs diff {err:.2e}, "
                  f"median |dm| {eff:.2e} -- effect is {eff / err:.0f}x the numerical "
                  f"difference between the two extractions")
        else:
            print("    CHECK: no _magchange file for this run")

    del ad, model
    if args.device != "cpu":
        torch.cuda.empty_cache()

print("done")
