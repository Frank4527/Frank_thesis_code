"""Characterise the bf16 dial-1 effect properly.

My first attempt used a 64x128 layer: the fp32 discrepancy (~5e-7) is far below bf16
rounding granularity, so both routes rounded to the SAME bf16 value and the test
'failed'. The real effect is a small FRACTION of weights straddling a rounding
boundary, which only shows at realistic scale.
"""
import sys
sys.path.insert(0, "/home/frank/code")
import torch
from Identification_and_Reversion.dora_reversion import (
    column_norm, DoRALayer, build_version, revert_direction)

torch.manual_seed(0)

def mk(out, inn, r=16, alpha=32):
    W0 = torch.randn(out, inn) * 0.02          # realistic weight scale
    A = torch.randn(r, inn) * 0.01
    B = torch.randn(out, r) * 0.01
    m_ft = column_norm(W0).clone() * (1 + 0.05 * torch.randn(out, 1))
    return DoRALayer(W0=W0, A=A, B=B, scaling=alpha / r, m_ft=m_ft)

print(f"{'layer size':>18s} {'fp32 max|d|':>12s} {'bf16 differing':>15s} {'bf16 max|d|':>12s}")
tot_diff = tot_el = 0
for (o, i) in [(64, 128), (512, 512), (4096, 4096)]:
    L = mk(o, i)
    full = build_version(L, True, True)
    rev = revert_direction(L, 1.0)
    d32 = (full - rev).abs().max().item()
    f16 = full.to(torch.bfloat16)
    r16 = rev.to(torch.bfloat16)
    ne = (f16 != r16)
    frac = ne.float().mean().item()
    d16 = (f16.float() - r16.float()).abs().max().item()
    tot_diff += ne.sum().item(); tot_el += ne.numel()
    print(f"{o:>8d}x{i:<9d} {d32:12.3e} {frac*100:14.4f}% {d16:12.3e}")

print()
print(f"overall: {tot_diff:,} of {tot_el:,} weights ({100*tot_diff/tot_el:.4f}%) land on a "
      f"different bf16 value")
print()
print("So the two routes are mathematically identical, agree to ~1e-7 in fp32, and")
print("differ on a small fraction of weights once cast to bf16 -- which is exactly")
print("why evalcore.apply_reversion dispatches dial 1 to the canonical full path")
print("instead of calling revert_direction(1.0).")
print()
# and confirm the guard actually works, on the real dispatcher
from Thesis_Experiment.pipeline import evalcore
import inspect
src = inspect.getsource(evalcore.apply_reversion)
ok = 'VERSION_FNS["full"]' in src and "alpha_mag == 1.0 and beta_dir == 1.0" in src
print("guard present in evalcore.apply_reversion:", "YES" if ok else "NO -- CHECK THIS")
