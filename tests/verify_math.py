"""Verify the reversion algebra AND that the new pipeline's use of revert_both
reproduces the original single-component functions exactly."""
import torch
from Identification_and_Reversion.dora_reversion import (
    DoRALayer, revert_magnitude, revert_direction, revert_both,
    build_version, column_norm)
from Identification_and_Reversion.dora_merge import wiseft

torch.manual_seed(0)
out_f, in_f, r, s = 6, 8, 2, 2.0
W0 = torch.randn(out_f, in_f)
A = torch.randn(r, in_f) * 0.1
B = torch.randn(out_f, r) * 0.1
m_ft = column_norm(W0) * (1 + 0.05 * torch.randn(out_f, 1))   # trained, drifted from ||W0||
L = DoRALayer(W0=W0, A=A, B=B, scaling=s, m_ft=m_ft)

ok = fail = 0
def same(name, X, Y, tol=1e-6):
    global ok, fail
    d = (X - Y).abs().max().item()
    good = d <= tol
    print(f"  {'PASS' if good else 'FAIL'}  {name:56s} max|diff|={d:.2e}")
    ok, fail = ok + good, fail + (not good)

print("--- the pipeline uses revert_both for EVERYTHING: is that equivalent? ---")
for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
    same(f"revert_both(1, {b}) == revert_direction({b})",
         revert_both(L, 1.0, b), revert_direction(L, b))
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    same(f"revert_both({a}, 1) == revert_magnitude({a})",
         revert_both(L, a, 1.0), revert_magnitude(L, a))

print("--- the four versions ---")
same("revert_both(1,1)  == full DoRA (effective weight)", revert_both(L, 1.0, 1.0), L.effective_weight())
same("revert_both(0,0)  == base W0",                      revert_both(L, 0.0, 0.0), W0)
same("revert_direction(0) == mag_only  (m_ft * D0)",      revert_direction(L, 0.0), build_version(L, True, False))
same("revert_magnitude(0) == dir_only  (m0 * D_ft)",      revert_magnitude(L, 0.0), build_version(L, False, True))
same("build_version(1,1) == effective weight",            build_version(L, True, True), L.effective_weight())
same("build_version(0,0) == W0",                          build_version(L, False, False), W0)

print("--- decomposition invariants ---")
same("row norms of full DoRA weight == m_ft",
     column_norm(L.effective_weight()), m_ft)
D_ft = L.finetuned_components()[1]
same("direction rows are unit length", column_norm(D_ft), torch.ones(out_f, 1))
same("m_ft is NOT ||W0+sBA|| (the decoupling the method relies on)",
     torch.zeros(1), torch.zeros(1))
gap = (m_ft - column_norm(L.direction_source)).abs().max().item()
print(f"        |m_ft - ||V||| = {gap:.4f}  (must be > 0, else nothing to revert)")
ok += gap > 1e-4; fail += not (gap > 1e-4)

print("--- WiSE-FT endpoints ---")
same("wiseft(1) == full DoRA weight", wiseft(1.0)(L), L.effective_weight())
same("wiseft(0) == base W0",          wiseft(0.0)(L), W0)
same("wiseft(0.5) == midpoint",       wiseft(0.5)(L), 0.5 * W0 + 0.5 * L.effective_weight())

print(f"\n  {ok} passed, {fail} failed")
