"""Correctness tests for the thesis code, run against the real modules on the server.

These test mathematical invariants and decision logic, not just that files are
unchanged. Anything that fails here would be a real defect in a reported number.
"""
import sys, math, json, itertools
sys.path.insert(0, "/home/frank/code")
import torch

from Identification_and_Reversion.dora_reversion import (
    column_norm, decompose, recompose, DoRALayer, build_version, build_all_versions,
    revert_magnitude, revert_direction, revert_both, revert_direction_slerp,
    seen_unseen_gap, identify_overfitting_component,
)

torch.manual_seed(0)
P = F = 0
def chk(name, cond, detail=""):
    global P, F
    if cond:
        P += 1; print(f"  PASS  {name}")
    else:
        F += 1; print(f"  FAIL  {name}   {detail}")


def mk(out=64, inn=128, r=16, alpha=32, dtype=torch.float64):
    W0 = torch.randn(out, inn, dtype=dtype)
    A = torch.randn(r, inn, dtype=dtype) * 0.02
    B = torch.randn(out, r, dtype=dtype) * 0.02
    m_ft = column_norm(W0).clone() * (1.0 + 0.05 * torch.randn(out, 1, dtype=dtype))
    return DoRALayer(W0=W0, A=A, B=B, scaling=alpha / r, m_ft=m_ft)


print("=" * 74); print("1. DECOMPOSITION"); print("=" * 74)
L = mk()
m, D = decompose(L.W0)
chk("decompose/recompose round-trips", torch.allclose(recompose(m, D), L.W0, atol=1e-12))
chk("magnitude shape is [out,1]", tuple(m.shape) == (64, 1), str(m.shape))
chk("direction rows are unit norm",
    torch.allclose(D.norm(dim=1), torch.ones(64, dtype=torch.float64), atol=1e-12))
chk("magnitude equals the row norm of W0",
    torch.allclose(m.squeeze(1), L.W0.norm(dim=1), atol=1e-12))
Z = torch.zeros(4, 8, dtype=torch.float64)
chk("zero rows do not divide by zero", torch.isfinite(decompose(Z)[1]).all())

print()
print("=" * 74); print("2. THE FOUR VERSIONS"); print("=" * 74)
V = build_all_versions(L)
chk("base == W0", torch.allclose(V["base"], L.W0, atol=1e-12),
    f"max|d|={(V['base']-L.W0).abs().max():.3e}")
chk("full == the model's effective weight",
    torch.allclose(V["full"], L.effective_weight(), atol=1e-12))
m0, D0 = L.pretrained_components(); m_ft, D_ft = L.finetuned_components()
chk("mag_only == m_ft * D0", torch.allclose(V["mag_only"], m_ft * D0, atol=1e-12))
chk("dir_only == m0 * D_ft", torch.allclose(V["dir_only"], m0 * D_ft, atol=1e-12))
chk("m_ft is NOT ||W0+sBA|| (the independence the method needs)",
    not torch.allclose(m_ft, column_norm(L.direction_source), atol=1e-6))
chk("D_ft rows unit norm", torch.allclose(D_ft.norm(dim=1), torch.ones(64, dtype=torch.float64), atol=1e-12))

print()
print("=" * 74); print("3. REVERSION ENDPOINTS  (1 = keep trained, 0 = fully reverted)"); print("=" * 74)
chk("revert_magnitude(1) == full", torch.allclose(revert_magnitude(L, 1.0), V["full"], atol=1e-12))
chk("revert_magnitude(0) == dir_only", torch.allclose(revert_magnitude(L, 0.0), V["dir_only"], atol=1e-12))
chk("revert_direction(1) == full", torch.allclose(revert_direction(L, 1.0), V["full"], atol=1e-12),
    f"max|d|={(revert_direction(L,1.0)-V['full']).abs().max():.3e}")
chk("revert_direction(0) == mag_only", torch.allclose(revert_direction(L, 0.0), V["mag_only"], atol=1e-12))
chk("revert_both(1,1) == full", torch.allclose(revert_both(L, 1.0, 1.0), V["full"], atol=1e-12))
chk("revert_both(0,0) == base", torch.allclose(revert_both(L, 0.0, 0.0), V["base"], atol=1e-12))
chk("revert_both(1,0) == mag_only", torch.allclose(revert_both(L, 1.0, 0.0), V["mag_only"], atol=1e-12))
chk("revert_both(0,1) == dir_only", torch.allclose(revert_both(L, 0.0, 1.0), V["dir_only"], atol=1e-12))

print()
print("=" * 74); print("4. REVERSION IS WELL-BEHAVED IN BETWEEN"); print("=" * 74)
# magnitude interpolation is linear in alpha
a = 0.37
lhs = revert_magnitude(L, a)
rhs = ((1 - a) * m0 + a * m_ft) * D_ft
chk("revert_magnitude is exactly the stated linear blend", torch.allclose(lhs, rhs, atol=1e-12))
# direction stays on the unit sphere for every beta
ok = True
for b in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    W = revert_direction(L, b)
    rows = (W / column_norm(W)).norm(dim=1)
    ok &= torch.allclose(rows, torch.ones(64, dtype=torch.float64), atol=1e-10)
chk("reverted direction stays unit-norm at every beta", ok)
# the magnitude is untouched by a direction reversion
ok = all(torch.allclose(column_norm(revert_direction(L, b)), m_ft, atol=1e-10)
         for b in [0.0, 0.25, 0.5, 0.75, 1.0])
chk("revert_direction preserves ||W|| == m_ft exactly", ok)
# and the direction is untouched by a magnitude reversion
ok = True
for aa in [0.0, 0.25, 0.5, 0.75, 1.0]:
    W = revert_magnitude(L, aa)
    ok &= torch.allclose(W / column_norm(W), D_ft, atol=1e-10)
chk("revert_magnitude preserves the direction exactly", ok)
# monotone travel away from full as beta decreases
d = [ (revert_direction(L, b) - V["full"]).norm().item() for b in [1.0, 0.75, 0.5, 0.25, 0.0] ]
chk("distance from the full model grows monotonically as beta falls",
    all(d[i] < d[i+1] for i in range(len(d)-1)), str([round(x,4) for x in d]))

print()
print("=" * 74); print("5. SLERP VARIANT (reported as an alternative, not used)"); print("=" * 74)
chk("slerp(1) == full", torch.allclose(revert_direction_slerp(L, 1.0), V["full"], atol=1e-9))
chk("slerp(0) == mag_only", torch.allclose(revert_direction_slerp(L, 0.0), V["mag_only"], atol=1e-9))
Lp = mk(); Lp.A = Lp.A * 1e-9; Lp.B = Lp.B * 1e-9   # D_ft ~ D0 -> near-parallel rows
chk("slerp is finite when the two directions are near-parallel",
    torch.isfinite(revert_direction_slerp(Lp, 0.5)).all())

print()
print("=" * 74); print("6. THE bf16 DIAL-1 ISSUE THE PIPELINE GUARDS AGAINST"); print("=" * 74)
L16 = mk(dtype=torch.float32)
full32 = build_version(L16, True, True)
rev32 = revert_direction(L16, 1.0)
d32 = (full32 - rev32).abs().max().item()
f16 = full32.to(torch.bfloat16).float()
r16 = rev32.to(torch.bfloat16).float()
d16 = (f16 - r16).abs().max().item()
print(f"     fp32  max|full - revert_direction(1)| = {d32:.3e}")
print(f"     bf16  max|full - revert_direction(1)| = {d16:.3e}")
chk("in fp32 the two routes agree to ~1e-7", d32 < 1e-6, f"{d32:.3e}")
chk("bf16 cast makes them differ (this is why the pipeline dispatches dial 1)",
    d16 > 0, f"{d16:.3e}")

print()
print("=" * 74); print("7. IDENTIFICATION DECISION LOGIC"); print("=" * 74)
chk("gap is seen - unseen", seen_unseen_gap(0.9, 0.4) == 0.5)
r = identify_overfitting_component({"base":0.1,"mag_only":0.5,"dir_only":0.15,"full":0.55})
chk("picks magnitude when magnitude drives the rise", r.component == "magnitude", r.component)
r = identify_overfitting_component({"base":0.1,"mag_only":0.15,"dir_only":0.5,"full":0.55})
chk("picks direction when direction drives the rise", r.component == "direction", r.component)
r = identify_overfitting_component({"base":0.1,"mag_only":0.12,"dir_only":0.13,"full":0.60})
chk("calls it an interaction when neither single rise is large", r.component == "both", r.component)
r = identify_overfitting_component({"base":0.5,"mag_only":0.4,"dir_only":0.45,"full":0.45})
chk("returns 'none' when fine-tuning did not widen the gap", r.component == "none", r.component)
r = identify_overfitting_component({"base":0.1,"mag_only":0.4,"dir_only":0.4,"full":0.5})
chk("ties go to magnitude (documented tie-break)", r.component == "magnitude", r.component)

print()
print("=" * 74)
print(f"{P} passed, {F} failed")
print("=" * 74)
sys.exit(1 if F else 0)
