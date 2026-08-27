"""
dora_reversion.py
=================

My implementation of the reversion method, working directly on weight tensors.
No GPU and no training here -- everything is plain tensor algebra on weights that
already exist after DoRA fine-tuning. I keep it separate so the maths is easy to
test on its own before feeding in real adapters.

Layout I use everywhere: a linear weight is W : [out_features, in_features].
DoRA splits it into a magnitude and a direction:

        W = m * D

The norm is taken along the input dim (dim=1), so there's one magnitude per output
row (one per neuron):

        m = ||W||   -> [out, 1]
        D = W / m   -> [out, in]   (each row is a unit vector)

The DoRA paper calls each per-neuron slice a "column"; in this [out, in] layout
that's a row (the dim=1 norm). `dim` is a parameter just in case, but dim=1 is the
standard DoRA setting.

The four things the method needs:
  from the pretrained weight W0:  m0 = ||W0|| ,  D0 = W0 / ||W0||
  from the fine-tuned DoRA layer: D_ft = (W0 + s*B*A) / ||W0 + s*B*A|| ,
                                  m_ft = the separately trained magnitude vector

The key point that makes this possible: m_ft is its own trained parameter, it is
NOT equal to ||W0 + s*B*A||. Magnitude and direction are trained separately, so I
can revert them separately. Plain LoRA has no such split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import torch

Tensor = torch.Tensor
Component = Literal["magnitude", "direction", "both", "none"]


# ---------------------------------------------------------------------------
# 1. Decomposition helpers
# ---------------------------------------------------------------------------

def column_norm(W: Tensor, dim: int = 1, eps: float = 1e-12) -> Tensor:
    """||W|| per row, kept 2-D ([out, 1]) so it broadcasts. Floored at eps so
    dividing by it is always safe."""
    return torch.linalg.norm(W, dim=dim, keepdim=True).clamp_min(eps)


def decompose(W: Tensor, dim: int = 1) -> Tuple[Tensor, Tensor]:
    """Split W into (magnitude, direction) with W = magnitude * direction.
    magnitude [out, 1], direction [out, in] with unit rows."""
    m = column_norm(W, dim=dim)
    D = W / m
    return m, D


def recompose(m: Tensor, D: Tensor) -> Tensor:
    """Put it back together: W = m * D."""
    return m * D


# ---------------------------------------------------------------------------
# 2. One fine-tuned DoRA layer, described by the tensors it actually has
# ---------------------------------------------------------------------------

@dataclass
class DoRALayer:
    """The tensors of one adapted linear layer.

    W0      : [out, in]  frozen pretrained weight
    A       : [r, in]    LoRA down-projection
    B       : [out, r]   LoRA up-projection
    scaling : float      LoRA scaling s = alpha / r (fixed, not trained)
    m_ft    : [out, 1]   the trained magnitude vector (its own parameter)
    dim     : int        norm axis (1 = DoRA default)
    """

    W0: Tensor
    A: Tensor
    B: Tensor
    scaling: float
    m_ft: Tensor
    dim: int = 1

    @property
    def delta(self) -> Tensor:
        """The LoRA update s*B*A  : [out, in]."""
        return self.scaling * (self.B @ self.A)

    @property
    def direction_source(self) -> Tensor:
        """V = W0 + s*B*A -- the matrix I normalise to get D_ft."""
        return self.W0 + self.delta

    def pretrained_components(self) -> Tuple[Tensor, Tensor]:
        """(m0, D0) from the frozen weight."""
        return decompose(self.W0, dim=self.dim)

    def finetuned_components(self) -> Tuple[Tensor, Tensor]:
        """(m_ft, D_ft). Note I throw away ||V|| and use the trained m_ft instead
        -- that independence is the whole point."""
        _, D_ft = decompose(self.direction_source, dim=self.dim)
        return self.m_ft, D_ft

    def effective_weight(self) -> Tensor:
        """The weight the DoRA model actually uses: m_ft * D_ft."""
        m_ft, D_ft = self.finetuned_components()
        return recompose(m_ft, D_ft)


# ---------------------------------------------------------------------------
# 3. The four diagnostic versions
#    Each sets each component to either its pretrained or trained value:
#      base     : (pretrained m, pretrained D)  == W0
#      mag_only : (trained    m, pretrained D)
#      dir_only : (pretrained m, trained    D)
#      full     : (trained    m, trained    D)  == the DoRA model
# ---------------------------------------------------------------------------

def build_version(
    layer: DoRALayer,
    use_trained_magnitude: bool,
    use_trained_direction: bool,
) -> Tensor:
    """Rebuild one of the four versions."""
    m0, D0 = layer.pretrained_components()
    m_ft, D_ft = layer.finetuned_components()
    m = m_ft if use_trained_magnitude else m0
    D = D_ft if use_trained_direction else D0
    return recompose(m, D)


def build_all_versions(layer: DoRALayer) -> Dict[str, Tensor]:
    """All four versions, keyed by name."""
    return {
        "base":     build_version(layer, False, False),  # == W0
        "mag_only": build_version(layer, True,  False),
        "dir_only": build_version(layer, False, True),
        "full":     build_version(layer, True,  True),   # == DoRA model
    }


# ---------------------------------------------------------------------------
# 4. Reverting a component
#    Coefficient convention (same as WiSE-FT's alpha):
#      coeff = 1 -> keep the fine-tuned value (no reversion)
#      coeff = 0 -> go fully back to pretrained
#    So smaller coeff = more reversion.
# ---------------------------------------------------------------------------

def revert_magnitude(layer: DoRALayer, alpha: float) -> Tensor:
    """Revert only the magnitude, keep the trained direction.
        m_rev   = (1 - alpha) * m0 + alpha * m_ft
        W_final = m_rev * D_ft
    alpha=1 -> full DoRA ; alpha=0 -> the 'dir_only' version.
    """
    m0, _ = layer.pretrained_components()
    m_ft, D_ft = layer.finetuned_components()
    m_rev = (1.0 - alpha) * m0 + alpha * m_ft
    return recompose(m_rev, D_ft)


def revert_direction(layer: DoRALayer, beta: float) -> Tensor:
    """Revert only the direction, keep the trained magnitude.
        D_rev   = normalise( (1 - beta) * D0 + beta * D_ft )
        W_final = m_ft * D_rev
    The renormalise is needed because a blend of two unit vectors isn't itself
    unit length. beta=1 -> full DoRA ; beta=0 -> the 'mag_only' version.
    """
    m0_unused, D0 = layer.pretrained_components()
    m_ft, D_ft = layer.finetuned_components()
    D_blend = (1.0 - beta) * D0 + beta * D_ft
    D_rev = D_blend / column_norm(D_blend, dim=layer.dim)
    return recompose(m_ft, D_rev)


def revert_both(layer: DoRALayer, alpha: float, beta: float) -> Tensor:
    """Revert both components at once (used for the 'revert both' baseline).
    alpha reverts the magnitude, beta reverts the direction:
        m_rev   = (1 - alpha) * m0 + alpha * m_ft
        D_rev   = normalise( (1 - beta) * D0 + beta * D_ft )
        W_final = m_rev * D_rev
    """
    m0, D0 = layer.pretrained_components()
    m_ft, D_ft = layer.finetuned_components()
    m_rev = (1.0 - alpha) * m0 + alpha * m_ft
    D_blend = (1.0 - beta) * D0 + beta * D_ft
    D_rev = D_blend / column_norm(D_blend, dim=layer.dim)
    return recompose(m_rev, D_rev)


def revert_direction_slerp(layer: DoRALayer, beta: float) -> Tensor:
    """Slerp version of direction reversion: interpolate along the sphere at
    constant angular speed instead of blending + renormalising. Cleaner in
    principle; here just for comparison, not used in the main result.
    """
    _, D0 = layer.pretrained_components()
    m_ft, D_ft = layer.finetuned_components()

    # angle per row between the two unit directions
    dot = (D0 * D_ft).sum(dim=layer.dim, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.arccos(dot)
    sin_theta = torch.sin(theta).clamp_min(1e-7)

    # slerp: D0 at beta=0, D_ft at beta=1
    w0 = torch.sin((1.0 - beta) * theta) / sin_theta
    w1 = torch.sin(beta * theta) / sin_theta
    D_rev = w0 * D0 + w1 * D_ft

    # near-parallel rows would divide by ~0, so fall back to the linear blend
    nearly_parallel = (theta < 1e-4)
    if nearly_parallel.any():
        lin = (1.0 - beta) * D0 + beta * D_ft
        lin = lin / column_norm(lin, dim=layer.dim)
        D_rev = torch.where(nearly_parallel, lin, D_rev)

    return recompose(m_ft, D_rev)


# ---------------------------------------------------------------------------
# 5. Deciding which component overfits
#    I evaluate each of the four versions on seen vs unseen, take each one's
#    seen->unseen gap, and decide which component is to blame. This function is
#    just the decision logic; the metric values come from the eval harness, so
#    it stays GPU-free.
# ---------------------------------------------------------------------------

def seen_unseen_gap(score_seen: float, score_unseen: float) -> float:
    """Generalisation gap for one version (higher score = better, so a bigger
    gap = worse generalisation)."""
    return score_seen - score_unseen


@dataclass
class IdentificationResult:
    component: Component            # 'magnitude', 'direction', 'both', or 'none'
    rise_magnitude: float          # how much mag_only's gap rises above base
    rise_direction: float          # how much dir_only's gap rises above base
    rise_full: float               # how much full's gap rises above base
    explanation: str


def identify_overfitting_component(
    gaps: Dict[str, float],
    interaction_ratio: float = 0.5,
    min_full_rise: float = 0.0,
) -> IdentificationResult:
    """Decide which DoRA component is responsible for the loss of generalisation.

    gaps: the seen->unseen gap of each version, keyed 'base','mag_only',
          'dir_only','full' (each computed with seen_unseen_gap).

    Idea: base has the smallest gap, full the largest. I measure how far each
    single-component version moves the gap up from base:
        rise_mag = gap(mag_only) - gap(base)
        rise_dir = gap(dir_only) - gap(base)
    If one clearly dominates, that's the culprit -> revert it.

    Interaction case: if BOTH single rises are small next to the full rise, then
    neither component overfits alone and the effect is in their product m*D ->
    revert both. `interaction_ratio` is the threshold (larger single rise below
    this fraction of the full rise -> interaction).

    Degenerate case: the method assumes full has a bigger gap than base. If it
    doesn't (rise_full <= min_full_rise), there is nothing to attribute and I
    return component='none' -- "DoRA didn't overfit here" is itself the finding.
    """
    g_base = gaps["base"]
    rise_mag = gaps["mag_only"] - g_base
    rise_dir = gaps["dir_only"] - g_base
    rise_full = gaps["full"] - g_base

    larger_single = max(rise_mag, rise_dir)

    # fine-tuning didn't widen the gap -> nothing to attribute
    if rise_full <= min_full_rise:
        return IdentificationResult(
            component="none",
            rise_magnitude=rise_mag,
            rise_direction=rise_dir,
            rise_full=rise_full,
            explanation=(
                f"No attributable overfitting: the full model's gap rise "
                f"({rise_full:+.4f}) is <= {min_full_rise:.4f}, so DoRA did NOT "
                f"widen the seen->unseen gap on this task/shift. The premise for "
                f"component identification does not hold; reverting cannot recover "
                f"generalisation that was not lost this way. Treat as a negative "
                f"result, not a magnitude/direction finding."
            ),
        )

    # neither single component reproduces much of the full gap -> interaction
    if rise_full > 0 and larger_single < interaction_ratio * rise_full:
        return IdentificationResult(
            component="both",
            rise_magnitude=rise_mag,
            rise_direction=rise_dir,
            rise_full=rise_full,
            explanation=(
                f"Interaction case: the larger single-component gap rise "
                f"({larger_single:.4f}) is below {interaction_ratio:.0%} of the "
                f"full-model rise ({rise_full:.4f}). Neither magnitude nor "
                f"direction overfits alone; revert both."
            ),
        )

    if rise_mag >= rise_dir:
        return IdentificationResult(
            component="magnitude",
            rise_magnitude=rise_mag,
            rise_direction=rise_dir,
            rise_full=rise_full,
            explanation=(
                f"Magnitude is the overfitting component: it raises the gap by "
                f"{rise_mag:.4f} vs {rise_dir:.4f} for direction. Revert magnitude "
                f"(sweep alpha in eq. 3.1)."
            ),
        )

    return IdentificationResult(
        component="direction",
        rise_magnitude=rise_mag,
        rise_direction=rise_dir,
        rise_full=rise_full,
        explanation=(
            f"Direction is the overfitting component: it raises the gap by "
            f"{rise_dir:.4f} vs {rise_mag:.4f} for magnitude. Revert direction "
            f"(sweep beta in eq. 3.2)."
        ),
    )


def revert_identified(layer: DoRALayer, result: IdentificationResult,
                      coeff: float) -> Tensor:
    """Apply the reversion chosen by identify_overfitting_component, with the
    coefficient picked on the validation set. For the 'both' case the same coeff
    is used for magnitude and direction."""
    if result.component == "magnitude":
        return revert_magnitude(layer, coeff)
    if result.component == "direction":
        return revert_direction(layer, coeff)
    return revert_both(layer, coeff, coeff)
