"""Six-panel early-stopping vs direction-reversion frontier plot.

One cell per re-split, each cell = a full-range panel above a dedicated zoom panel
(the zoom gets its own axes rather than an inset, so nothing is ever drawn on top
of the data). Axes are RAW field-level F1 on the two tasks:

    x = old task (the one being forgotten)
    y = new task (the one being learned)

Two paths are drawn, both starting from essentially the same place -- the stage-1
model, where the old task is at its ceiling and the new task is at zero:

  * direction sweep   -- the EARLY-STOPPED adapter's own uniform reversion,
                         beta = 1 -> 0. Its beta = 1 end IS the early-stopping model,
                         so this curve branches off the training path rather than
                         belonging to a different run: the comparison is PAIRED.
  * training path     -- stage-2 training itself, one point every 10 optimiser steps,
                         up to the early-stopping model chosen by patience-3 on the
                         new task

The question: does training less (the red path) reach the same trade-off frontier as
reverting after training (the blue path)? Everything is on the VALIDATION split,
which is where both the stopping rule and the dial selection act.

Inputs:  es_plot_data.json   (written by es_plot_data.py on the server)
Output:  es_frontier_6panel.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/tmp/es_plot_data_paired.json"
OUT = "/tmp/es_frontier_paired.png"

C_SWEEP = "#1a5fb4"
C_TRAJ = "#c01c28"
C_ANCHOR = "#000000"

PANELS = [("b2q_s1", "Bank $\\rightarrow$ Quarterly", 1),
          ("b2q_s2", "Bank $\\rightarrow$ Quarterly", 2),
          ("b2q_s3", "Bank $\\rightarrow$ Quarterly", 3),
          ("q2b_s1", "Quarterly $\\rightarrow$ Bank", 1),
          ("q2b_s2", "Quarterly $\\rightarrow$ Bank", 2),
          ("q2b_s3", "Quarterly $\\rightarrow$ Bank", 3)]

d = json.load(open(DATA))


def zoom_window(rec):
    """x/y window covering the region where the training path bunches up."""
    xs = np.array([t["old"] for t in rec["traj"]])
    hi = max(0.045, float(np.percentile(xs, 75)) * 1.45)
    ys = [t["new"] for t in rec["traj"] if t["old"] <= hi]
    sy = [s["new"] for s in rec["sweep"] if s["old"] <= hi]
    if len(ys) < 2:
        return None
    ylo, yhi = min(ys + sy), max(ys + sy)
    pad = max(0.02, (yhi - ylo) * 0.14)
    return (-0.0035, hi, ylo - pad, yhi + pad)


def draw(ax, rec, win=None, small=False, skip_win=None):
    """win=None -> full range panel; otherwise only draw/label inside the window."""
    sw, tr, a = rec["sweep"], rec["traj"], rec["anchor"]
    ms = 5.0 if small else 7.0
    lw = 1.5 if small else 2.0

    def inside(x, y):
        if win is None:
            return True
        return win[0] <= x <= win[1] and win[2] <= y <= win[3]

    def labelled_here(x, y):
        """Each point is labelled exactly ONCE across the panel pair: the zoom owns
        the points inside its window, the full-range panel owns the rest."""
        if not inside(x, y):
            return False
        if skip_win is None:
            return True
        return not (skip_win[0] <= x <= skip_win[1] and skip_win[2] <= y <= skip_win[3])

    ax.plot([s["old"] for s in sw], [s["new"] for s in sw], "-o", color=C_SWEEP,
            lw=lw, ms=ms, mfc="white", mew=1.7, zorder=3,
            label="direction reversion of the early-stopped adapter")
    tx = [a["old"]] + [t["old"] for t in tr]
    ty = [a["new"]] + [t["new"] for t in tr]
    ax.plot(tx, ty, "-o", color=C_TRAJ, lw=lw, ms=ms - 1.2, mfc="white", mew=1.5,
            zorder=4, label="stage-2 training path")

    if inside(a["old"], a["new"]):
        ax.plot([a["old"]], [a["new"]], "*", color=C_ANCHOR,
                ms=13 if small else 17, zorder=7, label="stage-1 model (step 0)")
    es = tr[-1]
    if inside(es["old"], es["new"]):
        ax.plot([es["old"]], [es["new"]], "s", color=C_TRAJ, ms=ms + 3.5,
                mew=1.8, mec="black", zorder=8, label="early-stopping model")
    b = rec["selected_beta"]
    sel = next((s for s in sw if abs(s["beta"] - b) < 1e-9), None) if b is not None else None
    if sel and inside(sel["old"], sel["new"]):
        ax.plot([sel["old"]], [sel["new"]], "o", color=C_SWEEP, ms=ms + 7,
                mfc="none", mew=2.4, zorder=6, label="selected dial $\\beta^*$")

    # EVERY point gets a label; placement is resolved later against collisions
    fs = 6.4 if small else 8.0
    jobs = []
    for sp in sw:
        if labelled_here(sp["old"], sp["new"]):
            jobs.append((sp["old"], sp["new"], f"{sp['beta']:g}", C_SWEEP, "bold", fs))
    for t in tr:
        if labelled_here(t["old"], t["new"]):
            jobs.append((t["old"], t["new"], str(t["step"]), C_TRAJ, "normal", fs))
    return jobs


def _overlap(a, b, pad=1.0):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or
                a[3] + pad < b[1] or b[3] + pad < a[1])


def place_labels(ax, jobs):
    """Annotate every job, choosing the first candidate offset that collides with
    neither an already-placed label nor a data marker."""
    fig = ax.figure
    k = fig.dpi / 72.0
    CAND = [(6, 5), (6, -13), (-17, 5), (-17, -13), (12, -4), (-23, -4),
            (6, 15), (6, -23), (-17, 15), (-17, -23), (0, 20), (0, -28)]
    # markers occupy space too: seed the occupancy map with them
    placed = []
    for (x, y, *_rest) in jobs:
        px, py = ax.transData.transform((x, y))
        placed.append((px - 5 * k, py - 5 * k, px + 5 * k, py + 5 * k))
    for (x, y, txt, color, weight, fs) in jobs:
        px, py = ax.transData.transform((x, y))
        w = max(1, len(txt)) * fs * 0.62 * k
        h = fs * 1.15 * k
        chosen = None
        for (dx, dy) in CAND:
            ox, oy = px + dx * k, py + dy * k
            box = (ox, oy, ox + w, oy + h)
            if any(_overlap(box, q) for q in placed):
                continue
            chosen = (dx, dy, box)
            break
        if chosen is None:
            dx, dy = CAND[0]
            ox, oy = px + dx * k, py + dy * k
            chosen = (dx, dy, (ox, oy, ox + w, oy + h))
        placed.append(chosen[2])
        ax.annotate(txt, (x, y), textcoords="offset points",
                    xytext=(chosen[0], chosen[1]), fontsize=fs, color=color,
                    fontweight=weight, zorder=10)


fig = plt.figure(figsize=(18.5, 14.2))
outer = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.235,
                         left=0.052, right=0.985, top=0.855, bottom=0.045)

label_jobs = []
for idx, (key, order_name, seed) in enumerate(PANELS):
    rec = d[key]
    r, c = divmod(idx, 3)
    cell = outer[r, c].subgridspec(2, 1, height_ratios=[2.05, 1.0], hspace=0.34)
    ax = fig.add_subplot(cell[0])
    azm = fig.add_subplot(cell[1])

    win = zoom_window(rec)
    jobs_main = draw(ax, rec, skip_win=win)
    ax.set_title(f"{order_name} — split {seed}\n"
                 f"old = {rec['old_task']}   new = {rec['new_task']}   "
                 f"ES model at step {rec['es_step']}",
                 fontsize=11, pad=9)
    ax.set_xlabel(f"old-task field F1  ({rec['old_task']})", fontsize=9.5)
    ax.set_ylabel(f"new-task field F1  ({rec['new_task']})", fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8.5)
    xmax = max([s["old"] for s in rec["sweep"]] + [rec["anchor"]["old"]]) * 1.13
    ymax = max([s["new"] for s in rec["sweep"]] + [t["new"] for t in rec["traj"]]) * 1.22
    ax.set_xlim(-0.028, xmax)
    ax.set_ylim(-0.055, ymax)

    label_jobs.append((ax, jobs_main))
    if win:
        x0, x1, y0, y1 = win
        jobs_zoom = draw(azm, rec, win=win, small=True)
        azm.set_xlim(x0, x1)
        azm.set_ylim(y0, y1)
        azm.grid(alpha=0.25, lw=0.5)
        azm.tick_params(labelsize=7.5)
        azm.set_xlabel(f"old-task field F1 — zoom of the dashed box  [{x0:.3f}, {x1:.3f}]", fontsize=8.2)
        azm.set_ylabel("new-task F1", fontsize=8.2)
        azm.set_facecolor("#f7f7f9")
        for sp in azm.spines.values():
            sp.set_edgecolor("0.45")
            sp.set_linewidth(1.2)
        # show the zoom window on the full-range panel
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec="0.40", lw=1.2, ls=(0, (4, 2)), zorder=5))
        label_jobs.append((azm, jobs_zoom))
    else:
        azm.axis("off")
        azm.text(0.5, 0.5, "no collapsed region\n(training path stays spread out)",
                 ha="center", va="center", fontsize=9, color="0.35",
                 transform=azm.transAxes)

fig.canvas.draw()
for _ax, _jobs in label_jobs:
    place_labels(_ax, _jobs)

h, l = fig.axes[0].get_legend_handles_labels()
seen, hh, ll = set(), [], []
for a, b in zip(h, l):
    if b not in seen:
        seen.add(b); hh.append(a); ll.append(b)
fig.legend(hh, ll, loc="upper center", ncol=5, fontsize=11.5, frameon=False,
           bbox_to_anchor=(0.5, 0.925))
fig.suptitle("Early stopping versus direction reversion, both from the same training run\n"
             "validation split  ·  blue labels are the reversion dial $\\beta$  ·  red labels are optimiser steps  ·  "
             "dashed box marks the zoom window shown beneath each panel",
             fontsize=13.5, y=0.988)
fig.savefig(OUT, dpi=170, facecolor="white")
print("wrote", OUT)
