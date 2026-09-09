"""Regenerate every data figure in the thesis, into figures/out/.

    python figures/make_all.py

Figures 2.1, 2.2 and 2.3 are schematics with no data behind them and are not
produced here.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    ("Figure 5.1  sweeps_grid.png",      "fig5_1_sweeps_grid.py"),
    ("Figure 5.2  sweeps_pareto.png",    "fig5_2_sweeps_pareto.py"),
    ("Figure 5.3  magdir_hist.png",      "fig5_3_magdir_hist.py"),
    ("Figure 5.4  sweeps_grid_sel.png",  "fig5_4_sweeps_grid_sel.py"),
    ("Figure 5.5  es_paired.png",        "fig5_5_es_paired.py"),
    ("Figures 5.6 and 5.7  layer_pareto.png, layer_zoom.png",
                                          "fig5_6_layer_pareto.py"),
]

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    for label, script in SCRIPTS:
        print(f"\n===== {label} " + "=" * max(0, 60 - len(label)))
        runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    print("\nAll figures written to", os.path.join(HERE, "out"))
