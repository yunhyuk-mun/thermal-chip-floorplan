"""Stage 1 experiment: wirelength-only vs thermal-aware annealing on ami33.

Same annealer, same iterations, same seeds; only the thermal weight changes.
Writes results/stage1_floorplan/stage1_runs.json and results/stage1_floorplan/stage1_layouts.npz.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import json
import time
from pathlib import Path

import numpy as np

from chipthermal.floorplan import load_problem, normalizers, anneal

WEIGHTS = [0.0, 0.5, 1.0, 2.0, 4.0]
SEEDS = [0, 1, 2]
ITERS = 20000
PENALTY = 50.0

from chipthermal.paths import STAGE1 as out
prob = load_problem()
norm = normalizers(prob, n=200)
runs, layouts = [], {}
t_start = time.perf_counter()
for wt in WEIGHTS:
    for s in SEEDS:
        t0 = time.perf_counter()
        c, pos, neg, rot, e = anneal(prob, wt, norm, iters=ITERS, seed=s, penalty=PENALTY)
        runs.append(dict(w_temp=wt, seed=s, cost=float(c), hpwl=float(e["hpwl"]), tmax=float(e["tmax"]),
                         dtmax=float(e["tmax"] - prob.solver.p.t_amb), overflow=float(e["overflow"]),
                         legal=bool(e["overflow"] == 0.0), seconds=time.perf_counter() - t0))
        key = f"w{wt}_s{s}"
        layouts[key + "_xywh"] = np.stack([e["x"], e["y"], e["w"], e["h"]])
        layouts[key + "_temp"] = e["temp"]
        print(json.dumps(runs[-1]), flush=True)

meta = dict(norm_hpwl=norm[0], norm_dtmax=norm[1], outline=prob.outline, names=prob.names,
            hot=[prob.names[i] for i in prob.hot], power=prob.power.tolist(), iters=ITERS,
            penalty=PENALTY, weights=WEIGHTS, seeds=SEEDS, total_seconds=time.perf_counter() - t_start)
(out / "stage1_runs.json").write_text(json.dumps(dict(meta=meta, runs=runs), indent=1))
np.savez(out / "stage1_layouts.npz", **layouts)
print("done", round(meta["total_seconds"]), "s")
