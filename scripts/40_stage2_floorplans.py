"""Stage 2 data: 200 legal ami33 floorplans, generated fast.

1) Four full anneals (different seeds/weights) give legal starting sequence pairs.
2) From each start, a random walk applies sequence-pair moves and keeps only legal results
   (inside the outline); a floorplan is recorded every STEP accepted moves.
Changed from the first version (independent short anneals) because only ~6% of those were legal.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import os
import random
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from chipthermal.paths import STAGE2
OUT = STAGE2 / "stage2_floorplans.npz"
N_WANT, STEP = 200, 25
STARTS = [(0.0, 101), (0.5, 102), (1.0, 103), (2.0, 104)]


def start_point(arg):
    from chipthermal.floorplan import load_problem, normalizers, anneal
    w, seed = arg
    prob = load_problem()
    norm = normalizers(prob, n=100)
    c, pos, neg, rot, e = anneal(prob, w, norm, iters=20000, seed=seed, penalty=50.0)
    return pos, neg, rot, e["overflow"]


def walk(prob, pos, neg, rot, n_out, seed):
    from chipthermal.floorplan import evaluate
    rng = random.Random(seed)
    n = len(pos); out = []; accepted = 0
    while len(out) < n_out:
        p2, n2, r2 = pos[:], neg[:], rot.copy()
        i, j = rng.sample(range(n), 2); mv = rng.random()
        if mv < 0.35:
            p2[i], p2[j] = p2[j], p2[i]
        elif mv < 0.7:
            n2[i], n2[j] = n2[j], n2[i]
        else:
            r2[i] = ~r2[i]
        e = evaluate(prob, p2, n2, r2)
        if e["overflow"] == 0.0:
            pos, neg, rot = p2, n2, r2; accepted += 1
            if accepted % STEP == 0:
                out.append(np.stack([e["x"], e["y"], e["w"], e["h"]]))
    return out


if __name__ == "__main__":
    from chipthermal.floorplan import load_problem
    with Pool(4) as pool:
        starts = pool.map(start_point, STARTS)
    print("start overflows:", [round(s[3], 4) for s in starts], flush=True)
    prob = load_problem()
    layouts = []
    legal = [s for s in starts if s[3] == 0.0]
    per = -(-N_WANT // len(legal))
    for k, (pos, neg, rot, _) in enumerate(legal):
        layouts += walk(prob, pos, neg, rot, per, seed=900 + k)
        print(f"start {k}: total {len(layouts)}", flush=True)
    # keep start order: floorplans 150-199 come from a separate start (strict "never seen" test family)
    arr = np.stack(layouts[:N_WANT])
    start_id = np.repeat(np.arange(len(legal)), per)[:N_WANT]
    np.savez(OUT, xywh=arr, start_id=start_id)
    print("saved", len(arr))
