"""Stage 1 robustness: 10 seeds per weight at 20k iterations, plus a 40k-iteration convergence check.

Runs in parallel processes; each finished run is appended to results/stage1_floorplan/stage1_robust.jsonl
so partial results survive an interruption. Seeds 0-2 at 20k should reproduce stage1_runs.json.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from chipthermal.paths import STAGE1
OUT = STAGE1 / "stage1_robust.jsonl"
PENALTY = 50.0
TASKS = [(w, s, 20000) for w in (0.0, 0.5, 1.0, 2.0, 4.0) for s in range(10)]
TASKS += [(w, s, 40000) for w in (0.0, 0.5) for s in range(5)]

_prob = _norm = None


def init():
    global _prob, _norm
    from chipthermal.floorplan import load_problem, normalizers
    _prob = load_problem()
    _norm = normalizers(_prob, n=200)


def work(task):
    from chipthermal.floorplan import anneal
    w, s, iters = task
    t0 = time.perf_counter()
    c, pos, neg, rot, e = anneal(_prob, w, _norm, iters=iters, seed=s, penalty=PENALTY)
    return dict(w_temp=w, seed=s, iters=iters, cost=float(c), hpwl=float(e["hpwl"]),
                tmax=float(e["tmax"]), overflow=float(e["overflow"]), legal=bool(e["overflow"] == 0.0),
                seconds=round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["w_temp"], r["seed"], r["iters"]))
    todo = [t for t in TASKS if t not in done]
    print(f"{len(todo)} runs to do ({len(done)} already done)", flush=True)
    t_start = time.perf_counter()
    with Pool(8, initializer=init) as pool, OUT.open("a") as f:
        for r in pool.imap_unordered(work, todo):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(json.dumps(r), flush=True)
    print("done", round(time.perf_counter() - t_start), "s", flush=True)
