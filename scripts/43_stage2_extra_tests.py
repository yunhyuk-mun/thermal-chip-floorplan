"""Stage 2 follow-up checks (added after review).

1) Physics-based baseline that uses the same floorplan information as the network:
   with the floorplan known, each block's thermal response G is computed, block powers are
   estimated from the 16 sensor readings by ridge regression toward the prior mean, and the
   full map is recomputed. This baseline knows the exact solver, i.e. more than the network.
2) A second test family: 52 floorplans from 4 NEW independent starts (different anneal seeds),
   so the test is not a single random-walk family. The trained network is reused (no retraining).

Writes results/stage2_nn/stage2_extra_results.json. All values are assumption-based simulation.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import importlib.util
import json
import os
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from chipthermal.paths import STAGE2 as R

NEW_STARTS = [(0.0, 105), (0.5, 106), (1.0, 107), (4.0, 108)]
PER_START = 13
PRIOR = 0.85                      # mean of U(0.2, 1.5) power multipliers


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def start_point(arg):
    """Same as scripts/40 start_point (defined here so worker processes can import it on Windows)."""
    from chipthermal.floorplan import load_problem, normalizers, anneal
    w, seed = arg
    prob = load_problem()
    c, pos, neg, rot, e = anneal(prob, w, normalizers(prob, n=100), iters=20000, seed=seed, penalty=50.0)
    return pos, neg, rot, e["overflow"]


def physics_recon(prob, xywh, readings, sensors, lam):
    """Ridge estimate of per-block power multipliers from sensor readings, then full-map recompute."""
    from chipthermal.sensors import block_responses
    G, _ = block_responses(prob, xywh)                       # (blocks, cells) K per W
    A = (prob.power[:, None] * G[:, sensors]).T              # (k, blocks): multiplier -> sensor rise
    b = readings - prob.solver.p.t_amb - PRIOR * A.sum(1)    # residual w.r.t. prior
    lam_eff = lam * np.trace(A.T @ A) / A.shape[1]
    s = PRIOR + np.linalg.solve(A.T @ A + lam_eff * np.eye(A.shape[1]), A.T @ b.T).T
    return prob.solver.p.t_amb + (s * prob.power[None, :]) @ G


if __name__ == "__main__":
    import torch
    from chipthermal.floorplan import load_problem
    from chipthermal.sensors import block_responses, workloads, pca_basis, recon_nearest, recon_pca
    from chipthermal.nn import N, UNet, predict, rmse, tmax_err, noisy

    m41 = _load("s41", "41_stage2_nn.py")
    m40 = _load("s40", "40_stage2_floorplans.py")
    GRID16 = m41.GRID16
    prob = load_problem()

    # original data (same generation as script 41)
    T, D = m41.build_unseen_dataset()
    fps = np.load(R / "stage2_floorplans.npz")["xywh"]
    dmax = D.max()
    va = [i for i in range(150) if i % 10 == 9]
    tr = [i for i in range(150) if i % 10 != 9]
    te = list(range(150, 200))
    Ttr = T[tr].reshape(-1, N * N)
    mu, phi = pca_basis(Ttr, 12)                              # rank chosen in script 41

    # lambda for the physics baseline, chosen on validation floorplans (noise-free)
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    va_err = {}
    for lam in lams:
        errs = [rmse(physics_recon(prob, fps[f], T[f][:, GRID16], GRID16, lam), T[f]) for f in va]
        va_err[lam] = float(np.mean(errs))
    lam = min(va_err, key=va_err.get)
    print("physics lambda (val):", {k: round(v, 3) for k, v in va_err.items()}, "->", lam, flush=True)

    model = UNet(3); model.load_state_dict(torch.load(R / "stage2_nn_floorplan.pt")); model.eval()

    # new independent test family
    with Pool(4) as pool:
        starts = pool.map(start_point, NEW_STARTS)
    legal = [s for s in starts if s[3] == 0.0]
    print("new starts legal:", len(legal), "of", len(starts), flush=True)
    new_fps = []
    for k, (pos, neg, rot, _) in enumerate(legal):
        new_fps += m40.walk(prob, pos, neg, rot, PER_START, seed=950 + k)
    rng = np.random.default_rng(456)
    T_new, D_new = [], []
    for xywh in new_fps:
        G, dens = block_responses(prob, xywh)
        T_new.append(workloads(prob, G, 50, rng)); D_new.append(dens)
    T_new = np.array(T_new, np.float32); D_new = np.array(D_new, np.float32)
    np.savez(R / "stage2_extra_floorplans.npz", xywh=np.array(new_fps))

    families = {"original_test": ([fps[f] for f in te], T[te], D[te]),
                "new_starts": (new_fps, T_new, D_new)}
    rows = []
    for fam, (flist, TT, DD) in families.items():
        F, S, _ = TT.shape
        Tflat = TT.reshape(-1, N * N)
        Dflat = np.repeat(DD / dmax, S, axis=0)
        for noise in (0.0, 0.5, 1.0):
            read = noisy(Tflat, GRID16, noise, np.random.default_rng(99))
            phys = np.concatenate([physics_recon(prob, flist[f], read[f * S:(f + 1) * S], GRID16, lam) for f in range(F)])
            ests = {"nearest": recon_nearest(read, GRID16, N), "pca_pooled": recon_pca(read, GRID16, mu, phi),
                    "physics": phys,
                    "nn_floorplan": predict(model, Tflat, Dflat, GRID16, noise, np.random.default_rng(99), True)}
            for m, e in ests.items():
                per_fp = [rmse(e[f * S:(f + 1) * S], Tflat[f * S:(f + 1) * S]) for f in range(F)]
                rows.append(dict(family=fam, method=m, noise=noise, n_floorplans=F, rmse=rmse(e, Tflat),
                                 tmax_err=tmax_err(e, Tflat), rmse_fp_p10=float(np.percentile(per_fp, 10)),
                                 rmse_fp_p90=float(np.percentile(per_fp, 90))))
                print(f"{fam:>13} {m:>12} noise={noise:<4} RMSE={rows[-1]['rmse']:6.3f} "
                      f"TmaxErr={rows[-1]['tmax_err']:6.3f}  per-floorplan p10-p90 "
                      f"{rows[-1]['rmse_fp_p10']:.2f}-{rows[-1]['rmse_fp_p90']:.2f}", flush=True)
    (R / "stage2_extra_results.json").write_text(json.dumps(dict(physics_lambda=lam, physics_val=va_err, rows=rows), indent=1))
