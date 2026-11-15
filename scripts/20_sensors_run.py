"""Stage 1.5 experiment: sensor placement x reconstruction x sensor noise on the thermal-aware layout."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import json

import numpy as np

from chipthermal.paths import SENSORS as R
from chipthermal.sensors import (setup, place_grid, place_hot, place_data, pca_basis, recon_nearest, recon_pca,
                                 K_LIST, NOISE, N_SCEN, N_TRAIN)


def evaluate():
    prob, xywh, T = setup()
    n = prob.solver.p.nx
    train, test = T[:N_TRAIN], T[N_TRAIN:]
    rng = np.random.default_rng(7)
    rows, examples = [], {}
    for k in K_LIST:
        mu, phi = pca_basis(train, k)
        placements = {"grid": place_grid(k, n), "hot": place_hot(k, prob, xywh, n), "data": place_data(k, train)}
        for pname, sens in placements.items():
            for noise in NOISE:
                read = test[:, sens] + (rng.uniform(-noise, noise, size=(len(test), len(sens))) if noise else 0.0)
                for rname, est in (("nearest", recon_nearest(read, sens, n)), ("pca", recon_pca(read, sens, mu, phi))):
                    rmse = float(np.sqrt(np.mean((est - test) ** 2)))
                    tmax_err = float(np.mean(np.abs(est.max(1) - test.max(1))))
                    rows.append(dict(k=k, placement=pname, recon=rname, noise=noise, rmse=rmse, tmax_err=tmax_err))
                    if k == 8 and noise == 0.0:
                        examples[f"{pname}_{rname}"] = dict(sensors=sens, est=est[0].tolist())
    examples["true"] = test[0].tolist()
    meta = dict(n_scen=N_SCEN, n_train=N_TRAIN, k_list=K_LIST, noise=NOISE, grid=n,
                test_tmax_mean=float(test.max(1).mean()), test_tmax_range=[float(test.max(1).min()), float(test.max(1).max())])
    (R / "sensors_results.json").write_text(json.dumps(dict(meta=meta, rows=rows, examples=examples)))
    return meta, rows, examples, xywh, prob


if __name__ == "__main__":
    meta, rows, *_ = evaluate()
    print(json.dumps(meta))
    print(f"{'k':>3} {'place':>5} {'recon':>8} {'noise':>5} {'RMSE':>7} {'TmaxErr':>8}")
    for r in rows:
        print(f"{r['k']:>3} {r['placement']:>5} {r['recon']:>8} {r['noise']:>5} {r['rmse']:7.3f} {r['tmax_err']:8.3f}")
