"""Stage 2: neural-network reconstruction of the full chip temperature map from a few sensors.

Experiment 2 (unseen floorplans, main): 200 random legal floorplans x 50 workloads,
train on 135 floorplans, validate on 15, test on 50 floorplans from a separate start; fixed 4x4 grid sensors.
Experiment 1 (same floorplan): stage-1.5 floorplan, 360 train / 40 val / 100 test, data-based 8 or 16 sensors.
Baselines: nearest-sensor fill, PCA pattern reconstruction (rank chosen on validation).
All values are assumption-based simulation.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import json
import time

import numpy as np
import torch

from chipthermal.floorplan import load_problem
from chipthermal.nn import N, T_AMB, train_model, predict, rmse, tmax_err, noisy
from chipthermal.paths import STAGE2 as R
from chipthermal.sensors import setup as setup_same, block_responses, workloads, pca_basis, place_data, recon_nearest, recon_pca

torch.set_num_threads(8)
NOISE = [0.0, 0.5, 1.0]
GRID16 = [int((i + 0.5) / 4 * N) * N + int((j + 0.5) / 4 * N) for i in range(4) for j in range(4)]


def build_unseen_dataset(n_scen=50, seed=123):
    prob = load_problem()
    fps = np.load(R / "stage2_floorplans.npz")["xywh"]
    rng = np.random.default_rng(seed)
    T, D = [], []
    for xywh in fps:
        G, dens = block_responses(prob, xywh)
        T.append(workloads(prob, G, n_scen, rng)); D.append(dens)
    return np.array(T, dtype=np.float32), np.array(D, dtype=np.float32)   # (F,S,1024), (F,1024)


def exp_unseen():
    t0 = time.perf_counter()
    T, D = build_unseen_dataset()
    F, S, _ = T.shape
    dnorm = D / D.max()
    # floorplans 0-149 come from 3 random-walk starts, 150-199 from a 4th start never used in training
    va = np.array([i for i in range(150) if i % 10 == 9])          # 15 validation floorplans
    tr = np.array([i for i in range(150) if i % 10 != 9])          # 135 training floorplans
    te = np.arange(150, 200)                                       # 50 test floorplans, separate start
    flat = lambda idx: (T[idx].reshape(-1, N * N), np.repeat(dnorm[idx], S, axis=0))
    Ttr, Dtr = flat(tr); Tva, Dva = flat(va); Tte, Dte = flat(te)
    print(f"dataset {T.shape} built in {time.perf_counter() - t0:.0f}s", flush=True)

    # PCA baseline: pooled patterns from training floorplans, rank chosen on validation
    ranks = [2, 4, 6, 8, 12, 16]
    va_err = {r: rmse(recon_pca(Tva[:, GRID16], GRID16, *pca_basis(Ttr, r)), Tva) for r in ranks}
    r_best = min(va_err, key=va_err.get)
    mu, phi = pca_basis(Ttr, r_best)

    models = {}
    for name, use_fp in (("nn_floorplan", True), ("nn_sensor_only", False)):
        t1 = time.perf_counter()
        models[name], vbest = train_model(Ttr, Dtr, GRID16, use_fp, Tva, Dva, epochs=20, log=name)
        print(f"{name}: val RMSE {vbest:.3f}, trained in {time.perf_counter() - t1:.0f}s", flush=True)
        models[name + "_sec"] = time.perf_counter() - t1

    rows = []
    for noise in NOISE:
        rng = np.random.default_rng(99)
        read = noisy(Tte, GRID16, noise, rng)
        ests = {"nearest": recon_nearest(read, GRID16, N), "pca_pooled": recon_pca(read, GRID16, mu, phi)}
        for name, use_fp in (("nn_floorplan", True), ("nn_sensor_only", False)):
            ests[name] = predict(models[name], Tte, Dte, GRID16, noise, np.random.default_rng(99), use_fp)
        for m, e in ests.items():
            rows.append(dict(exp="unseen", method=m, k=16, noise=noise, rmse=rmse(e, Tte), tmax_err=tmax_err(e, Tte)))
    # example for figure: first test floorplan, first workload
    rng = np.random.default_rng(99)
    ex = dict(true=Tte[0].tolist(), dens=Dte[0].tolist(),
              nearest=recon_nearest(Tte[:1, GRID16], GRID16, N)[0].tolist(),
              pca_pooled=recon_pca(Tte[:1, GRID16], GRID16, mu, phi)[0].tolist(),
              nn_floorplan=predict(models["nn_floorplan"], Tte[:1], Dte[:1], GRID16, 0.0, rng, True)[0].tolist(),
              nn_sensor_only=predict(models["nn_sensor_only"], Tte[:1], Dte[:1], GRID16, 0.0, rng, False)[0].tolist(),
              sensors=GRID16)
    meta = dict(pca_rank=r_best, pca_val=va_err, train_seconds={k: v for k, v in models.items() if k.endswith("_sec")},
                n_train=len(Ttr), n_test=len(Tte))
    torch.save(models["nn_floorplan"].state_dict(), R / "stage2_nn_floorplan.pt")
    return rows, ex, meta


def exp_same():
    prob, xywh, T = setup_same()
    _, dens = block_responses(prob, xywh)
    Dn = np.repeat((dens / dens.max())[None, :].astype(np.float32), len(T), 0)
    Ttr, Tte, Dtr, Dte = T[:360].astype(np.float32), T[400:].astype(np.float32), Dn[:360], Dn[400:]
    Tva, Dva = T[360:400].astype(np.float32), Dn[360:400]
    rows = []
    for k in (8, 16):
        sens = place_data(k, T[:400])
        mu, phi = pca_basis(T[:400], k)
        t1 = time.perf_counter()
        model, _ = train_model(Ttr, Dtr, sens, True, Tva, Dva, epochs=300, bs=32, log=f"same k={k}")
        print(f"same-floorplan k={k} trained in {time.perf_counter() - t1:.0f}s", flush=True)
        for noise in NOISE:
            rng = np.random.default_rng(99)
            read = noisy(Tte, sens, noise, rng)
            e_pca = recon_pca(read, sens, mu, phi)
            e_nn = predict(model, Tte, Dte, sens, noise, np.random.default_rng(99), True)
            rows += [dict(exp="same", method="pca", k=k, noise=noise, rmse=rmse(e_pca, Tte), tmax_err=tmax_err(e_pca, Tte)),
                     dict(exp="same", method="nn_floorplan", k=k, noise=noise, rmse=rmse(e_nn, Tte), tmax_err=tmax_err(e_nn, Tte))]
    return rows


if __name__ == "__main__":
    if "--same-only" in sys.argv:          # rerun only experiment 1, keep saved experiment-2 results
        old = json.loads((R / "stage2_results.json").read_text())
        meta, ex = old["meta"], old["example"]
        rows_u = [r for r in old["rows"] if r["exp"] == "unseen"]
    else:
        rows_u, ex, meta = exp_unseen()
    rows_s = exp_same()
    (R / "stage2_results.json").write_text(json.dumps(dict(meta=meta, rows=rows_u + rows_s, example=ex)))
    print(json.dumps(meta, indent=1, default=str))
    for r in rows_u + rows_s:
        print(f"{r['exp']:>6} {r['method']:>15} k={r['k']:>2} noise={r['noise']:<4} RMSE={r['rmse']:7.3f} TmaxErr={r['tmax_err']:7.3f}")
