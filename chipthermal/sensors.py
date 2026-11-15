"""Sensor placement and full-map reconstruction (stage 1.5), plus per-block thermal responses.

The thermal model is linear, so each block's temperature response to 1 W is solved once and any
workload is a linear combination. All values are assumption-based simulation.
"""
import numpy as np
import scipy.linalg as sla

from .floorplan import load_problem
from .paths import STAGE1
from .thermal import overlap_fraction

K_LIST = [2, 4, 8, 16]
NOISE = [0.0, 0.5, 1.0]          # uniform +/- deg C
N_SCEN, N_TRAIN = 500, 400
GRID_SHAPE = {2: (1, 2), 4: (2, 2), 8: (2, 4), 16: (4, 4)}


def block_responses(prob, xywh):
    """Per-block temperature rise per watt (n_blocks, cells) and design power density map (cells,) in W/m^2."""
    x, y, w, h = xywh
    p = prob.solver.p
    cell_m2 = (p.width / p.nx) * (p.height / p.ny)
    G, dens = [], np.zeros((p.ny, p.nx))
    for i in range(len(w)):
        flux_per_w = overlap_fraction(x[i], y[i], w[i], h[i], prob.outline, prob.outline, p.nx, p.ny) / cell_m2
        G.append((prob.solver.solve(flux_per_w) - p.t_amb).ravel())
        dens += flux_per_w * prob.power[i]
    return np.array(G), dens.ravel()


def workloads(prob, G, n, rng, low=0.2, high=1.5):
    """Random workloads: each block's power multiplied by U(low, high). Returns temperature maps (n, cells)."""
    P = prob.power[None, :] * rng.uniform(low, high, size=(n, G.shape[0]))
    return prob.solver.p.t_amb + P @ G


def setup():
    """Stage-1.5 setting: thermal-aware final layout (weight 1.0, seed 0) and 500 workloads (seed 42)."""
    prob = load_problem()
    xywh = np.load(STAGE1 / "stage1_layouts.npz")["w1.0_s0_xywh"]
    G, _ = block_responses(prob, xywh)
    T = workloads(prob, G, N_SCEN, np.random.default_rng(42))
    return prob, xywh, T


def cell_of(px, py, outline, n):
    return int(np.clip(py / outline * n, 0, n - 1)) * n + int(np.clip(px / outline * n, 0, n - 1))


def place_grid(k, n):
    r, c = GRID_SHAPE[k]
    return [int((i + 0.5) / r * n) * n + int((j + 0.5) / c * n) for i in range(r) for j in range(c)]


def place_hot(k, prob, xywh, n):
    x, y, w, h = xywh
    dens = prob.power / (w * h)
    cells = []
    for i in np.argsort(-dens):
        c = cell_of(x[i] + w[i] / 2, y[i] + h[i] / 2, prob.outline, n)
        if c not in cells:
            cells.append(c)
        if len(cells) == k:
            break
    return cells


def pca_basis(train, k):
    mu = train.mean(0)
    _, _, vt = np.linalg.svd(train - mu, full_matrices=False)
    return mu, vt[:k].T                                               # (cells,), (cells, k)


def place_data(k, train):
    """Data-based placement: QR with column pivoting on the top-k PCA modes."""
    _, phi = pca_basis(train, k)
    _, _, piv = sla.qr(phi.T, pivoting=True, mode="economic")
    return [int(c) for c in piv[:k]]


def recon_nearest(readings, sensors, n):
    yy, xx = np.divmod(np.arange(n * n), n)
    sy, sx = np.divmod(np.array(sensors), n)
    d = (yy[:, None] - sy[None, :]) ** 2 + (xx[:, None] - sx[None, :]) ** 2
    return readings[:, np.argmin(d, axis=1)]


def recon_pca(readings, sensors, mu, phi):
    A = phi[sensors, :]
    coef, *_ = np.linalg.lstsq(A, (readings - mu[sensors]).T, rcond=None)
    return mu + (phi @ coef).T
