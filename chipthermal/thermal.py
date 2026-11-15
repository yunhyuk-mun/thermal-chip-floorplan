"""2D steady-state chip thermal solver (finite volume, cell-centered).

Model (criteria.md):  k*t*lap(T) - h*(T - T_amb) + q'' = 0,  adiabatic side edges.
All parameter values are assumptions documented in criteria.md.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class ChipParams:
    width: float = 10e-3       # m
    height: float = 10e-3      # m
    thickness: float = 0.5e-3  # m, silicon die
    k: float = 150.0           # W/(m K), silicon
    h: float = 2.0e4           # W/(m^2 K), equivalent heat transfer to ambient below the die
    t_amb: float = 25.0        # deg C
    nx: int = 32
    ny: int = 32


def build_matrix(p: ChipParams) -> sp.csr_matrix:
    """Conductance matrix A so that A @ theta = q_cell, theta = T - T_amb."""
    dx, dy = p.width / p.nx, p.height / p.ny
    gx = p.k * p.thickness * dy / dx   # W/K between x-neighbours
    gy = p.k * p.thickness * dx / dy   # W/K between y-neighbours
    gs = p.h * dx * dy                 # W/K from each cell to ambient
    n = p.nx * p.ny
    idx = np.arange(n).reshape(p.ny, p.nx)

    rows, cols, vals = [], [], []
    diag = np.full(n, gs)
    for (a, b, g) in (
        (idx[:, :-1].ravel(), idx[:, 1:].ravel(), gx),
        (idx[:-1, :].ravel(), idx[1:, :].ravel(), gy),
    ):
        rows += [a, b]
        cols += [b, a]
        vals += [np.full(a.size, -g), np.full(a.size, -g)]
        np.add.at(diag, a, g)
        np.add.at(diag, b, g)
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )


class ThermalSolver:
    def __init__(self, p: ChipParams = ChipParams()):
        self.p = p
        self.cell_area = (p.width / p.nx) * (p.height / p.ny)
        self._lu = spla.splu(build_matrix(p).tocsc())  # factor once, reuse for every power map

    def solve(self, q_flux: np.ndarray) -> np.ndarray:
        """q_flux: (ny, nx) power density in W/m^2. Returns temperature (deg C), same shape."""
        assert q_flux.shape == (self.p.ny, self.p.nx)
        theta = self._lu.solve((q_flux * self.cell_area).ravel())
        return theta.reshape(q_flux.shape) + self.p.t_amb

    def heat_to_ambient(self, temp: np.ndarray) -> float:
        """Total heat leaving through the bottom, W (should equal total input power)."""
        return float(np.sum(self.p.h * (temp - self.p.t_amb) * self.cell_area))


def overlap_fraction(x0, y0, w, h, width, height, nx, ny):
    """Fraction of a w x h rectangle (lower-left x0, y0) that falls in each grid cell, shape (ny, nx).

    Shared by the floorplan power map, sensor responses, stage-2 data and the optical stage.
    """
    ex = np.linspace(0.0, width, nx + 1)
    ey = np.linspace(0.0, height, ny + 1)
    ox = np.clip(np.minimum(x0 + w, ex[1:]) - np.maximum(x0, ex[:-1]), 0, None)
    oy = np.clip(np.minimum(y0 + h, ey[1:]) - np.maximum(y0, ey[:-1]), 0, None)
    return np.outer(oy, ox) / (w * h)
