"""Optical stage thermal deformation -> alignment error (criteria.md #26-33).

Aluminium plate with heat sources (light source, motor); two optical mounts on stainless posts.
Uses the validated 2D thermal solver. All values are assumption-based simulation.
"""
from dataclasses import dataclass

import numpy as np

from .thermal import ChipParams, ThermalSolver, overlap_fraction

PLATE = ChipParams(width=0.3, height=0.3, thickness=0.0127, k=167.0, h=15.0, t_amb=25.0, nx=60, ny=60)
ALPHA_AL, ALPHA_POST = 23.6e-6, 17.3e-6
POST_L = 0.075
MOUNT_A, MOUNT_B = (0.100, 0.150), (0.200, 0.150)


@dataclass
class Source:
    name: str
    power: float      # W
    size: float       # m, square side
    x: float          # m, centre
    y: float


SOURCES = [Source("광원", 5.0, 0.020, 0.0, 0.0), Source("모터", 10.0, 0.040, 0.0, 0.0)]


class Stage:
    def __init__(self, p=PLATE):
        self.p = p
        self.solver = ThermalSolver(p)
        self.dx = p.width / p.nx
        self.xc = (np.arange(p.nx) + 0.5) * self.dx
        self.yc = (np.arange(p.ny) + 0.5) * (p.height / p.ny)

    def flux(self, sources):
        """Power density map (W/m^2) with exact area overlap of each square source."""
        p = self.p
        q = np.zeros((p.ny, p.nx))
        for s in sources:
            q += overlap_fraction(s.x - s.size / 2, s.y - s.size / 2, s.size, s.size,
                                  p.width, p.height, p.nx, p.ny) * s.power
        return q / (self.dx * (p.height / p.ny))

    def temp(self, sources):
        return self.solver.solve(self.flux(sources))

    def sample(self, T, pt):
        """Bilinear interpolation of the cell-centred field at a point (m)."""
        return float(_bilinear(T, self.xc, self.yc, pt))

    def alignment(self, T):
        """Return tilt (urad), in-plane A-B distance change (um), plate bending tilt over A-B (urad)."""
        ta, tb = self.sample(T, MOUNT_A), self.sample(T, MOUNT_B)
        d = MOUNT_B[0] - MOUNT_A[0]
        tilt = ALPHA_POST * POST_L * (ta - tb) / d                     # rad
        xs = np.linspace(MOUNT_A[0], MOUNT_B[0], 201)
        dT = np.array([self.sample(T, (x, MOUNT_A[1])) for x in xs]) - self.p.t_amb
        shift = ALPHA_AL * np.trapezoid(dT, xs)                          # m
        # through-thickness gradient: local flux to bottom ~ h_bottom*(T-Tamb), half of h assumed bottom
        q_z = 0.5 * self.p.h * (T - self.p.t_amb)
        kappa = ALPHA_AL * (q_z * self.p.thickness / self.p.k) / self.p.thickness   # 1/m
        bend_tilt = float(np.mean(kappa)) * d                            # rad, uniform-curvature estimate
        return dict(tilt_urad=tilt * 1e6, shift_um=shift * 1e6, bend_urad=bend_tilt * 1e6, ta=ta, tb=tb,
                    tmax=float(T.max()))


def _bilinear(T, xc, yc, pt):
    x, y = pt
    i = np.clip(np.searchsorted(xc, x) - 1, 0, len(xc) - 2)
    j = np.clip(np.searchsorted(yc, y) - 1, 0, len(yc) - 2)
    fx = np.clip((x - xc[i]) / (xc[i + 1] - xc[i]), 0, 1)
    fy = np.clip((y - yc[j]) / (yc[j + 1] - yc[j]), 0, 1)
    return (T[j, i] * (1 - fx) * (1 - fy) + T[j, i + 1] * fx * (1 - fy)
            + T[j + 1, i] * (1 - fx) * fy + T[j + 1, i + 1] * fx * fy)


def place(sources, positions):
    return [Source(s.name, s.power, s.size, *pos) for s, pos in zip(sources, positions)]
