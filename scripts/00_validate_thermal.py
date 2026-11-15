"""Hand-calculation checks for thermal.py (criteria.md: pass if within 1%)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import time

import numpy as np

from chipthermal.thermal import ChipParams, ThermalSolver

TOL = 0.01
p = ChipParams()
solver = ThermalSolver(p)
dx = p.width / p.nx
xc = (np.arange(p.nx) + 0.5) * dx
results = []


def report(name, numeric, exact):
    err = abs(numeric - exact) / abs(exact)
    results.append((name, numeric, exact, err, err <= TOL))


# Case 1: uniform power -> dT = q/h everywhere
q0 = 50e4  # W/m^2 (= 50 W/cm^2), assumed
T = solver.solve(np.full((p.ny, p.nx), q0))
exact = q0 / p.h
report("1 uniform: max dT [K]", T.max() - p.t_amb, exact)
report("1 uniform: min dT [K]", T.min() - p.t_amb, exact)
report("3 energy (case1) [W]", solver.heat_to_ambient(T), q0 * p.width * p.height)

# Case 2: power only in 0 < x < a (1D), adiabatic ends -> cosh solution
a = p.width / 2
L, b = p.width, p.width - a
m = np.sqrt(p.h / (p.k * p.thickness))
th_p = q0 / p.h
A = -th_p / (np.cosh(m * a) + np.sinh(m * a) / np.tanh(m * b))
B = -A * np.sinh(m * a) / np.sinh(m * b)
exact_profile = np.where(xc < a, th_p + A * np.cosh(m * xc), B * np.cosh(m * (L - xc)))
q = np.where(xc < a, q0, 0.0)[None, :].repeat(p.ny, axis=0)
T2 = solver.solve(q)
num_profile = T2[p.ny // 2] - p.t_amb
report("2 strip: dT at hottest cell [K]", num_profile[0], exact_profile[0])
report("2 strip: dT at coolest cell [K]", num_profile[-1], exact_profile[-1])
report("2 strip: max profile error / peak", np.max(np.abs(num_profile - exact_profile)) + exact_profile[0], exact_profile[0])
report("3 energy (case2) [W]", solver.heat_to_ambient(T2), q0 * a * p.height)

# Speed
t0 = time.perf_counter()
for _ in range(200):
    solver.solve(q)
per_solve_ms = (time.perf_counter() - t0) / 200 * 1e3

print(f"m = {m:.1f} 1/m, diffusion length 1/m = {1e3 / m:.2f} mm, m*dx = {m * dx:.3f}")
print(f"{'check':40s} {'numeric':>12s} {'exact':>12s} {'rel.err':>9s}  pass")
for name, n_, e_, err, ok in results:
    print(f"{name:40s} {n_:12.5f} {e_:12.5f} {err:9.2e}  {'PASS' if ok else 'FAIL'}")
print(f"solve time per power map (32x32, LU reused): {per_solve_ms:.3f} ms")
print("ALL PASS" if all(r[4] for r in results) else "SOME FAILED")
