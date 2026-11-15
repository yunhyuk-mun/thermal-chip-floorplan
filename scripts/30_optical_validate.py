"""Hand-calculation checks for optical_stage.py (pass if within 1%)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import numpy as np

from chipthermal.optical import Stage, Source, ALPHA_AL, ALPHA_POST, POST_L, MOUNT_A, MOUNT_B, PLATE

st = Stage()
p = PLATE
res = []


def check(name, num, exact, tol=0.01):
    err = abs(num - exact) / max(abs(exact), 1e-30)
    res.append((name, num, exact, err, err <= tol))


# 1) uniform heating: plate-sized source -> dT = P/(h A) everywhere; tilt 0; shift = alpha*dT*d
P = 15.0
T = st.temp([Source("uniform", P, p.width, p.width / 2, p.height / 2)])
dT = P / (p.h * p.width * p.height)
a = st.alignment(T)
d = MOUNT_B[0] - MOUNT_A[0]
check("1 uniform dT [K]", T.mean() - p.t_amb, dT)
check("1 uniform shift [um]", a["shift_um"], ALPHA_AL * dT * d * 1e6)
res.append(("1 uniform tilt [urad] (exact 0)", a["tilt_urad"], 0.0, abs(a["tilt_urad"]), abs(a["tilt_urad"]) < 1e-6))

# 2) post height difference by hand from the sampled base temperatures
T2 = st.temp([Source("motor", 10.0, 0.04, 0.08, 0.15)])
a2 = st.alignment(T2)
hand = ALPHA_POST * POST_L * (a2["ta"] - a2["tb"]) / d * 1e6
check("2 tilt = a*L*(Ta-Tb)/d [urad]", a2["tilt_urad"], hand)

# 3) energy balance
check("3 energy [W]", st.solver.heat_to_ambient(T2), 10.0)

# 4) bending curvature from through-thickness gradient, uniform case: kappa = a*(q_z t/k)/t
qz = 0.5 * p.h * dT
check("4 bend tilt uniform [urad]", a["bend_urad"], ALPHA_AL * (qz * p.thickness / p.k) / p.thickness * d * 1e6)

print(f"{'check':38s} {'numeric':>12s} {'exact':>12s} {'rel.err':>9s}")
for n_, v, e, err, ok in res:
    print(f"{n_:38s} {v:12.5f} {e:12.5f} {err:9.2e}  {'PASS' if ok else 'FAIL'}")
print("ALL PASS" if all(r[4] for r in res) else "SOME FAILED")
print(f"case2 (motor 10 W near A): Ta={a2['ta']:.2f} Tb={a2['tb']:.2f} tilt={a2['tilt_urad']:.1f} urad "
      f"shift={a2['shift_um']:.2f} um bend={a2['bend_urad']:.3f} urad Tmax={a2['tmax']:.1f}")
