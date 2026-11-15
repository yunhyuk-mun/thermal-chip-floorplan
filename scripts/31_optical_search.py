"""Search heat-source placement on the optical stage and test sensor-based tilt correction.

The model is linear in source power, so each source's response at each candidate position is solved once
and every layout is a sum. Writes results/optical_stage/optical_results.json and figures fig_o1..o3.
All values are assumption-based simulation.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np

from chipthermal.optical import Stage, SOURCES, Source, MOUNT_A, MOUNT_B, ALPHA_POST, POST_L, ALPHA_AL, PLATE

from chipthermal.paths import OPTICAL as R
st = Stage()
p = PLATE
d = MOUNT_B[0] - MOUNT_A[0]
cand = [(x, y) for y in np.linspace(0.03, 0.27, 13) for x in np.linspace(0.03, 0.27, 13)]


def clear_of_mounts(pos, size, margin=0.015):
    return all(max(abs(pos[0] - m[0]), abs(pos[1] - m[1])) > size / 2 + margin for m in (MOUNT_A, MOUNT_B))


xs_line = np.linspace(MOUNT_A[0], MOUNT_B[0], 201)
resp = []   # per source: list of (pos, T_rise_map, ta, tb, integral)
for s in SOURCES:
    rows = []
    for pos in cand:
        if not clear_of_mounts(pos, s.size):
            continue
        T = st.temp([Source(s.name, s.power, s.size, *pos)]) - p.t_amb
        ta, tb = st.sample(T, MOUNT_A), st.sample(T, MOUNT_B)
        integ = np.trapezoid([st.sample(T, (x, MOUNT_A[1])) for x in xs_line], xs_line)
        rows.append((pos, T, ta, tb, integ))
    resp.append(rows)
print("candidates:", [len(r) for r in resp])

layouts = []
for i, (pa, Ta_, ta1, tb1, in1) in enumerate(resp[0]):
    for j, (pb, Tb_, ta2, tb2, in2) in enumerate(resp[1]):
        if max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1])) < (SOURCES[0].size + SOURCES[1].size) / 2:
            continue   # sources overlap
        tilt = ALPHA_POST * POST_L * ((ta1 + ta2) - (tb1 + tb2)) / d * 1e6
        shift = ALPHA_AL * (in1 + in2) * 1e6
        layouts.append(dict(i=i, j=j, tilt=tilt, shift=shift))
tilts = np.array([abs(l["tilt"]) for l in layouts]); shifts = np.array([l["shift"] for l in layouts])
worst = layouts[int(np.argmax(tilts))]
# ties (symmetric layouts with zero tilt): take the one with the smallest shift, so the pick is stable
best = layouts[int(np.lexsort((shifts, np.round(tilts, 6)))[0])]
# "engineer's default": both sources pushed to the far corner region away from the optics (minimum shift)
far = layouts[int(np.argmin(shifts))]


def describe(l):
    pa, pb = resp[0][l["i"]][0], resp[1][l["j"]][0]
    T = p.t_amb + resp[0][l["i"]][1] + resp[1][l["j"]][1]
    a = st.alignment(T)
    return dict(light=[round(v * 1e3) for v in pa], motor=[round(v * 1e3) for v in pb], T=T, **a)


cases = {"worst_tilt": describe(worst), "min_shift_far": describe(far), "best_tilt": describe(best)}

# sensor-based correction over all layouts: estimate tilt from post-base sensors with noise
rng = np.random.default_rng(5)
corr = {}
for noise in (0.0, 0.1, 0.5):
    e = rng.uniform(-noise, noise, size=(len(layouts), 2)) if noise else np.zeros((len(layouts), 2))
    est_err = np.abs(ALPHA_POST * POST_L * (e[:, 0] - e[:, 1]) / d * 1e6)
    corr[str(noise)] = dict(mean_abs_err_urad=float(est_err.mean()), max_abs_err_urad=float(est_err.max()))
no_sensor_err = float(tilts.mean())

out = dict(n_layouts=len(layouts), tilt_abs_mean=float(tilts.mean()), tilt_abs_max=float(tilts.max()),
           tilt_abs_min=float(tilts.min()), shift_range=[float(shifts.min()), float(shifts.max())],
           cases={k: {kk: vv for kk, vv in v.items() if kk != "T"} for k, v in cases.items()},
           sensor_correction=corr, no_sensor_mean_err_urad=no_sensor_err,
           bend_vs_tilt_worst=[cases["worst_tilt"]["bend_urad"], cases["worst_tilt"]["tilt_urad"]])
(R / "optical_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))

# ---------------- figures ----------------
BG, FG, SUB = "#0f1115", "#e8e6e1", "#9a978f"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.color": FG,
                     "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB, "axes.edgecolor": "#2a2d33"})
W = p.width * 1e3

# Fig O1: three layouts
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor=BG)
vmin = min(c["T"].min() for c in cases.values()); vmax = max(c["T"].max() for c in cases.values())
titles = {"worst_tilt": "광축이 가장 많이 기우는 배치", "min_shift_far": "열원을 멀리 몰아 둔 배치", "best_tilt": "기울기가 가장 작은 배치"}
for ax, (key, c) in zip(axes, cases.items()):
    im = ax.imshow(c["T"], origin="lower", extent=[0, W, 0, W], cmap="inferno", vmin=vmin, vmax=vmax, interpolation="bicubic")
    for s, pos, col in ((SOURCES[0], c["light"], "#ffe066"), (SOURCES[1], c["motor"], "#ff7a59")):
        sz = s.size * 1e3
        ax.add_patch(Rectangle((pos[0] - sz / 2, pos[1] - sz / 2), sz, sz, fill=False, ec=col, lw=2))
        ty = pos[1] + sz / 2 + 6 if s.name == "광원" else pos[1] - sz / 2 - 14
        ty = min(max(ty, 6), W - 12)
        ax.text(pos[0] + (0 if s.name == "광원" else 0), ty, s.name, color=col, ha="center", fontsize=9, weight="bold")
    for m, lab in ((MOUNT_A, "A"), (MOUNT_B, "B")):
        ax.add_patch(Circle((m[0] * 1e3, m[1] * 1e3), 6, fc="none", ec="#38e1ff", lw=2))
        ax.text(m[0] * 1e3, m[1] * 1e3 - 16, lab, color="#38e1ff", ha="center", fontsize=10)
    ax.plot([MOUNT_A[0] * 1e3, MOUNT_B[0] * 1e3], [MOUNT_A[1] * 1e3] * 2, color="#38e1ff", lw=1, ls="--")
    ax.set_title(f"{titles[key]}\n광축 기울기 {c['tilt_urad']:+.1f} μrad · A–B 거리 변화 {c['shift_um']:.1f} μm",
                 color=FG, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01); cb.set_label("판 온도 (°C)", color=SUB); cb.outline.set_visible(False)
fig.suptitle("알루미늄 광학 스테이지(300 mm): 열원 위치에 따른 광학 정렬 오차 (하늘색 = 광학 장착점, 가정 기반 시뮬레이션)",
             color=FG, fontsize=13, x=0.01, ha="left")
fig.savefig(R / "fig_o1_layouts.png", dpi=140, facecolor=BG, bbox_inches="tight"); plt.close(fig)

# Fig O2: tilt map vs motor position with the light source at its best-case position
li = best["i"]
grid = np.full((13, 13), np.nan)
for j, (pb, Tb_, ta2, tb2, in2) in enumerate(resp[1]):
    pa = resp[0][li][0]
    if max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1])) < (SOURCES[0].size + SOURCES[1].size) / 2:
        continue
    _, _, ta1, tb1, _ = resp[0][li]
    gi, gj = cand.index(pb) // 13, cand.index(pb) % 13
    grid[gi, gj] = ALPHA_POST * POST_L * ((ta1 + ta2) - (tb1 + tb2)) / d * 1e6
fig, ax = plt.subplots(figsize=(6.6, 5.6), facecolor=BG); ax.set_facecolor(BG)
lim = np.nanmax(np.abs(grid))
im = ax.imshow(grid, origin="lower", extent=[15, W - 15, 15, W - 15], cmap="coolwarm", vmin=-lim, vmax=lim)
for m, lab in ((MOUNT_A, "A"), (MOUNT_B, "B")):
    ax.add_patch(Circle((m[0] * 1e3, m[1] * 1e3), 6, fc="none", ec="k", lw=2)); ax.text(m[0] * 1e3, m[1] * 1e3 - 16, lab, ha="center", color="k")
pa = resp[0][li][0]; ax.add_patch(Rectangle((pa[0] * 1e3 - 10, pa[1] * 1e3 - 10), 20, 20, fill=False, ec="#ffe066", lw=2))
cb = fig.colorbar(im, ax=ax); cb.set_label("광축 기울기 (μrad)", color=SUB); cb.outline.set_visible(False)
ax.set_title("모터(10 W)를 어디에 두면 광축이 얼마나 기우나\n(노란 칸 = 광원 위치 고정, 빈칸 = 장착점·광원과 겹침)", color=FG, fontsize=11, loc="left")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
fig.savefig(R / "fig_o2_motor_tilt_map.png", dpi=140, facecolor=BG, bbox_inches="tight"); plt.close(fig)

# Fig O3: mechanism size + sensor correction
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=BG)
ax = axes[0]; ax.set_facecolor(BG)
vals = [abs(cases["worst_tilt"]["tilt_urad"]), abs(cases["worst_tilt"]["bend_urad"])]
bars = ax.bar(["포스트 높이 차\n(장착점 온도 차)", "판 휨\n(두께 방향 온도 차)"], vals, color=["#ff7a59", "#7f77dd"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.03, f"{v:.2f} μrad", ha="center", color=FG)
ax.set_title("어떤 원인이 광축을 기울이나 (가장 나쁜 배치)", color=FG, loc="left", fontsize=12)
ax.set_ylabel("광축 기울기 크기 (μrad)")
ax = axes[1]; ax.set_facecolor(BG)
labels = ["센서 없음\n(0으로 가정)"] + [f"센서 2개\n±{n}°C" if float(n) else "센서 2개\n오차 없음" for n in corr]
vals = [no_sensor_err] + [corr[n]["mean_abs_err_urad"] for n in corr]
bars = ax.bar(labels, vals, color=["#b9b6ad", "#38e1ff", "#38e1ff", "#38e1ff"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", color=FG)
ax.set_title("포스트 밑 센서 2개로 기울기를 추정해 보정하면 남는 오차\n(모든 배치 평균)", color=FG, loc="left", fontsize=12)
ax.set_ylabel("남는 기울기 오차 (μrad)")
fig.savefig(R / "fig_o3_mechanism_sensor.png", dpi=140, facecolor=BG, bbox_inches="tight"); plt.close(fig)
print("figures saved")
