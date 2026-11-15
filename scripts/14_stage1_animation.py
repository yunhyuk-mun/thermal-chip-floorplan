"""Side-by-side animation: wirelength-only vs thermal-aware annealing on ami33.

Re-runs the same seeds as fig1 (w=0 seed 2, w=1 seed 0) with snapshots, then renders
results/stage1_animation/anim_compare.mp4 and results/stage1_animation/anim_compare.gif. All values are assumption-based simulation.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

from chipthermal.floorplan import load_problem, normalizers, anneal

from chipthermal.paths import ANIM as R
CACHE = R / "anim_snapshots.pkl"
ITERS, EVERY, PENALTY = 20000, 200, 50.0
RUNS = [("배선만 고려", 0.0, 2), ("배선 + 온도 고려", 1.0, 0)]

prob = load_problem()
if CACHE.exists() and "--rerun" not in sys.argv:
    snaps, finals = pickle.loads(CACHE.read_bytes())
else:
    norm = normalizers(prob, n=200)
    snaps, finals = [], []
    for title, wt, seed in RUNS:
        s = []
        best = anneal(prob, wt, norm, iters=ITERS, seed=seed, penalty=PENALTY, snapshot_every=EVERY, snapshots=s)
        e = best[4]
        finals.append(dict(it=ITERS, x=e["x"], y=e["y"], w=e["w"], h=e["h"], temp=e["temp"],
                           tmax=e["tmax"], hpwl=e["hpwl"], overflow=e["overflow"]))
        snaps.append(s)
        print(title, "final", round(e["tmax"], 2), round(e["hpwl"]), flush=True)
    CACHE.write_bytes(pickle.dumps((snaps, finals)))

# frames: snapshots, then hold the best result
HOLD = 30
n_frames = len(snaps[0]) + HOLD
S = prob.outline
hot = set(prob.hot)
# colour scale fitted to the final layouts so their differences are visible; early hot frames saturate
fin_t = np.concatenate([[f["temp"].min(), f["temp"].max()] for f in finals])
vmin, vmax = np.floor(fin_t.min()) - 2, np.ceil(fin_t.max()) + 4

BG, FG, SUB = "#0f1115", "#e8e6e1", "#9a978f"
HOT, COOL = "#38e1ff", "#ffffff"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                     "text.color": FG, "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB})
fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.035, 0.10, 0.62], height_ratios=[1, 1],
                      left=0.03, right=0.975, top=0.86, bottom=0.13, wspace=0.08, hspace=0.55)
ax_maps = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])]
ax_cb = fig.add_subplot(gs[:, 2])
ax_t = fig.add_subplot(gs[0, 4])
ax_w = fig.add_subplot(gs[1, 4])
fig.text(0.03, 0.94, "칩 블록 배치가 진화하는 과정", fontsize=20, weight="bold")
fig.text(0.03, 0.905, "ami33 · 33개 블록 · 총 50 W · 뜨거운 블록 4개(하늘색) · 가정 기반 시뮬레이션",
         fontsize=11, color=SUB)
cols = ["#b9b6ad", "#ffb347"]

ims, rect_groups, titles, stats = [], [], [], []
for k, ax in enumerate(ax_maps):
    ax.set_facecolor(BG)
    im = ax.imshow(snaps[k][0]["temp"], origin="lower", extent=[0, S, 0, S], cmap="inferno",
                   vmin=vmin, vmax=vmax, interpolation="bicubic")
    ax.add_patch(Rectangle((0, 0), S, S, fill=False, ec=SUB, lw=1.0, ls="--"))
    ax.set_xlim(-0.04 * S, 1.12 * S); ax.set_ylim(-0.04 * S, 1.12 * S)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    rects = []
    for i in range(len(prob.names)):
        r = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=6",
                           fill=False, ec=HOT if i in hot else COOL,
                           lw=2.6 if i in hot else 0.7, alpha=1.0 if i in hot else 0.55)
        ax.add_patch(r); rects.append(r)
    rect_groups.append(rects); ims.append(im)
    titles.append(ax.text(0.0, 1.10 * S, RUNS[k][0], fontsize=15, weight="bold", color=cols[k]))
    stats.append(ax.text(0.0, 1.045 * S, "", fontsize=11, color=FG))

cb = fig.colorbar(ims[1], cax=ax_cb, extend="max")
cb.set_label("온도 (°C)", color=SUB); cb.outline.set_visible(False)
cb.ax.yaxis.set_tick_params(color=SUB)

its = [f["it"] for f in snaps[0]]
for ax, key, label in ((ax_t, "tmax", "칩 최고 온도 (°C)"), (ax_w, "hpwl", "배선 길이")):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color("#2a2d33")
    ax.set_title(label, color=FG, fontsize=12, loc="left")
    ax.set_xlim(0, ITERS)
    ys = [f[key] for run in snaps for f in run[5:]]
    pad = 0.05 * (max(ys) - min(ys))
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.grid(color="#22252b", lw=0.6)
ax_w.set_xlabel("탐색 반복 횟수")
ax_t.tick_params(labelbottom=False)
# faint = current state being explored, bold = best found so far
cur_t = [ax_t.plot([], [], color=c, lw=0.8, alpha=0.35)[0] for c in cols]
cur_w = [ax_w.plot([], [], color=c, lw=0.8, alpha=0.35)[0] for c in cols]
lines_t = [ax_t.plot([], [], color=c, lw=2.4, label=RUNS[k][0])[0] for k, c in enumerate(cols)]
lines_w = [ax_w.plot([], [], color=c, lw=2.4)[0] for c in cols]
leg = ax_t.legend(loc="upper right", fontsize=9, frameon=False)
for t, c in zip(leg.get_texts(), cols):
    t.set_color(c)
fig.text(0.03, 0.065, "오른쪽 그래프: 굵은 선 = 지금까지 찾은 가장 좋은 배치의 값 · 옅은 선 = 탐색 중인 배치",
         fontsize=9, color=SUB, ha="left")
prog = fig.add_axes([0.03, 0.035, 0.945, 0.008]); prog.set_xlim(0, 1); prog.axis("off")
prog_bar = prog.barh(0, 0, color="#ffb347", height=1)[0]


NORM = normalizers(prob, n=200)


def best_so_far(k, m):
    """tmax/hpwl of the lowest-cost snapshot seen so far, using run k's own annealing cost."""
    wt = RUNS[k][1]
    bt, bw, best_c = [], [], None
    for s in snaps[k][:m]:
        c = s["hpwl"] / NORM[0] + wt * (s["tmax"] - prob.solver.p.t_amb) / NORM[1] + PENALTY * s["overflow"]
        if best_c is None or c < best_c:
            best_c, cur = c, (s["tmax"], s["hpwl"])
        bt.append(cur[0]); bw.append(cur[1])
    return bt, bw


def frame_state(k, f):
    run = snaps[k]
    return run[f] if f < len(run) else finals[k]


def update(f):
    done = f >= len(snaps[0])
    for k in range(2):
        st = frame_state(k, f)
        ims[k].set_data(st["temp"])
        for i, r in enumerate(rect_groups[k]):
            r.set_bounds(st["x"][i], st["y"][i], st["w"][i], st["h"][i])
        tag = "  최종(가장 좋은 배치)" if done else ""
        stats[k].set_text(f"최고 {st['tmax']:.1f}°C   배선 {st['hpwl']:,.0f}{tag}")
        m = min(f + 1, len(snaps[k]))
        vt = [s["tmax"] for s in snaps[k][:m]]
        vw = [s["hpwl"] for s in snaps[k][:m]]
        cur_t[k].set_data(its[:m], vt)
        cur_w[k].set_data(its[:m], vw)
        bt, bw = best_so_far(k, m)
        lines_t[k].set_data(its[:m], bt)
        lines_w[k].set_data(its[:m], bw)
    prog_bar.set_width(min(1.0, (f + 1) / len(snaps[0])))
    return []


ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / 15, blit=False)
ani.save(R / "anim_compare.mp4", writer=animation.FFMpegWriter(fps=15, bitrate=4000), savefig_kwargs={"facecolor": BG})
update(n_frames - 1)
fig.savefig(R / "anim_compare_final.png", dpi=150, facecolor=BG)
for f in (len(snaps[0]) // 4, len(snaps[0]) // 2):
    update(f)
    fig.savefig(R / f"anim_frame_{f:03d}.png", dpi=80, facecolor=BG)
print("saved", n_frames, "frames")
