"""Figures for stage 2 (sensor-reconstruction neural network). Reads results/stage2_nn/stage2_results.json."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chipthermal.nn import N
from chipthermal.paths import STAGE2 as R

d = json.loads((R / "stage2_results.json").read_text())
rows, ex = d["rows"], d["example"]
BG, FG, SUB, GRID = "#0f1115", "#e8e6e1", "#9a978f", "#22252b"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.color": FG,
                     "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB, "axes.edgecolor": "#2a2d33"})
NAMES = {"nearest": "가까운 센서 값", "pca_pooled": "패턴 복원(공통)", "nn_sensor_only": "신경망(센서만)",
         "nn_floorplan": "신경망(센서+평면도)", "pca": "패턴 복원"}
COLORS = {"nearest": "#b9b6ad", "pca_pooled": "#ff7a59", "nn_sensor_only": "#7c8cff", "nn_floorplan": "#38e1ff",
          "pca": "#ff7a59"}


def get(exp, method, k, noise, key):
    return next(r[key] for r in rows if r["exp"] == exp and r["method"] == method and r["k"] == k and r["noise"] == noise)


def style_ax(ax, title):
    ax.set_facecolor(BG); ax.grid(color=GRID, lw=0.6, axis="y"); ax.set_axisbelow(True)
    ax.set_title(title, color=FG, loc="left", fontsize=12)


# --- Fig N1: example reconstruction on an unseen floorplan ---
keys = ["true", "nearest", "pca_pooled", "nn_floorplan"]
titles = {"true": "정답 (열 해석기)", **NAMES}
maps = {k: np.array(ex[k]).reshape(N, N) for k in keys}
vmin, vmax = maps["true"].min(), maps["true"].max()
sy, sx = np.divmod(np.array(ex["sensors"]), N)
fig = plt.figure(figsize=(15.5, 4.6), facecolor=BG)
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.12)
for i, k in enumerate(keys):
    ax = fig.add_subplot(gs[0, i]); ax.set_facecolor(BG)
    im = ax.imshow(maps[k], origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.scatter(sx, sy, s=18, facecolors="none", edgecolors="#38e1ff", lw=1.0)
    err = "" if k == "true" else f"\nRMSE {np.sqrt(np.mean((maps[k] - maps['true']) ** 2)):.2f}°C"
    ax.set_title(titles[k] + err, color=FG, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 4])); cb.set_label("온도 (°C)", color=SUB, rotation=270, labelpad=14)
fig.suptitle("처음 보는 칩 배치에서 센서 16개(○)로 전체 온도 지도 복원 — 시뮬레이션", color=FG, x=0.02, ha="left", fontsize=13)
fig.savefig(R / "fig_n1_example_unseen.png", dpi=170, bbox_inches="tight", facecolor=BG); plt.close(fig)

# --- Fig N2: unseen floorplans, error by method and sensor noise ---
methods = ["nearest", "pca_pooled", "nn_sensor_only", "nn_floorplan"]
noises = [0.0, 0.5, 1.0]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
for ax, key, title in ((axes[0], "rmse", "칩 전체 온도 지도 오차 (RMSE, °C)"), (axes[1], "tmax_err", "최고 온도 오차 (°C)")):
    style_ax(ax, title)
    w = 0.2
    for j, m in enumerate(methods):
        vals = [get("unseen", m, 16, nz, key) for nz in noises]
        bars = ax.bar(np.arange(3) + (j - 1.5) * w, vals, w, color=COLORS[m], label=NAMES[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=SUB)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"센서 오차 ±{nz:g}°C" for nz in noises])
axes[0].legend(loc="upper center", bbox_to_anchor=(1.08, -0.1), ncol=4, frameon=False, fontsize=10)
fig.suptitle("처음 보는 칩 배치 50개 × 동작 상황 50개 (학습에 안 쓴 별도 출발점) — 시뮬레이션",
             color=FG, x=0.02, ha="left", fontsize=13)
fig.savefig(R / "fig_n2_unseen_bars.png", dpi=170, bbox_inches="tight", facecolor=BG); plt.close(fig)

# --- Fig N3: same floorplan (stage 1.5 setting): pattern reconstruction vs network ---
fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=BG)
style_ax(ax, "같은 칩 배치: 칩 전체 온도 지도 오차 (RMSE, °C)")
w = 0.35
for j, m in enumerate(["pca", "nn_floorplan"]):
    vals = [get("same", m, k, 0.0, "rmse") for k in (8, 16)]
    bars = ax.bar(np.arange(2) + (j - 0.5) * w, vals, w, color=COLORS[m], label=NAMES[m])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=SUB)
ax.set_xticks(range(2)); ax.set_xticklabels(["센서 8개", "센서 16개"]); ax.set_ylim(0, 1.45)
ax.legend(frameon=False, fontsize=10, loc="upper right")
fig.savefig(R / "fig_n3_same_floorplan.png", dpi=170, bbox_inches="tight", facecolor=BG); plt.close(fig)
print("saved fig_n1..n3")
