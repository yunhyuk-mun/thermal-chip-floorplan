"""Figures for stage 1.5 (sensor placement). Reads results/stage15_sensors/sensors_results.json, recomputes condition numbers."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle
import numpy as np

from chipthermal.sensors import setup, pca_basis, place_grid, place_hot, place_data, K_LIST, N_TRAIN

from chipthermal.paths import SENSORS as R
d = json.loads((R / "sensors_results.json").read_text())
rows, ex, meta = d["rows"], d["examples"], d["meta"]
BG, FG, SUB, GRID = "#0f1115", "#e8e6e1", "#9a978f", "#22252b"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.color": FG,
                     "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB, "axes.edgecolor": "#2a2d33"})
STYLE = {("grid", "nearest"): ("#b9b6ad", "-", "격자 배치 + 가까운 값"),
         ("hot", "nearest"): ("#ff7a59", "-", "뜨거운 블록 위 + 가까운 값"),
         ("data", "pca"): ("#38e1ff", "-", "데이터 기반 배치 + 패턴 복원"),
         ("grid", "pca"): ("#b9b6ad", ":", "격자 배치 + 패턴 복원"),
         ("hot", "pca"): ("#ff7a59", ":", "뜨거운 블록 위 + 패턴 복원"),
         ("data", "nearest"): ("#38e1ff", ":", "데이터 기반 배치 + 가까운 값")}


def sel(p, r, noise, key):
    return [next(x[key] for x in rows if x["k"] == k and x["placement"] == p and x["recon"] == r and x["noise"] == noise)
            for k in K_LIST]


def style_ax(ax, title):
    ax.set_facecolor(BG); ax.grid(color=GRID, lw=0.6); ax.set_title(title, color=FG, loc="left", fontsize=12)


# --- Fig S1: error vs number of sensors ---
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), facecolor=BG)
for ax, key, title in ((axes[0], "rmse", "칩 전체 온도 지도 오차 (RMSE, °C)"), (axes[1], "tmax_err", "최고 온도 오차 (°C)")):
    style_ax(ax, title)
    for (p, r), (c, ls, lab) in STYLE.items():
        ax.plot(K_LIST, sel(p, r, 0.0, key), color=c, ls=ls, lw=2.4 if ls == "-" else 1.4, marker="o", ms=5, label=lab)
    ax.set_yscale("log"); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.set_xscale("log", base=2); ax.set_xticks(K_LIST); ax.set_xticklabels(K_LIST)
    ax.set_xlabel("센서 개수")
fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=10, frameon=False, labelcolor=FG, bbox_to_anchor=(0.5, -0.12))
fig.suptitle("센서 몇 개로 칩 온도를 얼마나 알 수 있나 (시험 상황 100개 평균, 센서 오차 없음, 가정 기반 시뮬레이션)",
             color=FG, fontsize=13, x=0.01, ha="left")
fig.savefig(R / "fig_s1_error_vs_k.png", dpi=140, facecolor=BG, bbox_inches="tight"); plt.close(fig)

# --- Fig S2: example maps, 8 sensors ---
prob, xywh, T = setup()
n = prob.solver.p.nx; S = prob.outline
true = np.array(ex["true"]).reshape(n, n)
panels = [("true", None, "실제 온도 지도"),
          ("grid_nearest", ("grid", "nearest"), "격자 배치 + 가까운 값"),
          ("hot_nearest", ("hot", "nearest"), "뜨거운 블록 위 + 가까운 값"),
          ("data_pca", ("data", "pca"), "데이터 기반 배치 + 패턴 복원")]
vmin, vmax = true.min() - 1, true.max() + 1
fig, axes = plt.subplots(1, 4, figsize=(16, 4.9), facecolor=BG)
x, y, w, h = xywh
for ax, (key, combo, title) in zip(axes, panels):
    m = true if key == "true" else np.array(ex[key]["est"]).reshape(n, n)
    im = ax.imshow(m, origin="lower", extent=[0, S, 0, S], cmap="inferno", vmin=vmin, vmax=vmax, interpolation="bicubic")
    for i in range(len(x)):
        hot = i in prob.hot
        ax.add_patch(Rectangle((x[i], y[i]), w[i], h[i], fill=False, ec="#38e1ff" if hot else "white",
                               lw=1.8 if hot else 0.4, alpha=1 if hot else 0.4))
    sub = f"최고 {m.max():.1f}°C"
    if combo:
        sy, sx = np.divmod(np.array(ex[key]["sensors"]), n)
        ax.scatter((sx + 0.5) * S / n, (sy + 0.5) * S / n, s=70, c="#7CFC00", ec="k", lw=1.2, zorder=5)
        rm = float(np.sqrt(np.mean((m - true) ** 2)))
        sub += f" · 이 상황 RMSE {rm:.2f}°C"
    ax.set_title(f"{title}\n{sub}", color=FG, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01); cb.set_label("온도 (°C)", color=SUB); cb.outline.set_visible(False)
fig.suptitle("센서 8개(연두색 점)로 복원한 온도 지도 — 시험 상황 1개 예시", color=FG, fontsize=13, x=0.01, ha="left")
fig.savefig(R / "fig_s2_maps_k8.png", dpi=140, facecolor=BG, bbox_inches="tight"); plt.close(fig)

# --- Fig S3: why bad placement + pattern reconstruction explodes (conditioning) ---
train = T[:N_TRAIN]
conds = {"grid": [], "hot": [], "data": []}
for k in K_LIST:
    mu, phi = pca_basis(train, k)
    for p, sens in (("grid", place_grid(k, n)), ("hot", place_hot(k, prob, xywh, n)), ("data", place_data(k, train))):
        conds[p].append(float(np.linalg.cond(phi[sens, :])))
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), facecolor=BG)
style_ax(axes[0], "센서 위치의 '조건수' (클수록 오차를 크게 키움)")
for p, c, lab in (("grid", "#b9b6ad", "격자 배치"), ("hot", "#ff7a59", "뜨거운 블록 위"), ("data", "#38e1ff", "데이터 기반 배치")):
    axes[0].plot(K_LIST, conds[p], color=c, lw=2.4, marker="o", label=lab)
axes[0].set_yscale("log"); axes[0].set_xscale("log", base=2); axes[0].set_xticks(K_LIST); axes[0].set_xticklabels(K_LIST)
axes[0].set_xlabel("센서 개수"); axes[0].legend(frameon=False, labelcolor=FG, fontsize=9)
style_ax(axes[1], "센서 오차에 따른 지도 오차 (센서 16개, 패턴 복원)")
xs = np.arange(len(meta["noise"]))
for j, (p, c, lab) in enumerate((("grid", "#b9b6ad", "격자 배치"), ("hot", "#ff7a59", "뜨거운 블록 위"), ("data", "#38e1ff", "데이터 기반 배치"))):
    vals = [next(r["rmse"] for r in rows if r["k"] == 16 and r["placement"] == p and r["recon"] == "pca" and r["noise"] == nz)
            for nz in meta["noise"]]
    bars = axes[1].bar(xs + (j - 1) * 0.27, vals, width=0.25, color=c, label=lab)
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2g}", ha="center", color=FG, fontsize=8)
axes[1].set_yscale("log"); axes[1].set_xticks(xs); axes[1].set_xticklabels([f"±{v}°C" if v else "오차 없음" for v in meta["noise"]])
axes[1].set_ylabel("RMSE (°C)"); axes[1].set_ylim(top=2000); axes[1].legend(frameon=False, labelcolor=FG, fontsize=9, ncol=3, loc="upper left")
fig.savefig(R / "fig_s3_conditioning.png", dpi=140, facecolor=BG, bbox_inches="tight"); plt.close(fig)
(R / "sensors_condition_numbers.json").write_text(json.dumps(dict(k=K_LIST, cond=conds), indent=1))
print("cond", {p: [f"{c:.3g}" for c in v] for p, v in conds.items()})
