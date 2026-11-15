"""Figures for stage 1 from results/stage1_floorplan/stage1_runs.json and results/stage1_floorplan/stage1_layouts.npz."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
from chipthermal.paths import STAGE1 as R
data = json.loads((R / "stage1_runs.json").read_text())
meta, runs = data["meta"], data["runs"]
lay = np.load(R / "stage1_layouts.npz")
legal = [r for r in runs if r["legal"]]


def best_of(wt):
    c = [r for r in legal if r["w_temp"] == wt]
    return min(c, key=lambda r: r["cost"]) if c else None


# --- Figure 1: wirelength-only vs thermal-aware (best legal run of each) ---
base = best_of(0.0)
thermal_ws = [w for w in meta["weights"] if w > 0 and best_of(w)]
pick_w = max(thermal_ws, key=lambda w: base["dtmax"] - best_of(w)["dtmax"]) if base and thermal_ws else None
if base and pick_w is not None:
    th = best_of(pick_w)
    vmin = meta and min(lay[f"w0.0_s{base['seed']}_temp"].min(), lay[f"w{pick_w}_s{th['seed']}_temp"].min())
    vmax = max(lay[f"w0.0_s{base['seed']}_temp"].max(), lay[f"w{pick_w}_s{th['seed']}_temp"].max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    S = meta["outline"]
    for ax, r, title in ((axes[0], base, "배선만 고려"), (axes[1], th, f"배선 + 온도 고려 (가중치 {pick_w})")):
        key = f"w{r['w_temp']}_s{r['seed']}"
        im = ax.imshow(lay[key + "_temp"], origin="lower", extent=[0, S, 0, S], cmap="inferno", vmin=vmin, vmax=vmax)
        x, y, w, h = lay[key + "_xywh"]
        for i in range(len(x)):
            hot = meta["names"][i] in meta["hot"]
            ax.add_patch(Rectangle((x[i], y[i]), w[i], h[i], fill=False,
                                   ec="cyan" if hot else "white", lw=2.0 if hot else 0.6))
        ax.set_xlim(0, S); ax.set_ylim(0, S); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\n최고 {r['tmax']:.1f}°C · 배선 길이 {r['hpwl']:,.0f}")
    fig.colorbar(im, ax=axes, shrink=0.85, label="온도 (°C)")
    fig.suptitle("ami33 배치와 온도 지도 (하늘색 테두리 = 뜨거운 블록 4개, 가정값 기반)")
    fig.savefig(R / "fig1_compare.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

# --- Figure 2: trade-off ---
fig, ax = plt.subplots(figsize=(6.4, 4.6))
cmap = plt.get_cmap("viridis")
for k, wt in enumerate(meta["weights"]):
    rs = [r for r in legal if r["w_temp"] == wt]
    if not rs:
        continue
    col = cmap(k / max(1, len(meta["weights"]) - 1))
    ax.scatter([r["hpwl"] for r in rs], [r["tmax"] for r in rs], color=col, s=40, label=f"온도 가중치 {wt}")
    ax.scatter(np.mean([r["hpwl"] for r in rs]), np.mean([r["tmax"] for r in rs]), color=col, marker="X", s=140, ec="k")
ax.set_xlabel("배선 길이 (HPWL, 벤치마크 단위)")
ax.set_ylabel("칩 최고 온도 (°C)")
ax.set_title("배선 길이 vs 최고 온도 (점 = 시드별, X = 평균, 합법 배치만)")
ax.legend(fontsize=8)
fig.savefig(R / "fig2_tradeoff.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# --- Summary table ---
lines = ["| 온도 가중치 | 합법 배치 | 배선 길이 평균 | 최고 온도 평균 (°C) | 온도 상승 평균 (K) |", "|---|---|---|---|---|"]
for wt in meta["weights"]:
    rs = [r for r in runs if r["w_temp"] == wt]
    ok = [r for r in rs if r["legal"]]
    if ok:
        lines.append(f"| {wt} | {len(ok)}/{len(rs)} | {np.mean([r['hpwl'] for r in ok]):,.0f} | "
                     f"{np.mean([r['tmax'] for r in ok]):.2f} | {np.mean([r['dtmax'] for r in ok]):.2f} |")
    else:
        lines.append(f"| {wt} | 0/{len(rs)} | - | - | - |")
(R / "stage1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print("figure1 pair:", "w0.0 vs", pick_w)
