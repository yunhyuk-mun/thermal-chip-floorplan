"""Summarize results/stage1_floorplan/stage1_robust.jsonl: 10-seed statistics and the 20k vs 40k convergence check."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chipthermal.paths import STAGE1 as R
runs = [json.loads(l) for l in (R / "stage1_robust.jsonl").read_text().splitlines()]
old = {(r["w_temp"], r["seed"]): r for r in json.loads((R / "stage1_runs.json").read_text())["runs"]}
lines = ["| 온도 가중치 | 합법 | 배선 길이 평균 ± 표준편차 | 최고 온도 평균 ± 표준편차 (°C) | 최고 온도 중앙값 |", "|---|---|---|---|---|"]
stats = {}
for w in (0.0, 0.5, 1.0, 2.0, 4.0):
    rs = [r for r in runs if r["w_temp"] == w and r["iters"] == 20000]
    ok = [r for r in rs if r["legal"]]
    wl = np.array([r["hpwl"] for r in ok]); tm = np.array([r["tmax"] for r in ok])
    stats[w] = (wl, tm)
    lines.append(f"| {w} | {len(ok)}/{len(rs)} | {wl.mean():,.0f} ± {wl.std(ddof=1):,.0f} | "
                 f"{tm.mean():.2f} ± {tm.std(ddof=1):.2f} | {np.median(tm):.2f} |")

# reproducibility: seeds 0-2 at 20k must match the first experiment
repro = [abs(r["tmax"] - old[(r["w_temp"], r["seed"])]["tmax"]) < 1e-9
         for r in runs if r["iters"] == 20000 and (r["w_temp"], r["seed"]) in old]

# paired difference weight 0.5 vs 0 (same seed)
d = {}
for w in (0.5, 1.0):
    pairs = [(a["tmax"], b["tmax"], a["hpwl"], b["hpwl"]) for a in runs for b in runs
             if a["iters"] == b["iters"] == 20000 and a["seed"] == b["seed"] and a["w_temp"] == 0.0 and b["w_temp"] == w
             and a["legal"] and b["legal"]]
    dt = np.array([p[0] - p[1] for p in pairs]); dw = np.array([(p[3] - p[2]) / p[2] for p in pairs])
    d[w] = (len(pairs), dt.mean(), dt.std(ddof=1), int((dt > 0).sum()), dw.mean() * 100)

conv = ["| 온도 가중치 | 반복 | 시드 | 최고 온도 평균 (°C) | 배선 길이 평균 |", "|---|---|---|---|---|"]
for w in (0.0, 0.5):
    for it in (20000, 40000):
        rs = [r for r in runs if r["w_temp"] == w and r["iters"] == it and r["seed"] < 5 and r["legal"]]
        conv.append(f"| {w} | {it:,} | {len(rs)}개 | {np.mean([r['tmax'] for r in rs]):.2f} | {np.mean([r['hpwl'] for r in rs]):,.0f} |")

out = ["## 시드 10개 (반복 2만 회)", *lines, "",
       f"- 처음 실험(시드 0~2) 재현: {sum(repro)}/{len(repro)} 일치",
       *[f"- 같은 시드끼리 비교, 가중치 {w} vs 0: {n}쌍, 최고 온도 {m:.2f} ± {s:.2f}°C 낮음, {k}/{n}쌍에서 낮음, 배선 길이 평균 {pw:+.1f}%"
         for w, (n, m, s, k, pw) in d.items()], "",
       "## 수렴 확인 (시드 0~4)", *conv]
(R / "stage1_robust_summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))

BG, FG, SUB = "#0f1115", "#e8e6e1", "#9a978f"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.color": FG,
                     "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB, "axes.edgecolor": "#2a2d33"})
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG); ax.set_facecolor(BG); ax.grid(color="#22252b", lw=0.6)
cols = ["#b9b6ad", "#38e1ff", "#7f77dd", "#ff7a59", "#ffb347"]
for c, (w, (wl, tm)) in zip(cols, stats.items()):
    ax.scatter(wl, tm, color=c, s=28, alpha=0.6)
    ax.errorbar(wl.mean(), tm.mean(), xerr=wl.std(ddof=1), yerr=tm.std(ddof=1), color=c, marker="X", ms=11,
                mec="k", capsize=3, lw=1.5, label=f"온도 가중치 {w}")
ax.set_xlabel("배선 길이 (HPWL)"); ax.set_ylabel("칩 최고 온도 (°C)")
ax.set_title("시드 10개: 배선 길이 vs 최고 온도 (점 = 시드, X = 평균 ± 표준편차)", color=FG, loc="left")
ax.legend(frameon=False, labelcolor=FG, fontsize=9)
fig.savefig(R / "fig3_tradeoff_10seeds.png", dpi=140, facecolor=BG, bbox_inches="tight")
