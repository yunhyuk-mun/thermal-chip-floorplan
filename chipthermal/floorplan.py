"""Sequence-pair simulated-annealing floorplanner with optional thermal cost (stage 1).

Choices follow criteria.md: power nets excluded, pads ignored, 15% whitespace,
total 50 W with 4 hot blocks at 5x power density (fixed seed).
"""
import math
import random
from dataclasses import dataclass
import numpy as np

from .bookshelf import read_design
from .paths import AMI33
from .thermal import ChipParams, ThermalSolver, overlap_fraction


@dataclass
class Problem:
    names: list
    w: np.ndarray            # block widths (benchmark units)
    h: np.ndarray
    nets: list               # list of (block_idx array, dx_pct array, dy_pct array)
    outline: float           # square chip side (benchmark units)
    power: np.ndarray        # W per block
    hot: list                # indices of hot blocks
    solver: ThermalSolver


def load_problem(total_power=50.0, n_hot=4, hot_factor=5.0, whitespace=0.15, seed=0) -> Problem:
    d = read_design(AMI33)
    names = list(d.blocks)
    index = {n: i for i, n in enumerate(names)}
    w = np.array([d.blocks[n][0] for n in names])
    h = np.array([d.blocks[n][1] for n in names])
    nets = []
    for net in d.nets:
        if net.is_power:
            continue
        pins = [(index[p], dx, dy) for p, dx, dy in net.pins if p in index]  # drop pads
        if len({p[0] for p in pins}) >= 2:
            a = np.array(pins)
            nets.append((a[:, 0].astype(int), a[:, 1] / 100.0, a[:, 2] / 100.0))
    area = w * h
    outline = math.sqrt(area.sum() * (1 + whitespace))
    rng = random.Random(seed)
    hot = sorted(rng.sample(range(len(names)), n_hot))
    dens = np.ones(len(names))
    dens[hot] = hot_factor
    power = total_power * dens * area / np.sum(dens * area)
    return Problem(names, w, h, nets, outline, power, hot, ThermalSolver(ChipParams()))


def pack(pos, neg, w, h):
    """Sequence pair -> lower-left coordinates. O(n^2), fine for 33 blocks."""
    n = len(pos)
    rank_neg = np.empty(n, int)
    rank_neg[neg] = np.arange(n)
    x = np.zeros(n)
    y = np.zeros(n)
    # x: block b is left of a if b before a in both sequences
    for i, a in enumerate(pos):
        xa = 0.0
        for b in pos[:i]:
            if rank_neg[b] < rank_neg[a]:
                xa = max(xa, x[b] + w[b])
        x[a] = xa
    # y: block b is below a if b after a in pos and before a in neg
    for i in range(n - 1, -1, -1):
        a = pos[i]
        ya = 0.0
        for b in pos[i + 1:]:
            if rank_neg[b] < rank_neg[a]:
                ya = max(ya, y[b] + h[b])
        y[a] = ya
    return x, y


def hpwl(prob, x, y, w, h, rot=None):
    """Half-perimeter wirelength. Pin offsets are fractions of block size from the centre.

    A rotated block is turned 90 degrees counter-clockwise, so its pin offsets turn with it:
    (dx, dy) -> (-dy, dx), still as fractions of the (swapped) width and height.
    """
    total = 0.0
    for idx, dx, dy in prob.nets:
        if rot is not None:
            r = rot[idx]
            dx, dy = np.where(r, -dy, dx), np.where(r, dx, dy)
        px = x[idx] + w[idx] * (0.5 + dx)
        py = y[idx] + h[idx] * (0.5 + dy)
        total += (px.max() - px.min()) + (py.max() - py.min())
    return total


def block_fractions(prob, x, y, w, h):
    """Per-block overlap-fraction maps on the thermal grid, shape (n_blocks, ny, nx)."""
    p = prob.solver.p
    return np.array([overlap_fraction(x[i], y[i], w[i], h[i], prob.outline, prob.outline, p.nx, p.ny)
                     for i in range(len(w))])


def power_map(prob, x, y, w, h, power=None):
    """Rasterize block power onto the thermal grid (W/m^2). Parts outside the outline are clipped."""
    p = prob.solver.p
    power = prob.power if power is None else power
    q = np.zeros((p.ny, p.nx))
    for i in range(len(w)):
        q += overlap_fraction(x[i], y[i], w[i], h[i], prob.outline, prob.outline, p.nx, p.ny) * power[i]
    cell_area_m2 = (p.width / p.nx) * (p.height / p.ny)
    return q / cell_area_m2


def evaluate(prob, pos, neg, rot):
    w = np.where(rot, prob.h, prob.w)
    h = np.where(rot, prob.w, prob.h)
    x, y = pack(pos, neg, w, h)
    W, H = (x + w).max(), (y + h).max()
    overflow = (max(0.0, W - prob.outline) + max(0.0, H - prob.outline)) / prob.outline
    wl = hpwl(prob, x, y, w, h, rot)
    temp = prob.solver.solve(power_map(prob, x, y, w, h))
    return dict(x=x, y=y, w=w, h=h, W=W, H=H, overflow=overflow, hpwl=wl,
                tmax=float(temp.max()), temp=temp)


def normalizers(prob, n=200, seed=1):
    rng = random.Random(seed)
    n_blk = len(prob.names)
    wl, dt = [], []
    for _ in range(n):
        pos = list(range(n_blk)); rng.shuffle(pos)
        neg = list(range(n_blk)); rng.shuffle(neg)
        e = evaluate(prob, pos, neg, np.zeros(n_blk, bool))
        wl.append(e["hpwl"]); dt.append(e["tmax"] - prob.solver.p.t_amb)
    return float(np.mean(wl)), float(np.mean(dt))


def anneal(prob, w_temp, norm, iters=20000, seed=0, penalty=10.0, t0=1.0, t_end=1e-3,
           snapshot_every=0, snapshots=None):
    """Cost = hpwl/WL0 + w_temp * dTmax/DT0 + penalty * overflow.

    If snapshot_every > 0, the current (not best) state is appended to `snapshots`
    every that many iterations; this does not change the random sequence.
    """
    rng = random.Random(seed)
    n = len(prob.names)
    wl0, dt0 = norm
    t_amb = prob.solver.p.t_amb
    pos = list(range(n)); rng.shuffle(pos)
    neg = list(range(n)); rng.shuffle(neg)
    rot = np.zeros(n, bool)

    def cost(e):
        return e["hpwl"] / wl0 + w_temp * (e["tmax"] - t_amb) / dt0 + penalty * e["overflow"]

    cur = evaluate(prob, pos, neg, rot); cur_c = cost(cur)
    best = (cur_c, pos[:], neg[:], rot.copy(), cur)
    alpha = (t_end / t0) ** (1.0 / iters)
    T = t0
    for it in range(iters):
        if snapshot_every and it % snapshot_every == 0 and snapshots is not None:
            snapshots.append(dict(it=it, x=cur["x"], y=cur["y"], w=cur["w"], h=cur["h"],
                                  temp=cur["temp"], tmax=cur["tmax"], hpwl=cur["hpwl"],
                                  overflow=cur["overflow"]))
        p2, n2, r2 = pos[:], neg[:], rot.copy()
        mv = rng.random()
        i, j = rng.sample(range(n), 2)
        if mv < 0.3:
            p2[i], p2[j] = p2[j], p2[i]
        elif mv < 0.6:
            n2[i], n2[j] = n2[j], n2[i]
        elif mv < 0.85:
            a, b = p2[i], p2[j]
            p2[i], p2[j] = b, a
            ia, ib = n2.index(a), n2.index(b)
            n2[ia], n2[ib] = b, a
        else:
            r2[i] = ~r2[i]
        e = evaluate(prob, p2, n2, r2); c = cost(e)
        if c < cur_c or rng.random() < math.exp(-(c - cur_c) / T):
            pos, neg, rot, cur, cur_c = p2, n2, r2, e, c
            if c < best[0]:
                best = (c, pos[:], neg[:], rot.copy(), e)
        T *= alpha
    return best
