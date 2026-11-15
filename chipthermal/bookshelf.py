"""Reader for GSRC Bookshelf floorplan files (.blocks / .nets / .pl), e.g. MCNC ami33."""
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Net:
    pins: list            # list of (name, dx_pct, dy_pct); offsets are % of block size from centre
    is_power: bool = False


@dataclass
class Design:
    blocks: dict = field(default_factory=dict)      # name -> (w, h)
    terminals: list = field(default_factory=list)   # terminal names
    terminal_pos: dict = field(default_factory=dict)
    nets: list = field(default_factory=list)


def read_design(prefix: Path) -> Design:
    d = Design()
    for line in Path(f"{prefix}.blocks").read_text().splitlines():
        m = re.match(r"(\S+)\s+hardrectilinear\s+4\s+(.*)", line)
        if m:
            pts = [tuple(map(float, p)) for p in re.findall(r"\(([-\d.]+),\s*([-\d.]+)\)", m.group(2))]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            d.blocks[m.group(1)] = (max(xs) - min(xs), max(ys) - min(ys))
        elif re.match(r"\S+\s+terminal\s*$", line):
            d.terminals.append(line.split()[0])

    cur, power = None, False
    for line in Path(f"{prefix}.nets").read_text().splitlines():
        s = line.strip()
        if s.startswith("NetDegree"):
            cur = Net(pins=[], is_power=power)
            d.nets.append(cur)
            power = False
        elif s.startswith("#"):
            power = power or 'type "PWR"' in s
        elif cur is not None and s and not s.startswith(("UCLA", "Num")):
            parts = s.split()
            name = parts[0]
            off = re.findall(r"%([-\d.]+)", s)
            dx, dy = (float(off[0]), float(off[1])) if len(off) == 2 else (0.0, 0.0)
            cur.pins.append((name, dx, dy))

    for line in Path(f"{prefix}.pl").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in d.terminals:
            d.terminal_pos[parts[0]] = (float(parts[1]), float(parts[2]))
    return d


if __name__ == "__main__":
    d = read_design(Path(__file__).resolve().parents[1] / "data" / "ami33" / "ami33")
    area = sum(w * h for w, h in d.blocks.values())
    degs = [len(n.pins) for n in d.nets]
    print(f"blocks={len(d.blocks)} terminals={len(d.terminals)} (with position: {len(d.terminal_pos)})")
    print(f"nets={len(d.nets)} pins={sum(degs)} power_nets={sum(n.is_power for n in d.nets)} "
          f"max_degree={max(degs)}")
    print(f"total block area={area:.0f} (side of equal square={area ** 0.5:.0f})")
    big = sorted(d.blocks.items(), key=lambda kv: -kv[1][0] * kv[1][1])[:3]
    print("largest blocks:", [(k, v) for k, v in big])
