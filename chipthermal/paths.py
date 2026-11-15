"""Project folders. Every script reads and writes through these paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AMI33 = DATA / "ami33" / "ami33"            # prefix for .blocks / .nets / .pl
RESULTS = ROOT / "results"
STAGE1 = RESULTS / "stage1_floorplan"
ANIM = RESULTS / "stage1_animation"
SENSORS = RESULTS / "stage15_sensors"
OPTICAL = RESULTS / "optical_stage"
STAGE2 = RESULTS / "stage2_nn"
VALIDATION = RESULTS / "validation"
LOGS = ROOT / "logs"

for _d in (STAGE1, ANIM, SENSORS, OPTICAL, STAGE2, VALIDATION, LOGS):
    _d.mkdir(parents=True, exist_ok=True)
