"""Development-only runtime equivalence and timing check, never a method result."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

from src.traffic_control.cli import run
from src.traffic_control.ppo import PPOModel
from src.traffic_control.scenario import ROOT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
model = PPOModel.load(ROOT / "models/v3/ippo-w3-r0.json")
records = []
for pedestrian_count in (120, 200):
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="runtime-probe-", dir=args.output.parent) as scratch:
        output = Path(scratch)
        run("policy", 6001, output, ROOT / "config/milestone.json", junctions=3,
            config_overrides={"vehicle_count_per_axis": 24, "pedestrian_count": pedestrian_count,
                              "vehicle_depart_last_second": 360, "pedestrian_depart_last_second": 360},
            policy_callback=lambda tls, obs, global_obs, eligible:
                model.select(tls, obs, global_obs, eligible, training=False),
            objective="pedestrian", pedestrian_reward_weight=3)
        records.append({"pedestrian_count": pedestrian_count,
            "seconds": time.perf_counter() - start,
            "sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                       for name in ("metrics.json", "crossing_events.json", "demand.rou.xml")}})
args.output.write_text(json.dumps({"python": sys.version, "executable": sys.executable,
                                   "records": records}, indent=2) + "\n", encoding="utf-8")
print(args.output)
