"""Train four controllers: IPPO/MAPPO crossed with two reward objectives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli import run
from .ppo import PPOModel
from .scenario import ROOT


def train(episodes: int | None = None, output: Path | None = None) -> list[dict]:
    settings = json.loads((ROOT / "config" / "training.json").read_text(encoding="utf-8"))
    episodes = episodes or settings["episodes"]
    output = output or ROOT / "runs" / "training"
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    results = []
    for method in ("ippo", "mappo"):
        for objective in ("vehicle", "pedestrian"):
            label = f"{method}-{objective}"
            model = PPOModel(method, objective, ["C0", "C1", "C2"], settings,
                             seed=settings["seed_start"])
            episodes_log = []
            for episode in range(episodes):
                factor = (0.75, 1.0, 1.25)[episode % 3]
                overrides = {"vehicle_count_per_axis": round(24 * factor),
                             "pedestrian_count": round(16 * factor),
                             "vehicle_depart_last_second": 360,
                             "pedestrian_depart_last_second": 360}
                seed = settings["seed_start"] + episode
                trace = {}
                metrics = run("policy", seed, output / label / f"episode-{episode:03d}",
                              ROOT / "config" / "milestone.json", junctions=3,
                              config_overrides=overrides,
                              policy_callback=lambda tls, obs, global_obs, eligible:
                                  model.select(tls, obs, global_obs, eligible, training=True),
                              trace_sink=trace, objective=objective)
                update = model.update(trace)
                episodes_log.append({"episode": episode, "seed": seed,
                                     "demand_factor": factor, "update": update,
                                     "completed_vehicles": metrics["vehicles"]["completed_count"],
                                     "total_pedestrian_entries": sum(
                                         item["entered_count"] for item in
                                         metrics["pedestrians_by_crossing"].values())})
                print(f"{label}: episode {episode + 1}/{episodes}", flush=True)
            model_path = model_dir / f"{label}.json"
            model.save(model_path)
            report = {"label": label, "method": method, "objective": objective,
                      "model_path": str(model_path), "episodes": episodes_log,
                      "settings": settings}
            (output / label / "training.json").write_text(json.dumps(report, indent=2) + "\n",
                                                          encoding="utf-8")
            results.append(report)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    train(args.episodes, args.output)


if __name__ == "__main__":
    main()
