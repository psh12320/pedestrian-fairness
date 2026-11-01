"""Replay one frozen stress run with read-only pedestrian position observations.

This post-test diagnostic does not tune models, alter simulation settings or
replace any final evaluation row. Console warnings remain in the caller's log.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".local/sumo-1.27.1/tools"))

import traci
from src.traffic_control.cli import run
from src.traffic_control.evaluate import run_rows, scenario_specs
from src.traffic_control.ppo import PPOModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=9006)
    args = parser.parse_args()
    seed, label, replica = args.seed, "ippo-w10", 0
    output = ROOT / f"runs/jam-diagnostic-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    model = PPOModel.load(ROOT / f"models/v3/{label}-r{replica}.json")
    settings = json.loads((ROOT / "config/evaluation.json").read_text())
    overrides, slow = scenario_specs(settings)["short-spacing"]
    observations = []
    original_step = traci.simulationStep

    def observed_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        second = traci.simulation.getTime()
        if 220 <= second <= 290 and "p110" in traci.person.getIDList():
            x, y = traci.person.getPosition("p110")
            observations.append({"second": second, "person": "p110",
                                 "edge": traci.person.getRoadID("p110"),
                                 "speed": traci.person.getSpeed("p110"), "x": x, "y": y,
                                 "C0_state": traci.trafficlight.getRedYellowGreenState("C0")})
        return result

    traci.simulationStep = observed_step
    try:
        metrics = run("policy", seed, output, ROOT / "config/milestone.json", junctions=3,
                      slow_pedestrians=slow, config_overrides=overrides,
                      policy_callback=lambda tls, obs, global_obs, eligible:
                      model.select(tls, obs, global_obs, eligible, training=False),
                      objective="pedestrian", pedestrian_reward_weight=10)
    finally:
        traci.simulationStep = original_step
    with (output / "person_trace.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["second", "person", "edge", "speed", "x", "y", "C0_state"])
        writer.writeheader()
        writer.writerows(observations)
    events = json.loads((output / "crossing_events.json").read_text())
    measured, _, _ = run_rows(metrics, events, "short-spacing", seed, label, replica, 1101)
    with (ROOT / "results/final-study/runs.csv").open() as stream:
        saved = next(r for r in csv.DictReader(stream) if r["scenario"] == "short-spacing"
                     and r["seed"] == str(seed) and r["controller"] == label
                     and r["training_run"] == str(replica))
    mismatches = {key: [saved[key], value] for key, value in measured.items()
                  if str(value) != saved[key]}
    if mismatches:
        raise AssertionError(mismatches)
    (output / "replay_check.json").write_text(json.dumps({
        "matches_every_saved_run_measure": True, "scenario": "short-spacing",
        "seed": seed, "controller": label, "replica": replica,
        "scope": "Read-only trace of p110; unchanged locked simulation and policy",
        "source": "https://sumo.dlr.de/docs/Simulation/Pedestrians.html#jamming",
    }, indent=2) + "\n")
    print("All saved demand and outcome measures reproduced exactly.")


if __name__ == "__main__":
    main()
