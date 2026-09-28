"""Replay matched baseline and learned traffic-light decisions in SUMO GUI.

The original evaluation uses TraCI to choose phases. This viewer converts the
recorded legal phase histories into SUMO static programs, then runs the same
network, routes, and seed in the graphical simulator without a TraCI client.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from .cli import run
from .connected import incoming_groups
from .ppo import PPOModel
from .scenario import ROOT, executable
from .signals import SignalPlan


CONTROLLERS = ("fixed", "actuated", "ippo-vehicle", "ippo-pedestrian",
               "mappo-vehicle", "mappo-pedestrian")


def source_run(controller: str, seed: int, slow_pedestrians: bool) -> Path:
    scenario = "vmedium-pmedium-slow" if slow_pedestrians else "vmedium-pmedium"
    existing = ROOT / "runs" / "evaluation" / scenario / f"seed-{seed}" / controller
    if (existing / "metrics.json").exists():
        return existing
    destination = ROOT / "runs" / "gui" / "source" / scenario / f"seed-{seed}" / controller
    if (destination / "metrics.json").exists():
        return destination
    overrides = {"vehicle_count_per_axis": 24, "pedestrian_count": 16,
                 "vehicle_depart_last_second": 360,
                 "pedestrian_depart_last_second": 360}
    policy = None
    kind = controller
    if controller not in ("fixed", "actuated"):
        model = PPOModel.load(ROOT / "models" / f"{controller}.json")
        kind = "policy"
        policy = lambda tls, obs, global_obs, eligible: model.select(
            tls, obs, global_obs, eligible, training=False)
    run(kind, seed, destination, ROOT / "config" / "milestone.json",
        junctions=3, slow_pedestrians=slow_pedestrians,
        config_overrides=overrides, policy_callback=policy)
    return destination


def write_replay(source: Path, output: Path) -> tuple[dict, Path]:
    """Make a static phase schedule from the measured TraCI signal history."""
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    network = ET.parse(source / "connected.net.xml").getroot()
    crossing_ids = set(metrics["crossing_lengths_metres"])
    end = int(metrics["config"]["end_seconds"])
    root = ET.Element("additional")
    for tls, history in sorted(metrics["signal_history"].items()):
        original = next(node for node in network.findall("tlLogic")
                        if node.attrib["id"] == tls)
        state_length = len(original.find("phase").attrib["state"])
        groups = {name: set() for name in ("EW", "NS", "PED")}
        for connection in network.findall("connection"):
            if connection.get("tl") != tls:
                continue
            index = int(connection.attrib["linkIndex"])
            source_edge = connection.attrib["from"]
            target_edge = connection.attrib["to"]
            if source_edge in crossing_ids or target_edge in crossing_ids:
                group = "PED"
            elif source_edge in incoming_groups(3)[tls]["EW"]:
                group = "EW"
            elif source_edge in incoming_groups(3)[tls]["NS"]:
                group = "NS"
            else:
                raise ValueError(f"Unclassified {tls} connection {connection.attrib}")
            groups[group].add(index)
        if set.union(*groups.values()) != set(range(state_length)):
            raise ValueError(f"Incomplete controlled links at {tls}: {groups}")
        plan = SignalPlan(groups, state_length,
                          metrics["pedestrian_clearance_seconds"], metrics["config"])
        logic = ET.SubElement(root, "tlLogic", id=tls, type="static",
                              programID="recorded-evaluation", offset="0")
        for index, (begin, stage) in enumerate(history):
            if int(begin) >= end:
                break
            finish = history[index + 1][0] if index + 1 < len(history) else end
            duration = int(finish) - int(begin)
            if duration <= 0:
                raise ValueError(f"Invalid {tls} phase duration: {begin} to {finish}")
            ET.SubElement(logic, "phase", duration=str(duration),
                          state=plan.state(stage))
        # SUMO repeats a static program after its final phase. Complete the
        # post-horizon transition legally so the replay does not jump from a
        # vehicle green straight to its first east-west green on wraparound.
        last_begin, last_stage = history[-1]
        elapsed = end - int(last_begin)

        def append(stage: str, duration: int) -> None:
            if duration > 0:
                ET.SubElement(logic, "phase", duration=str(duration),
                              state=plan.state(stage))

        if last_stage == "NS":
            append("NS", max(0, metrics["config"]["vehicle_green_min_seconds"] - elapsed))
            append("NS_YELLOW", metrics["config"]["yellow_seconds"])
            append("ALL_RED", metrics["config"]["all_red_seconds"])
            append("EW", metrics["config"]["vehicle_green_min_seconds"])
        elif last_stage == "NS_YELLOW":
            append("NS_YELLOW", max(0, metrics["config"]["yellow_seconds"] - elapsed))
            append("ALL_RED", metrics["config"]["all_red_seconds"])
            append("EW", metrics["config"]["vehicle_green_min_seconds"])
        elif last_stage == "EW_YELLOW":
            append("EW_YELLOW", max(0, metrics["config"]["yellow_seconds"] - elapsed))
            append("ALL_RED", metrics["config"]["all_red_seconds"])
            append("NS", metrics["config"]["vehicle_green_min_seconds"])
            append("NS_YELLOW", metrics["config"]["yellow_seconds"])
            append("ALL_RED", metrics["config"]["all_red_seconds"])
            append("EW", metrics["config"]["vehicle_green_min_seconds"])
        elif last_stage == "ALL_RED":
            append("ALL_RED", max(0, metrics["config"]["all_red_seconds"] - elapsed))
            append("EW", metrics["config"]["vehicle_green_min_seconds"])
        elif last_stage == "PED":
            append("PED", max(0, metrics["config"]["pedestrian_green_seconds"] - elapsed))
            append("PED_CLEARANCE", plan.clearance)
            append("EW", metrics["config"]["vehicle_green_min_seconds"])
        elif last_stage == "PED_CLEARANCE":
            append("PED_CLEARANCE", max(0, plan.clearance - elapsed))
            append("EW", metrics["config"]["vehicle_green_min_seconds"])
        elif last_stage != "EW":
            raise ValueError(f"Unexpected final phase: {last_stage}")
    destination = output / "replay.add.xml"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return metrics, destination


def launch(controller: str, seed: int, slow_pedestrians: bool, delay_ms: float,
           window_pos: tuple[int, int] | None) -> tuple[subprocess.Popen, dict]:
    source = source_run(controller, seed, slow_pedestrians)
    output = ROOT / "runs" / "gui" / f"seed-{seed}" / controller
    metrics, replay = write_replay(source, output)
    command = [executable("sumo-gui"), "--net-file", str(source / "connected.net.xml"),
               "--route-files", str(source / "demand.rou.xml"),
               "--additional-files", str(replay), "--begin", "0",
               "--end", str(metrics["config"]["end_seconds"]),
               "--step-length", "1", "--seed", str(seed),
               "--pedestrian.striping.dawdling", "0", "--time-to-teleport", "-1",
               "--start", "--delay", str(delay_ms),
               "--window-size", "900,700", "--log", str(output / "sumo-gui.log")]
    if window_pos is not None:
        command.extend(["--window-pos", f"{window_pos[0]},{window_pos[1]}"])
    process = subprocess.Popen(command, cwd=ROOT)
    print(f"Opened {controller} (PID {process.pid}); phase replay: {replay}", flush=True)
    return process, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compare", action="store_true",
                      help="Open fixed-time and a trained controller side by side")
    mode.add_argument("--controller", choices=CONTROLLERS)
    parser.add_argument("--rl-controller", choices=CONTROLLERS[2:],
                        default="ippo-pedestrian")
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--delay-ms", type=float, default=200)
    parser.add_argument("--slow-pedestrians", action="store_true")
    args = parser.parse_args()
    if args.delay_ms < 0:
        parser.error("--delay-ms must be nonnegative")
    if args.compare:
        _, fixed = launch("fixed", args.seed, args.slow_pedestrians,
                          args.delay_ms, (0, 0))
        _, learned = launch(args.rl_controller, args.seed, args.slow_pedestrians,
                            args.delay_ms, (920, 0))
        if fixed["demand_sha256"] != learned["demand_sha256"]:
            raise AssertionError("The two windows received different demand")
        print("Matched demand verified. The windows replay the recorded "
              "signal phases and remain open after the run. Zoom into a "
              "junction to see pedestrians clearly.", flush=True)
    else:
        launch(args.controller, args.seed, args.slow_pedestrians, args.delay_ms, None)


if __name__ == "__main__":
    main()
