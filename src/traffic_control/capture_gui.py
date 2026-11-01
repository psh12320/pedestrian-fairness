"""Capture original, matched-demand SUMO GUI stills for a presentation.

The screenshot is a visualization of a recorded legal phase replay, not a
measurement. The source run's JSON/TripInfo files remain the quantitative data.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from .scenario import ROOT, executable
from .viewer import default_model_directory, source_run, write_replay


def capture(controllers: list[str], seed: int, pedestrian_count: int,
            model_dir: Path, output: Path, second: int,
            boundary: tuple[float, float, float, float] | None) -> list[Path]:
    import traci

    if second < 0 or pedestrian_count < 0:
        raise ValueError("Simulation second and pedestrian count must be nonnegative")
    if len(set(controllers)) != len(controllers):
        raise ValueError("Controller names must be distinct")
    if boundary is not None and (boundary[0] >= boundary[2] or boundary[1] >= boundary[3]):
        raise ValueError("Invalid GUI boundary")
    output.mkdir(parents=True, exist_ok=True)
    captured = []
    demand_hash = None
    for controller in controllers:
        source = source_run(controller, seed, False, model_dir, pedestrian_count)
        if boundary is None:
            centre = next(node for node in ET.parse(source / "connected.net.xml")
                          .iter("junction") if node.get("id") == "C1")
            x, y = float(centre.attrib["x"]), float(centre.attrib["y"])
            view_boundary = (x - 85, y - 80, x + 85, y + 80)
        else:
            view_boundary = boundary
        replay_dir = ROOT / "runs" / "gui" / "capture-replay" / f"seed-{seed}" / controller
        metrics, replay = write_replay(source, replay_dir)
        if demand_hash is None:
            demand_hash = metrics["demand_sha256"]
        elif metrics["demand_sha256"] != demand_hash:
            raise AssertionError("Screenshot controllers do not share demand")
        destination = output / f"sumo-{controller}-seed-{seed}-t{second}.png"
        command = [executable("sumo-gui"), "--net-file", str(source / "connected.net.xml"),
                   "--route-files", str(source / "demand.rou.xml"),
                   "--additional-files", str(replay), "--begin", "0",
                   "--gui-settings-file", str(ROOT / "config" / "demo.gui.xml"),
                   "--end", str(metrics["config"]["end_seconds"]),
                   "--step-length", "1", "--seed", str(seed),
                   "--pedestrian.striping.dawdling", "0", "--time-to-teleport", "-1",
                   "--start", "--no-step-log", "true", "--window-size", "1200,800"]
        traci.start(command)
        try:
            views = traci.gui.getIDList()
            if not views:
                raise RuntimeError("SUMO GUI exposed no view")
            view = views[0]
            traci.gui.setBoundary(view, *view_boundary)
            for _ in range(second):
                traci.simulationStep()
            traci.gui.screenshot(view, str(destination.resolve()), 1200, 800)
            traci.simulationStep()  # SUMO writes the requested image on the next step.
        finally:
            # A GUI may remain visible after its TraCI session closes; do not
            # wait for the user to close that window before capturing the next.
            traci.close(wait=False)
        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError(f"SUMO GUI did not write {destination}")
        captured.append(destination)
    return captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controllers", nargs="+", default=["fixed", "ippo-pedestrian"])
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--pedestrian-count", type=int, default=120)
    parser.add_argument("--model-dir", type=Path, default=default_model_directory())
    parser.add_argument("--output", type=Path,
                        default=ROOT / "deliverables" / "presentation" / "assets" / "final-protocol")
    parser.add_argument("--second", type=int, default=200)
    parser.add_argument("--boundary", type=float, nargs=4,
                        help="x-min y-min x-max y-max; default centres on C1")
    args = parser.parse_args()
    for path in capture(args.controllers, args.seed, args.pedestrian_count,
                        args.model_dir, args.output, args.second,
                        tuple(args.boundary) if args.boundary else None):
        print(path)


if __name__ == "__main__":
    main()
