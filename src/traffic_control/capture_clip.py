"""Record matched-seed SUMO GUI clips from legal recorded signal phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import struct
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .scenario import ROOT, executable
from .viewer import default_model_directory, source_run, write_replay


def frame_directory(source: Path, start_second: int, frames: int) -> Path:
    """Keep frames from different safety protocols and models separate."""
    source_root = ROOT / "runs" / "gui" / "source"
    source_key = source.relative_to(source_root)
    return (ROOT / "runs" / "gui" / "clip-frames" / source_key /
            f"t{start_second}-n{frames}")


def prepare_frame_directory(frame_dir: Path) -> None:
    """Do not count frames left by an earlier interrupted capture as new ones."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    for previous in frame_dir.glob("frame-*.png"):
        previous.unlink()


def capture_isolated_frames(source: Path, replay: Path, frame_dir: Path,
                            seed: int, start_second: int, frames: int,
                            centre: tuple[float, float]) -> None:
    """Use one owned GUI process per frame when repeated OpenGL capture stalls.

    Every frame replays the identical seed and recorded phases up to its time.
    Only read-only GUI calls differ. No interpolation or generated motion is used.
    """
    import traci

    x, y = centre
    for index in range(frames):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        destination = frame_dir / f"frame-{index:03d}.png"
        command = [executable("sumo-gui"), "--net-file", str(source / "connected.net.xml"),
                   "--route-files", str(source / "demand.rou.xml"),
                   "--additional-files", str(replay), "--begin", "0", "--end", "600",
                   "--step-length", "1", "--seed", str(seed),
                   "--pedestrian.striping.dawdling", "0", "--time-to-teleport", "-1",
                   "--gui-settings-file", str(ROOT / "config/demo.gui.xml"),
                   "--remote-port", str(port), "--start", "--delay", "0",
                   "--no-step-log", "true", "--window-size", "1200,800"]
        connection = None
        with (frame_dir / "capture.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                connection = traci.connect(port=port, numRetries=60, proc=process,
                                           waitBetweenRetries=0.1)
                # TraCI uses blocking socket reads. Bound graphics-driver stalls
                # in this visualization-only helper (SUMO 1.27.1 connection API).
                connection._socket.settimeout(15)
                view = connection.gui.getIDList()[0]
                connection.gui.setBoundary(view, x - 60, y - 40, x + 60, y + 40)
                print(f"Frame {index}: advance to {start_second + index}", flush=True)
                # Single-step as in the validated still-capture path. Jumping
                # straight to a late time can race initial OpenGL painting.
                for _ in range(start_second + index):
                    connection.simulationStep()
                time.sleep(0.2)
                print(f"Frame {index}: screenshot", flush=True)
                connection.gui.screenshot(view, str(destination.resolve()), 1200, 800)
                connection.simulationStep()
                deadline = time.monotonic() + 5
                while not destination.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                if not destination.exists():
                    raise RuntimeError(f"SUMO did not capture {destination}")
            finally:
                if connection is not None:
                    try:
                        connection.close(wait=False)
                    except Exception:
                        pass  # Terminate this owned process even after a lost socket.
                # This task created and owns this process; no user GUI is closed.
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)
        print(f"Captured {index + 1}/{frames}: {source.parent.name}", flush=True)


def record(controllers: list[str], seed: int, start_second: int, frames: int,
           fps: int, model_dir: Path, output: Path,
           isolated_frames: bool = False) -> list[Path]:
    if start_second < 0 or frames < 1 or fps < 1:
        raise ValueError("Start second, frame count and frame rate must be valid")
    if len(controllers) != len(set(controllers)):
        raise ValueError("Controller names must be distinct")
    output.mkdir(parents=True, exist_ok=True)
    outputs = []
    sources = {}
    demand_hash = None
    for controller in controllers:
        source = source_run(controller, seed, False, model_dir, 120)
        replay_dir = ROOT / "runs" / "gui" / "clip-replay" / f"seed-{seed}" / controller
        metrics, replay = write_replay(source, replay_dir)
        sources[controller] = {
            "source_run": str(source.resolve()),
            "metrics_sha256": hashlib.sha256((source / "metrics.json").read_bytes()).hexdigest(),
            "phase_replay_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
        }
        if controller not in ("fixed", "actuated", "pressure"):
            model_path = model_dir / f"{controller}.json"
            sources[controller]["model_path"] = str(model_path.resolve())
            sources[controller]["model_sha256"] = hashlib.sha256(
                model_path.read_bytes()).hexdigest()
        if demand_hash is None:
            demand_hash = metrics["demand_sha256"]
        elif metrics["demand_sha256"] != demand_hash:
            raise AssertionError("Clip controllers do not share scheduled demand")
        centre = next(node for node in ET.parse(source / "connected.net.xml")
                      .iter("junction") if node.get("id") == "C1")
        x, y = float(centre.attrib["x"]), float(centre.attrib["y"])
        frame_dir = frame_directory(source, start_second, frames)
        prepare_frame_directory(frame_dir)
        # Scheduled snapshots let SUMO synchronize its simulation and renderer.
        # Repeated TraCI screenshots can lose or block frames on Windows.
        # https://sumo.dlr.de/docs/sumo-gui.html#screenshots_1
        settings = ET.parse(ROOT / "config" / "demo.gui.xml").getroot()
        ET.SubElement(settings, "viewport", x=str(x), y=str(y), zoom="600")
        for index in range(frames):
            ET.SubElement(settings, "snapshot", time=str(start_second + index),
                          file=str((frame_dir / f"frame-{index:03d}.png").resolve()))
        settings_path = frame_dir / "viewsettings.xml"
        ET.ElementTree(settings).write(settings_path, encoding="utf-8", xml_declaration=True)
        if isolated_frames:
            capture_isolated_frames(source, replay, frame_dir, seed,
                                    start_second, frames, (x, y))
        else:
            subprocess.run([executable("sumo-gui"), "--net-file", str(source / "connected.net.xml"),
                     "--route-files", str(source / "demand.rou.xml"),
                     "--additional-files", str(replay), "--begin", "0",
                     "--end", str(start_second + frames + 1),
                     "--step-length", "1", "--seed", str(seed),
                     "--pedestrian.striping.dawdling", "0", "--time-to-teleport", "-1",
                     "--gui-settings-file", str(settings_path), "--quit-on-end",
                     "--start", "--delay", "0", "--no-step-log", "true", "--window-size", "1200,800"],
                       check=True, timeout=120)
        saved = sorted(frame_dir.glob("frame-*.png"))
        if len(saved) != frames or any(item.stat().st_size == 0 for item in saved):
            raise RuntimeError(f"SUMO wrote {len(saved)} of {frames} frames for {controller}")
        dimensions = {struct.unpack(">II", item.read_bytes()[16:24]) for item in saved}
        if len(dimensions) != 1 or min(next(iter(dimensions))) < 100:
            raise RuntimeError("SUMO needs an interactive desktop for full-size video frames")
        destination = output / f"sumo-{controller}-seed-{seed}-t{start_second}.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Install ffmpeg or use the saved PNG frames under runs/gui")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(frame_dir / "frame-%03d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                        str(destination)], check=True)
        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError(f"Video encoder did not write {destination}")
        outputs.append(destination)
    manifest = {"seed": seed, "start_second": start_second, "frames": frames,
                "fps": fps, "controllers": controllers, "demand_sha256": demand_hash,
                "source": "SUMO GUI phase replay; visualization only",
                "capture_mode": "independent deterministic frame replays" if isolated_frames else "scheduled snapshots",
                "source_runs": sources,
                "display": "Pedestrian icons enlarged 3x; simulation geometry unchanged"}
    (output / f"sumo-clips-seed-{seed}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controllers", nargs="+", default=["fixed", "ippo-pedestrian"])
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--start-second", type=int, default=138)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--isolated-frames", action="store_true",
                        help="Use one GUI process per frame to avoid repeated-capture driver stalls")
    parser.add_argument("--model-dir", type=Path, default=default_model_directory())
    parser.add_argument("--output", type=Path,
                        default=ROOT / "deliverables" / "presentation" / "assets" / "final-protocol")
    args = parser.parse_args()
    for path in record(args.controllers, args.seed, args.start_second, args.frames,
                       args.fps, args.model_dir, args.output, args.isolated_frames):
        print(path)


if __name__ == "__main__":
    main()
