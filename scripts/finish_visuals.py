"""Build final matched SUMO captures and slides on an interactive desktop.

Run only one copy. The optional wait follows this project's existing local
study process; it does not schedule a recurring job or contact any service.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def comparison_filter() -> str:
    """Keep exported videos understandable when played outside the slide deck."""
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"
    font = ("fontfile='" + font_path.as_posix().replace(":", "\\:") + "':"
            if font_path.exists() else "")
    style = ":fontsize=28:fontcolor=white:box=1:boxcolor=black@0.75:boxborderw=8"
    return ("hstack=inputs=2,scale=1920:-2,pad=iw:ih+50:0:0:color=white,"
            f"drawtext={font}text='Fixed-time':x=24:y=20{style},"
            f"drawtext={font}text='IPPO pedestrian weight 3':x=984:y=20{style},"
            f"drawtext={font}text='Matched seed 6001 | 2x playback | illustrative replay':"
            "x=24:y=h-34:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.75:boxborderw=5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-study", action="store_true")
    parser.add_argument("--node", default=shutil.which("node"))
    args = parser.parse_args()
    if not args.node or not shutil.which("ffmpeg"):
        parser.error("Node.js with the presentation runtime, and ffmpeg, are required")
    folder = ROOT / "runs" / "final-visuals"
    folder.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / ".local/sumo-1.27.1/tools") + os.pathsep + environment.get("PYTHONPATH", "")
    environment.pop("SUMO_HOME", None)

    def status(stage: str, state: str, **details) -> None:
        data = {"stage": stage, "state": state,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details}
        temporary = folder / "status.tmp"
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(folder / "status.json")
        print(f"{state}: {stage}", flush=True)

    def execute(stage: str, command: list[str]) -> None:
        status(stage, "running")
        with (folder / f"{stage}.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=ROOT, env=environment,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            status(stage, "failed", exit_code=code)
            raise SystemExit(code)
        status(stage, "complete")

    pipeline = ROOT / "runs/final-pipeline/status.json"
    if args.wait_for_study:
        status("study", "waiting_for_existing_run")
        deadline = time.monotonic() + 12 * 3600
        while True:
            current = json.loads(pipeline.read_text(encoding="utf-8")) if pipeline.exists() else {}
            if current.get("state") == "failed":
                status("study", "failed", reason=current)
                raise SystemExit(1)
            if current.get("stage") == "measured-study-and-pdf" and current.get("state") == "complete":
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Twelve-hour study wait expired")
            time.sleep(15)
    decision = json.loads((ROOT / "runs/training-v3/extension_decision.json").read_text(encoding="utf-8"))
    models = ROOT / ("models/v4" if decision["extend_all_to_300"] else "models/v3")
    manifest = json.loads((models / "training_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete_protocol") is not True or len(manifest["models"]) != 30:
        raise ValueError("All 30 final-budget replicas are required")
    if not (ROOT / "results/final-study/summary.json").exists():
        raise ValueError("Complete the locked evaluation before final visuals")
    assets = ROOT / "deliverables/presentation/assets/final-protocol"
    common = ["--controllers", "fixed", "ippo-pedestrian", "--model-dir", str(models),
              "--output", str(assets), "--seed", "6001"]
    execute("stills", [sys.executable, "-m", "src.traffic_control.capture_gui", *common,
                       "--second", "147", "--boundary", "240", "70", "320", "150"])
    execute("clips", [sys.executable, "-m", "src.traffic_control.capture_clip", *common,
                      "--start-second", "138", "--frames", "20", "--fps", "2",
                      "--isolated-frames"])
    execute("comparison-video", [shutil.which("ffmpeg"), "-y", "-loglevel", "error",
        "-i", str(assets / "sumo-fixed-seed-6001-t138.mp4"),
        "-i", str(assets / "sumo-ippo-pedestrian-seed-6001-t138.mp4"),
        "-filter_complex", comparison_filter(), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "22",
        str(assets / "sumo-fixed-vs-ippo-pedestrian-seed-6001.mp4")])
    execute("presentation", [args.node, "deliverables/presentation/build_team38_deck.mjs"])
    status("visual-artifacts", "complete", remaining="Render and visually inspect the final slides and PDF")


if __name__ == "__main__":
    main()
