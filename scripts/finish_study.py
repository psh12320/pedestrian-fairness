"""Finish the frozen study, keeping resumable stages and compact logs.

Run from the repository root. Do not run two copies against the same folders.
Use --wait-for-training only when training-v3 is already running separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "src/traffic_control/cli.py", "src/traffic_control/connected.py",
    "src/traffic_control/scenario.py", "src/traffic_control/signals.py",
    "src/traffic_control/metrics.py", "src/traffic_control/ppo.py",
    "src/traffic_control/train.py", "src/traffic_control/train_parallel.py",
    "config/milestone.json", "config/training.json",
)


def snapshot(directory: Path) -> None:
    """Never resume a training directory under changed source files."""
    file = directory / "training_source_hashes.json"
    if file.exists():
        saved = json.loads(file.read_text(encoding="utf-8-sig"))
        changed = [name for name, digest in saved["files"].items()
                   if not (ROOT / name).is_file() or
                   hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
        if changed:
            raise RuntimeError(f"Training source changed; use a new protocol version: {changed}")
        return
    names = list(SOURCES)
    portable = ".local/sumo-1.27.1/bin/sumo.exe"
    if (ROOT / portable).is_file():
        names.append(portable)
    directory.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({
        "purpose": "Frozen source snapshot before training",
        "training_directory": str(directory.relative_to(ROOT)),
        "files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in names},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def complete(manifest: Path) -> bool:
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("complete_protocol") is True and len(data.get("models", [])) == 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-training", action="store_true")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    logs = ROOT / "runs" / "final-pipeline"
    logs.mkdir(parents=True, exist_ok=True)
    status_file = logs / "status.json"
    environment = os.environ.copy()
    portable_tools = ROOT / ".local" / "sumo-1.27.1" / "tools"
    if portable_tools.is_dir():
        environment["PYTHONPATH"] = str(portable_tools) + os.pathsep + environment.get("PYTHONPATH", "")
        environment.pop("SUMO_HOME", None)

    def status(stage: str, state: str, **details) -> None:
        record = {"stage": stage, "state": state,
                  "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  **details}
        temp = status_file.with_suffix(".tmp")
        temp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temp.replace(status_file)
        print(f"{state}: {stage}", flush=True)

    def run(stage: str, module: str, *arguments: str, python: str = sys.executable) -> None:
        command = [python, "-m", module, *arguments]
        status(stage, "running", command=command)
        with (logs / f"{stage}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status(stage, "failed", exit_code=result.returncode,
                   log=str(logs / f"{stage}.log"))
            raise SystemExit(result.returncode)
        status(stage, "complete")

    training = ROOT / "runs" / "training-v3"
    models = ROOT / "models" / "v3"
    run("measurement-audit", "src.traffic_control.measurement_resolution")
    snapshot(training)
    if args.wait_for_training:
        status("training-150", "waiting_for_existing_run")
        deadline = time.monotonic() + 6 * 60 * 60
        while not complete(models / "training_manifest.json"):
            if time.monotonic() >= deadline:
                status("training-150", "failed", reason="Six-hour wait expired")
                raise SystemExit(1)
            time.sleep(15)
    else:
        run("training-150", "src.traffic_control.train_parallel", "--episodes", "150",
            "--output", str(training), "--model-dir", str(models),
            "--workers", str(args.workers))
    snapshot(training)
    run("validation-decision", "src.traffic_control.validation_decision")
    decision = json.loads((training / "extension_decision.json").read_text(encoding="utf-8"))
    evaluation = ROOT / "runs" / "evaluation-v4"
    if decision["extend_all_to_300"]:
        training, models = ROOT / "runs" / "training-v4", ROOT / "models" / "v4"
        evaluation = ROOT / "runs" / "evaluation-v5"
        snapshot(training)
        run("training-300", "src.traffic_control.train_parallel", "--episodes", "300",
            "--output", str(training), "--model-dir", str(models),
            "--workers", str(args.workers))
        snapshot(training)
    run("locked-evaluation", "src.traffic_control.evaluate_parallel",
        "--model-dir", str(models), "--output", str(evaluation),
        "--workers", str(args.workers))
    run("singapore-transfer", "src.traffic_control.singapore_transfer",
        "--model-dir", str(models), "--output", str(ROOT / "results" / "singapore-transfer"))
    status("measured-study", "complete", models=str(models),
           selected_episodes=decision["selected_training_episodes"],
           remaining="Analyse evaluation outputs and prepare the report and presentation")


if __name__ == "__main__":
    main()
