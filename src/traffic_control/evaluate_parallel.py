"""Evaluate independent (scenario, demand-seed) groups with up to four workers.

Workers create only distinct atomic group checkpoints. After all groups finish,
the parent reuses those checkpoints to produce one deterministic summary.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .evaluate import evaluate, scenario_specs
from .scenario import ROOT


def _worker(task: tuple[str, int, str, str, bool]) -> tuple[str, int]:
    scenario, seed, output, model_dir, pilot = task
    evaluate(Path(output), [seed], [scenario], Path(model_dir),
             smoke=pilot, group_only=True)
    return scenario, seed


def evaluate_parallel(output: Path | None = None, model_dir: Path | None = None,
                      workers: int = 4, pilot: bool = False,
                      seeds: list[int] | None = None,
                      scenarios: list[str] | None = None) -> dict:
    if not 1 <= workers <= 4:
        raise ValueError("Use between one and four SUMO workers")
    settings = json.loads((ROOT / "config" / "evaluation.json").read_text(
        encoding="utf-8"))
    if pilot:
        output = output or ROOT / "runs" / "pilot-parallel-local-reward-evaluation"
        model_dir = model_dir or ROOT / "runs" / "pilot-parallel-local-reward-models"
        seeds = seeds or [6001]
        scenarios = scenarios or ["vmedium-pmedium"]
    else:
        output = output or ROOT / "runs" / "evaluation-v4"
        model_dir = model_dir or ROOT / "models" / "v3"
        seeds = seeds or settings["seeds"]
        scenarios = scenarios or list(scenario_specs(settings))
    if not pilot and set(seeds) & set(settings["exploratory_seeds"]):
        raise ValueError("Final evaluation cannot use exploratory seeds")
    known = scenario_specs(settings)
    unknown = set(scenarios) - set(known)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(scenario, seed, str(output.resolve()), str(model_dir.resolve()), pilot)
             for scenario in scenarios for seed in seeds]
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, task): task for task in tasks}
        for future in as_completed(futures):
            scenario, seed, _, _, _ = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{scenario} seed {seed}: {exc}")
                print(f"FAILED {failures[-1]}", flush=True)
            else:
                print(f"completed {scenario} seed {seed}", flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))
    # evaluate() reads every completed group in the same scenario/seed order as
    # the serial protocol, then writes the final aggregate files exactly once.
    return evaluate(output, seeds, scenarios, model_dir, smoke=pilot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--scenarios", nargs="+")
    args = parser.parse_args()
    result = evaluate_parallel(args.output, args.model_dir, args.workers,
                               args.pilot, args.seeds, args.scenarios)
    print(f"{len(result['scenarios'])} scenarios and "
          f"{len(result['evaluation_seeds'])} demand seed(s) aggregated")


if __name__ == "__main__":
    main()
