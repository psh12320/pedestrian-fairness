"""Apply the frozen validation-only rule for a 300-episode extension."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .scenario import ROOT


def decide(manifest: dict, rows: list[dict]) -> dict:
    settings = manifest["settings"]
    if not manifest.get("complete_protocol") or manifest["episodes_requested"] != 150:
        raise ValueError("A complete 150-episode, 30-replica training run is required")
    expected = {(f"{method}-w{weight:g}", index)
                for method in ("ippo", "mappo")
                for weight in settings["reward_weights"]
                for index in range(len(settings["training_run_seeds"]))}
    present = {(item["controller"], int(item["training_run"]))
               for item in manifest["models"]}
    if present != expected:
        raise ValueError("Training manifest is missing a method, weight, or replica")
    scores = {}
    for item in rows:
        key = (item["controller"], int(item["training_run"]), int(item["episode"]))
        if key in scores or not math.isfinite(float(item["score"])):
            raise ValueError("Duplicate or nonfinite validation score")
        scores[key] = float(item["score"])
    comparisons = {}
    for label in sorted({name for name, _ in expected}):
        before, after = [], []
        for index in range(len(settings["training_run_seeds"])):
            try:
                before.append(scores[(label, index, 100)])
                after.append(scores[(label, index, 150)])
            except KeyError as exc:
                raise ValueError(f"Missing validation checkpoint for {label}-r{index}") from exc
        at_100, at_150 = statistics.mean(before), statistics.mean(after)
        improvement = (at_100 - at_150) / abs(at_100) if at_100 else 0.0
        improved_replicas = sum(new < old for old, new in zip(before, after))
        comparisons[label] = {
            "mean_score_episode_100": at_100,
            "mean_score_episode_150": at_150,
            "relative_improvement": improvement,
            "improved_replicas": improved_replicas,
            "triggers_extension": improvement >= 0.02 and improved_replicas >= 4,
        }
    extend = any(item["triggers_extension"] for item in comparisons.values())
    return {
        "rule": "At least 2% lower group mean score and at least 4 of 5 replicas improve, 100→150",
        "source": "validation scores only; current locked final seeds not inspected",
        "comparisons": comparisons,
        "extend_all_to_300": extend,
        "selected_training_episodes": 300 if extend else 150,
    }


def check_selected_budget(manifest: dict, decision: dict) -> int:
    """Require a complete study to use the budget chosen before test evaluation."""
    selected = decision.get("selected_training_episodes")
    extended = decision.get("extend_all_to_300")
    if type(extended) is not bool or selected != (300 if extended else 150):
        raise ValueError("Validation extension decision is malformed")
    budgets = {item.get("episodes_run") for item in manifest.get("models", [])}
    if (not manifest.get("complete_protocol")
            or manifest.get("episodes_requested") != selected
            or budgets != {selected}):
        raise ValueError(
            f"Trained models do not match the validation-selected {selected}-episode budget")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path,
                        default=ROOT / "runs" / "training-v3")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "v3")
    args = parser.parse_args()
    manifest = json.loads((args.model_dir / "training_manifest.json").read_text(
        encoding="utf-8"))
    rows = [json.loads(line) for line in
            (args.training_dir / "validation.jsonl").read_text(
                encoding="utf-8").splitlines()]
    result = decide(manifest, rows)
    destination = args.training_dir / "extension_decision.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"{result['selected_training_episodes']} episodes selected; {destination}")


if __name__ == "__main__":
    main()
