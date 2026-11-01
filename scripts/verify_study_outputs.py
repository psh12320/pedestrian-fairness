"""Independently check completed tables, demand matching and source provenance.

This audits saved evidence; it does not rerun or change the frozen experiment.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(folder: Path, expected_runs: int, expected_seeds: set[int]) -> dict:
    rows = read_rows(folder / "runs.csv")
    assert len(rows) == expected_runs, (folder, len(rows), expected_runs)
    keys = [(r["scenario"], r["seed"], r["controller"], r["training_run"]) for r in rows]
    assert len(set(keys)) == len(keys), "Duplicate run keys"
    assert {int(r["seed"]) for r in rows} == expected_seeds
    groups = defaultdict(list)
    for row in rows:
        counts = [int(row[name]) for name in ("scheduled", "completed", "unfinished", "uninserted")]
        assert all(value >= 0 for value in counts)
        assert counts[0] == sum(counts[1:]), ("Vehicle conservation", row)
        groups[(row["scenario"], row["seed"])].append(row)
    for group in groups.values():
        assert len(group) == 33
        assert len({r["demand_sha256"] for r in group}) == 1, "Unmatched demand"
        for label in ("fixed", "actuated", "pressure"):
            assert sum(r["controller"] == label for r in group) == 1
        for method in ("ippo", "mappo"):
            for weight in (0, 3, 10):
                assert {int(r["training_run"]) for r in group
                        if r["controller"] == f"{method}-w{weight}"} == set(range(5))
    crossings = read_rows(folder / "crossings.csv")
    per_run = defaultdict(list)
    for row in crossings:
        per_run[(row["scenario"], row["seed"], row["controller"], row["training_run"])].append(row)
    assert set(per_run) == set(keys)
    for row, key in zip(rows, keys):
        assert len({r["crossing"] for r in per_run[key]}) == 12
        assert len(per_run[key]) == 12
        assert sum(int(r["unserved_count"]) for r in per_run[key]) == int(row["unserved_requests"])
    return {"runs": len(rows), "matched_demand_groups": len(groups),
            "crossing_rows": len(crossings), "unique_runs": True,
            "vehicle_conservation": True, "crossing_pending_counts_reconcile": True,
            "all_groups_have_three_baselines_and_thirty_replicas": True}


def main() -> None:
    folder = ROOT / "results/final-study"
    summary = json.loads((folder / "summary.json").read_text())
    result = {"main_study": audit(folder, 4290, set(range(9001, 9011)))}
    hashes = dict(summary["source_sha256"])
    training = json.loads((folder / "training_source_hashes.json").read_text())
    for name, digest in training["files"].items():
        if name in hashes:
            assert hashes[name] == digest, ("Training/evaluation mismatch", name)
        hashes[name] = digest
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    result["frozen_source_hashes_match"] = len(hashes)
    overload = [r for r in read_rows(folder / "runs.csv")
                if r["scenario"] == "vehicle-overload" and r["controller"] == "fixed"]
    fractions = [(int(r["unfinished"]) + int(r["uninserted"])) / int(r["scheduled"])
                 for r in overload]
    result["fixed_overload_unserved_fraction"] = {
        "min": min(fractions), "max": max(fractions),
        "mean": sum(fractions) / len(fractions),
        "all_ten_seeds_exceed_five_percent": all(f >= .05 for f in fractions)}
    entries = read_rows(folder / "crossing_comparison.csv")
    result["primary_minimum_crossing_entries"] = min(
        min(int(r["actuated_entries"]), int(r["minimum_model_entries_per_training_run"]))
        for r in entries)
    transfer = ROOT / "results/singapore-transfer"
    if (transfer / "summary.json").is_file():
        result["singapore_derived_transfer"] = audit(transfer, 330, set(range(9101, 9111)))
    else:
        result["singapore_derived_transfer"] = "pending; rerun this audit after completion"
    destination = ROOT / "results/reproducibility/final_output_audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
