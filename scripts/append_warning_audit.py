"""Preserve SUMO warnings and add their limitations to the completed reports.

Run after finish_study.py. This does not change frozen simulations or summaries.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "\n\n## Final console-warning audit\n"


def main() -> None:
    records = {}
    audit_path = ROOT / "results/reproducibility/final_log_audit.json"
    previous = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    for report_dir, log_names in (
        ("final-study", ["locked-evaluation.log", "locked-evaluation-interrupted-2026-09-30.log"]),
        ("singapore-transfer", ["singapore-transfer.log"]),
    ):
        report = ROOT / "results" / report_dir / "REPORT.md"
        if not report.exists():
            raise FileNotFoundError(f"Complete the study before this audit: {report}")
        warnings = []
        log_counts = {}
        for name in log_names:
            path = ROOT / "runs/final-pipeline" / name
            if not path.exists():
                continue
            lines = path.read_text(errors="replace").splitlines()
            jam = [line for line in lines if re.search(r"Person.*jammed", line)]
            warnings.extend(jam)
            log_counts[name] = {
                "pedestrian_jam_warning_lines": len(jam),
                "collision_warning_lines": sum(bool(re.search(r"\bcollision\b", line, re.I)) for line in lines),
                "teleporting_warning_lines": sum(bool(re.search(r"\bteleporting\b", line, re.I)) for line in lines),
            }
        summary_hash = hashlib.sha256((report.parent / "summary.json").read_bytes()).hexdigest()
        record = {"summary_sha256": summary_hash, "logs": log_counts, "jam_warning_lines": warnings}
        old = previous.get("studies", {}).get(report_dir, {})
        # A later cached evaluation may print only 'reused'. Preserve evidence
        # tied to the identical summary instead of falsely replacing it by zero.
        if (old.get("summary_sha256") == summary_hash
                and len(old.get("jam_warning_lines", [])) > len(warnings)):
            record = old
            warnings = old["jam_warning_lines"]
        records[report_dir] = record
        text = report.read_text(encoding="utf-8").split(MARKER)[0]
        text += MARKER + (
            f"\nThe retained console logs contain **{len(warnings)} pedestrian-jam warning lines**. "
            "This is a warning-line count, not a count of unique people or failed runs. "
            "SUMO's striping jam recovery permits movement ignoring obstacles, and "
            "does not register pedestrian collisions while jammed. Passing the "
            "signal-state checks therefore does not establish collision-free trajectories. "
            "Crossing wait ends at entry and excludes delays during traversal. "
            "See the [official jamming explanation](https://sumo.dlr.de/docs/Simulation/Pedestrians.html#jamming) "
            "and [compact log audit](../reproducibility/final_log_audit.json). "
            "A read-only replay of one short-spacing case reproduced every saved "
            "outcome measure and kept vehicle links red while the person occupied "
            "the crossing; its [trace and audit](../reproducibility/pedestrian_jam_audit.json) "
            "provide a diagnostic, not a validation of physical safety.\n")
        report.write_text(text, encoding="utf-8")
    audit_path.write_text(
        json.dumps({"scope": "Console warnings from completed protocols; no filtering of measured runs",
                    "studies": records}, indent=2) + "\n", encoding="utf-8")
    print({name: len(record["jam_warning_lines"]) for name, record in records.items()})


if __name__ == "__main__":
    main()
