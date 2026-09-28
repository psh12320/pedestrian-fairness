"""Generate original report tables and an SVG trade-off plot from held-out runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

from .evaluate import CONTROLLERS, bootstrap_interval
from .scenario import ROOT


def fmt(item: dict, decimals: int = 1) -> str:
    if item["mean"] is None:
        return "NA"
    return (f"{item['mean']:.{decimals}f} "
            f"[{item['ci95_low']:.{decimals}f}, {item['ci95_high']:.{decimals}f}]")


def tradeoff_svg(summary: dict, destination: Path) -> None:
    medium = summary["vmedium-pmedium"]
    points = [(label, medium[label]["vehicle_time_loss"]["mean"],
               medium[label]["worst_crossing_p95"]["mean"]) for label in CONTROLLERS]
    x_values = [x for _, x, _ in points]
    y_values = [y for _, _, y in points]
    x_min, x_max = min(x_values) - 2, max(x_values) + 2
    y_min, y_max = max(0, min(y_values) - 10), max(y_values) + 10
    def x_coord(value: float) -> float:
        return 85 + 640 * (value - x_min) / (x_max - x_min)
    def y_coord(value: float) -> float:
        return 430 - 340 * (value - y_min) / (y_max - y_min)
    colors = {"fixed": "#6b7280", "actuated": "#2563eb", "ippo-vehicle": "#16a34a",
              "ippo-pedestrian": "#15803d", "mappo-vehicle": "#d97706",
              "mappo-pedestrian": "#b45309"}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="850" height="520" viewBox="0 0 850 520">',
           '<rect width="850" height="520" fill="white"/>',
           '<text x="85" y="40" font-size="22" font-family="sans-serif">Vehicle–pedestrian trade-off (medium demand)</text>',
           '<line x1="85" y1="430" x2="725" y2="430" stroke="black"/>',
           '<line x1="85" y1="90" x2="85" y2="430" stroke="black"/>',
           '<text x="250" y="485" font-size="15" font-family="sans-serif">Mean completed-trip vehicle time loss (s)</text>',
           '<text transform="translate(25 355) rotate(-90)" font-size="15" font-family="sans-serif">Worst crossing p95 wait (s)</text>']
    for tick in range(5):
        x_value = x_min + (x_max - x_min) * tick / 4
        y_value = y_min + (y_max - y_min) * tick / 4
        svg.append(f'<text x="{x_coord(x_value)-12:.1f}" y="453" font-size="12" font-family="sans-serif">{x_value:.1f}</text>')
        svg.append(f'<text x="{y_coord(y_value)-38:.1f}" y="{y_coord(y_value)+4:.1f}" font-size="12" font-family="sans-serif">{y_value:.0f}</text>')
    repeated = {}
    for label, x_value, y_value in points:
        x, y = x_coord(x_value), y_coord(y_value)
        key = (round(x), round(y))
        offset = repeated.get(key, 0)
        repeated[key] = offset + 1
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colors[label]}"/>')
        svg.append(f'<text x="{x+9:.1f}" y="{y-8+16*offset:.1f}" font-size="12" font-family="sans-serif">{html.escape(label)}</text>')
    svg.append('</svg>')
    destination.write_text("\n".join(svg) + "\n", encoding="utf-8")


def generate(source: Path | None = None, destination: Path | None = None) -> None:
    source = source or ROOT / "runs" / "evaluation"
    destination = destination or ROOT / "results"
    destination.mkdir(parents=True, exist_ok=True)
    result = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    summary = result["summary"]
    rows = list(csv.DictReader((source / "runs.csv").open(encoding="utf-8")))
    crossing_rows = list(csv.DictReader((source / "crossings.csv").open(encoding="utf-8")))
    if len(rows) != 10 * 5 * len(CONTROLLERS):
        raise ValueError(f"Expected 300 complete held-out runs, found {len(rows)}")
    for name in ("runs.csv", "crossings.csv", "summary.json"):
        shutil.copyfile(source / name, destination / name)

    crossing_groups = defaultdict(list)
    for row in crossing_rows:
        crossing_groups[(row["scenario"], row["controller"], row["crossing"],
                         row["crossing_edges"])].append(row)
    crossing_summary = []
    for (scenario, controller, crossing, edges), group in sorted(crossing_groups.items()):
        mean_wait = bootstrap_interval([float(row["mean_wait_seconds"]) for row in group
                                        if row["mean_wait_seconds"]], 1000, 4246)
        p95_wait = bootstrap_interval([float(row["p95_wait_seconds"]) for row in group
                                       if row["p95_wait_seconds"]], 1000, 4246)
        crossing_summary.append({"scenario": scenario, "controller": controller,
                                 "crossing": crossing, "crossing_edges": edges,
                                 "mean_wait_seconds": mean_wait["mean"],
                                 "mean_wait_seed_n": mean_wait["n"],
                                 "mean_wait_ci95_low": mean_wait["ci95_low"],
                                 "mean_wait_ci95_high": mean_wait["ci95_high"],
                                 "p95_wait_seconds": p95_wait["mean"],
                                 "p95_wait_seed_n": p95_wait["n"],
                                 "p95_wait_ci95_low": p95_wait["ci95_low"],
                                 "p95_wait_ci95_high": p95_wait["ci95_high"],
                                 "total_entries": sum(int(row["entered_count"]) for row in group),
                                 "total_unserved": sum(int(row["unserved_count"]) for row in group)})
    with (destination / "crossing_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(crossing_summary[0]))
        writer.writeheader()
        writer.writerows(crossing_summary)
    tradeoff_svg(summary, destination / "tradeoff.svg")

    lines = ["# CS4246 Team 38 — Simulation study", "",
             "## Question and scope", "",
             "Can local multi-agent PPO signal control improve vehicle flow without shifting excessive delay to pedestrians? This is a newly built, synthetic three-junction SUMO corridor. It is not a Singapore field validation or evidence for deployment. Pedestrian-aware RL and multi-agent signal control already exist; this study focuses on matched comparisons and per-crossing tail waits.", "",
             "## Methods", "",
             "- Three connected four-arm signalised junctions, each with four protected crossings and sidewalks. Vehicle flows traverse the corridor or cross a local north–south arm. Pedestrians use explicit trips. All controllers share one signal state machine: minimum vehicle green, yellow, all-red, exclusive pedestrian green, and clearance based on the longest crossing and 0.8 m/s walking speed. Occupancy extends clearance if a person remains on a crossing.",
             "- Fixed-time and pedestrian-actuated baselines are compared with IPPO and MAPPO under vehicle-only and pedestrian-penalised rewards. Each learned actor sees its junction's queues, crossing requests and phase only. IPPO uses local value estimators; MAPPO uses a network-wide value estimator during training. The implementation uses linear policies and critics trained with clipped PPO and GAE, so its function class and training budget are limited.",
             "- Crossing wait is the elapsed time between the first stopped crossing-approach event and entry onto the crossing edge, sampled every one second. SUMO's person `waitingTime` is only an event trigger. Immediate entry gives zero observed wait. Requests with no entry by the horizon remain unserved. Crowding before the crossing line and one-second resolution limit this measure.",
             "- Thirty-six training episodes per learned controller use seeds 1000–1035 with three rotating demand levels. Evaluation uses the five held-out seeds 3001–3005, all nine combinations of three vehicle and three pedestrian levels, plus medium demand with 0.8 m/s pedestrians. Controllers in each paired run receive byte-identical demand. The tables give means and percentile-bootstrap 95% intervals across seeds; intervals with only five seeds are descriptive and should not be read as strong significance tests.", "",
             "## Medium-demand results", "",
             "Mean [95% interval] across five held-out runs. Vehicle time loss is for **completed trips** and must be read with unfinished and uninserted counts. The worst-crossing p95 is selected within each run before averaging across seeds.", "",
             "| Controller | Vehicle time loss (s) | Pedestrian mean wait (s) | Worst crossing p95 (s) | Completed | Unfinished | Uninserted | Unserved requests |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    medium = summary["vmedium-pmedium"]
    for label in CONTROLLERS:
        item = medium[label]
        lines.append("| " + label + " | " + " | ".join(fmt(item[key]) for key in
            ("vehicle_time_loss", "pedestrian_mean_wait", "worst_crossing_p95",
             "completed", "unfinished", "uninserted", "unserved_requests")) + " |")
    lines.extend(["", "![Medium-demand vehicle–pedestrian trade-off](tradeoff.svg)", "",
                  "## Paired differences from fixed-time at medium demand", "",
                  "Negative vehicle time loss is a gain; positive pedestrian waiting is a cost. Every difference uses the same demand seed.", "",
                  "| Controller | Vehicle time loss difference (s) | Pedestrian mean-wait difference (s) | Worst crossing p95 difference (s) |",
                  "|---|---:|---:|---:|"])
    for label in CONTROLLERS[1:]:
        item = result["paired_vs_fixed"]["vmedium-pmedium"][label]
        lines.append("| " + label + " | " + " | ".join(fmt(item[key]) for key in
            ("vehicle_time_loss", "pedestrian_mean_wait", "worst_crossing_p95")) + " |")
    lines.extend(["", "## Demand sensitivity", "",
                  "Each cell is mean completed-trip vehicle time loss (s) across five held-out runs. Full uncertainty and all service counts are in `summary.json` and `runs.csv`.", "",
                  "| Demand | " + " | ".join(CONTROLLERS) + " |",
                  "|---|" + "---:|" * len(CONTROLLERS)])
    for scenario in sorted(name for name in summary if not name.endswith("slow")):
        lines.append("| " + scenario + " | " + " | ".join(
            f"{summary[scenario][label]['vehicle_time_loss']['mean']:.1f}" for label in CONTROLLERS) + " |")
    lines.extend(["", "## Slower-walker check", "",
                  "Medium demand is rerun with pedestrian free speed reduced from 1.3 to 0.8 m/s. The clearance rule is shared by all controllers.", "",
                  "| Controller | Ordinary mean wait (s) | Slow mean wait (s) | Ordinary worst crossing p95 (s) | Slow worst crossing p95 (s) | Slow unserved requests (mean/run) |",
                  "|---|---:|---:|---:|---:|---:|"])
    slow = summary["vmedium-pmedium-slow"]
    for label in CONTROLLERS:
        lines.append("| " + label + " | " + " | ".join(fmt(item) for item in
            (medium[label]["pedestrian_mean_wait"], slow[label]["pedestrian_mean_wait"],
             medium[label]["worst_crossing_p95"], slow[label]["worst_crossing_p95"],
             slow[label]["unserved_requests"])) + " |")
    lines.extend(["", "## Individual crossings", "",
                  "`crossing_summary.csv` reports mean wait, p95 wait, bootstrap intervals, entries and unserved requests for **every crossing under every controller and scenario**. `crossings.csv` retains every seed-level crossing result. The following table names each controller's crossing with the largest mean p95 wait at medium demand. This crossing can differ across seeds, so its mean p95 need not equal the mean of each run's worst crossing above.", "",
                  "| Controller | Crossing | Road edges crossed | Mean wait (s) | Mean p95 wait (s) | Entries | Unserved |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for label in CONTROLLERS:
        candidates = [row for row in crossing_summary if row["scenario"] == "vmedium-pmedium"
                      and row["controller"] == label]
        worst = max(candidates, key=lambda row: row["p95_wait_seconds"] or 0)
        lines.append(f"| {label} | {worst['crossing']} | {worst['crossing_edges']} | "
                     f"{worst['mean_wait_seconds']:.1f} | {worst['p95_wait_seconds']:.1f} | "
                     f"{worst['total_entries']} | {worst['total_unserved']} |")
    matching = 0
    paired_runs = defaultdict(dict)
    for row in rows:
        paired_runs[(row["scenario"], row["seed"])][row["controller"]] = row
    compare_keys = ("completed", "vehicle_time_loss", "pedestrian_mean_wait",
                    "worst_crossing_p95", "unserved_requests")
    for pair in paired_runs.values():
        if all(pair["ippo-vehicle"][key] == pair["mappo-pedestrian"][key]
               for key in compare_keys):
            matching += 1
    all_vehicles_served = all(int(row["unfinished"]) == 0 and int(row["uninserted"]) == 0
                              for row in rows)
    lines.extend(["", "## Interpretation and limits", "",
                  ("Every scheduled vehicle completed in these evaluated demand levels. The accounting code still reports unfinished and uninserted trips, but this experiment does not test controllers under vehicle spillback severe enough to leave trips unserved. " if all_vehicles_served else "Some runs leave vehicle trips unfinished or uninserted; read time loss together with those counts. ")
                  + f"IPPO-vehicle and MAPPO-pedestrian have identical aggregate outcomes on {matching} of {len(paired_runs)} paired scenario-seed cases, despite different trained weights. Their labels therefore should not be read as evidence of distinct behaviour everywhere. "
                  + "The synthetic corridor, straight-only vehicle movements, exclusive pedestrian phase, linear policies, short training budget, small number of held-out seeds, and event sampling limit external validity. Comparisons reveal simulation trade-offs under matched demand; they do not establish a better deployed signal controller.", "",
                  "## Reproduce", "",
                  "Run `python -m unittest discover -s tests -v`, then `python -m src.traffic_control.train`, `python -m src.traffic_control.evaluate`, and `python -m src.traffic_control.report` from the repository root with SUMO and TraCI available. Configuration and seeds are in `config/`. Raw episode files are under Git-ignored `runs/`; the aggregate data and trained model weights are saved with this project.", "",
                  "## Team roles", "",
                  "Jason and Jarrell lead simulation and baselines; Shricharan and Jewi lead MARL implementation and training; Reet and Huan Kiat lead pedestrian evaluation and analysis. All team members contribute to the final report and presentation.", "",
                  "## References", "",
                  "- [SUMO pedestrian simulation](https://sumo.dlr.de/docs/Simulation/Pedestrians.html), [TraCI crossing tutorial](https://sumo.dlr.de/docs/Tutorials/TraCIPedCrossing.html), and [tripinfo output](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html).",
                  "- [The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](https://arxiv.org/abs/2103.01955) describes centralized-value MAPPO.",
                  "- [PedSLight](https://www.frontiersin.org/journals/future-transportation/articles/10.3389/ffutr.2026.1953597/full) is an existing pedestrian- and safety-aware RL signal-control study; pedestrian-aware MARL is not claimed as a new idea here.", ""])
    (destination / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    slides = ["# CS4246 Team 38 — presentation outline", "",
              "## 1. Research question", "Can multi-agent control improve vehicle flow without imposing long waits at specific crossings?", "",
              "## 2. Experimental setup", "New synthetic three-junction SUMO corridor; six controllers; matched held-out demand and seeds.", "",
              "## 3. Safe action boundary", "Show EW, NS and all-walk states with minimum green, yellow, all-red and occupancy-extended clearance.", "",
              "## 4. Measurement", "Explain crossing approach/entry event timestamps, per-crossing p95 and unserved requests; show vehicle trip accounting.", "",
              "## 5. IPPO versus MAPPO", "Local actors in both methods. IPPO local critic; MAPPO training critic sees the corridor. Vehicle and pedestrian rewards give four learned controllers.", "",
              "## 6. Main comparison", "Use the medium-demand table and `tradeoff.svg` from `REPORT.md`; read service counts beside time loss.", "",
              "## 7. Stress tests", "Use the demand grid, slow-walker table and worst-crossing table; explain seed uncertainty.", "",
              "## 8. Conclusion", "State only what the simulated paired results support. Highlight limits of the synthetic network, linear policies and five evaluation seeds.", ""]
    (destination / "PRESENTATION.md").write_text("\n".join(slides), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    generate(args.source, args.output)


if __name__ == "__main__":
    main()
