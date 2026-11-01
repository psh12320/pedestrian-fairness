"""Export research figures from saved study tables with Matplotlib.

Figures never recompute or select study outcomes. SVG and PNG versions share
the same plotted data; the CSV and JSON remain the numerical evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def paper_plots(source, study, rows, records, plt) -> None:
    """Fixed 6.5-inch figures keep every label at least 12 pt in the course PDF."""
    primary = study["settings"]["primary_scenario"]
    if primary not in study["summary"]:
        return
    palette = {"ippo": "#2872b5", "mappo": "#c86316", "fixed": "#777777",
               "pressure": "#008475", "actuated": "#111111"}

    def color(label):
        return palette[label.split("-")[0]]

    def save(fig, name):
        # Do not crop: embedding at 6.5 inches preserves the specified font size.
        fig.savefig(source / f"{name}.png", dpi=180, bbox_inches=None)
        fig.savefig(source / f"{name}.svg", bbox_inches=None)
        plt.close(fig)

    with plt.rc_context({"font.size": 12, "axes.titlesize": 12,
                         "axes.labelsize": 12, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 12}):
        selected = sorted((row for row in rows if row["scenario"] == primary
                           and row["controller"] == "ippo-w3"), key=lambda row: row["crossing"])
        if selected:
            fig, ax = plt.subplots(figsize=(6.5, 4.6))
            fig.subplots_adjust(left=.17, right=.66, bottom=.16, top=.85)
            for y, row in enumerate(selected):
                point = row["p95_difference_seconds"]
                if point not in ("", None):
                    low, high = row["ci95_low"], row["ci95_high"]
                    if low not in ("", None) and high not in ("", None):
                        ax.plot([float(low), float(high)], [y, y], color=color("ippo"))
                    enough = min(int(row["actuated_entries"]),
                                 int(row["minimum_model_entries_per_training_run"])) >= 100
                    ax.scatter(float(point), y, edgecolor=color("ippo"),
                               facecolor=color("ippo") if enough else "white", s=38, zorder=3)
                else:
                    ax.text(0, y, "NA", va="center")
                ax.text(1.04, y, f"{int(row['actuated_entries'])}/{int(row['minimum_model_entries_per_training_run'])}",
                        transform=ax.get_yaxis_transform(), va="center")
                ax.text(1.43, y, f"{int(row['actuated_unserved'])}/{float(row['mean_model_unserved']):g}",
                        transform=ax.get_yaxis_transform(), va="center")
            ax.text(1.04, 1.04, "Entries\nA / M", transform=ax.transAxes)
            ax.text(1.43, 1.04, "Pending\nA / M", transform=ax.transAxes)
            ax.axvline(0, color="#777777", linestyle="--", linewidth=.8)
            ax.set_yticks(range(len(selected)), [row["crossing"].lstrip(":") for row in selected])
            ax.set_ylim(len(selected) - .5, -.5)
            ax.set_xlabel("p95 difference (s)")
            ax.grid(axis="x", alpha=.15)
            save(fig, "paper_crossings")

        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        fig.subplots_adjust(left=.16, right=.97, bottom=.31, top=.97)
        ordered_labels = ["actuated", "fixed", "pressure", *[
            f"{method}-w{weight}" for method in ("ippo", "mappo") for weight in (0, 3, 10)]]
        for label in ordered_labels:
            metrics = study["summary"][primary].get(label)
            if metrics is None:
                continue
            vehicle = metrics["vehicle_time_loss"]
            crossing = ({"mean": 0, "ci95_low": 0, "ci95_high": 0} if label == "actuated"
                        else study["primary_crossing_criterion"][label]["maximum_crossing_p95_difference"])
            x, y = vehicle["mean"], crossing["mean"]
            if x is None or y is None:
                continue
            if vehicle.get("ci95_low") is not None and vehicle.get("ci95_high") is not None:
                ax.plot([vehicle["ci95_low"], vehicle["ci95_high"]], [y, y], color=color(label), alpha=.6)
            if crossing.get("ci95_low") is not None and crossing.get("ci95_high") is not None:
                ax.plot([x, x], [crossing["ci95_low"], crossing["ci95_high"]], color=color(label), alpha=.6)
            marker = "s" if label.endswith("w3") else "^" if label.endswith("w10") else "o"
            ax.scatter(x, y, label=label, color=color(label), marker=marker, s=50, zorder=3)
        ax.axhline(0, color="#777777", linestyle="--", linewidth=.8)
        ax.set_xlabel("Completed-vehicle time loss (s)")
        ax.set_ylabel("Worst crossing p95 change (s)")
        ax.grid(alpha=.15)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.55, .015),
                   ncol=3, handletextpad=.3, columnspacing=.9, borderaxespad=0)
        save(fig, "paper_tradeoff")

        if records:
            weights = sorted({float(row["controller"].split("-w")[1]) for row in records})
            fig, axes = plt.subplots(len(weights), 1, figsize=(6.5, 5.6), sharex=True, squeeze=False)
            fig.subplots_adjust(left=.18, right=.97, bottom=.11, top=.95, hspace=.62)
            for ax, weight in zip(axes.flat, weights):
                groups, means = defaultdict(list), defaultdict(list)
                for row in records:
                    if row["controller"].endswith(f"-w{weight:g}"):
                        method = row["controller"].split("-")[0]
                        groups[(method, row["training_run"])].append((row["episode"], row["score"]))
                        means[(method, row["episode"])].append(row["score"])
                for (method, _), values in groups.items():
                    ax.plot(*zip(*sorted(values)), color=color(method), alpha=.35, linewidth=.8)
                for method in ("ippo", "mappo"):
                    values = sorted((episode, statistics.mean(scores))
                                    for (name, episode), scores in means.items() if name == method)
                    if values:
                        ax.plot(*zip(*values), color=color(method), linewidth=2,
                                linestyle="--" if method == "mappo" else "-",
                                marker="s" if method == "mappo" else "o",
                                markerfacecolor="none" if method == "mappo" else color(method),
                                label=method.upper())
                ax.set_title(f"Pedestrian weight {weight:g}", loc="left")
                ax.set_ylabel("Score")
                ax.grid(alpha=.15)
            axes[0, 0].legend(loc="best", ncol=2)
            axes[-1, 0].set_xticks(sorted({row["episode"] for row in records}))
            axes[-1, 0].set_xlabel("Training episode")
            save(fig, "paper_learning_curves")


def render_study_plots(source: Path) -> None:
    runtime_file = ROOT / ".local" / "report-python-runtime.txt"
    executable = (runtime_file.read_text(encoding="utf-8").strip()
                  if runtime_file.exists() else sys.executable)
    subprocess.run([executable, "-m", "src.traffic_control.figures",
                    "--source", str(source.resolve())], cwd=ROOT, check=True)


def create(source: Path) -> None:
    local = ROOT / ".local" / "report-python"
    if local.is_dir():
        sys.path.insert(0, str(local))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    study = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    primary = study["settings"]["primary_scenario"]
    minimum = study["settings"]["minimum_pooled_crossing_entries"]
    rows = list(csv.DictReader((source / "crossing_comparison.csv").open(encoding="utf-8")))

    def number(value):
        return None if value in (None, "") else float(value)

    def color(label):
        return ("#2872b5" if label.startswith("ippo") else "#c86316"
                if label.startswith("mappo") else {"actuated": "#111111",
                    "fixed": "#777777", "pressure": "#008475"}.get(label, "#555555"))

    def save(fig, name):
        fig.savefig(source / f"{name}.svg", bbox_inches="tight")
        fig.savefig(source / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # Trade-off: absolute completed-trip loss and the worst paired crossing change.
    fig, ax = plt.subplots(figsize=(9.5, 5.5), layout="constrained")
    for label, metrics in study["summary"].get(primary, {}).items():
        vehicle = metrics["vehicle_time_loss"]
        crossing = ({"mean": 0, "ci95_low": 0, "ci95_high": 0} if label == "actuated"
                    else study["primary_crossing_criterion"][label]["maximum_crossing_p95_difference"])
        x, y = vehicle["mean"], crossing["mean"]
        if x is None or y is None:
            continue
        # Draw interval endpoints directly, including intervals not centered on
        # the point estimate, which can occur for a bootstrapped maximum.
        if vehicle.get("ci95_low") is not None and vehicle.get("ci95_high") is not None:
            ax.plot([vehicle["ci95_low"], vehicle["ci95_high"]], [y, y], color=color(label), alpha=.6)
        if crossing.get("ci95_low") is not None and crossing.get("ci95_high") is not None:
            ax.plot([x, x], [crossing["ci95_low"], crossing["ci95_high"]], color=color(label), alpha=.6)
        marker = "s" if label.endswith("w3") else "^" if label.endswith("w10") else "o"
        ax.scatter(x, y, label=label, color=color(label), marker=marker, s=55, zorder=3)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=.8)
    ax.set(xlabel="Mean completed-trip vehicle time loss (s)",
           ylabel="Largest crossing p95 change vs actuated (s)",
           title="Vehicle and pedestrian trade-off: primary medium demand")
    ax.grid(alpha=.15)
    if primary in study["summary"]:
        ax.legend(title="Controller", loc="upper left", bbox_to_anchor=(1, 1))
        save(fig, "tradeoff")
    else:
        plt.close(fig)

    # All crossing estimates, with entry denominators visible beside each row.
    rows = [row for row in rows if row["scenario"] == primary]
    labels = [label for label in study["controllers"] if label != "actuated"
              and any(row["controller"] == label for row in rows)]
    crossings = sorted({row["crossing"] for row in rows})
    if labels and crossings:
        fig, axes = plt.subplots((len(labels) + 1) // 2, 2, figsize=(15, 3.9 * ((len(labels) + 1) // 2)),
                                 sharex=True, squeeze=False, layout="constrained")
        lookup = {(row["controller"], row["crossing"]): row for row in rows}
        for ax, label in zip(axes.flat, labels):
            for y, crossing_id in enumerate(crossings):
                row = lookup.get((label, crossing_id))
                if not row:
                    continue
                point = number(row["p95_difference_seconds"])
                baseline_n = int(row["actuated_entries"])
                model_n = int(row["minimum_model_entries_per_training_run"])
                ax.text(1.02, y, f"{baseline_n}/{model_n}", va="center", fontsize=8,
                        transform=ax.get_yaxis_transform())
                if point is None:
                    ax.text(0, y, "NA", va="center", fontsize=8)
                    continue
                low, high = number(row["ci95_low"]), number(row["ci95_high"])
                if low is not None and high is not None:
                    ax.plot([low, high], [y, y], color=color(label), linewidth=1.2)
                ax.scatter(point, y, s=30, edgecolor=color(label), zorder=3,
                           facecolor=color(label) if min(baseline_n, model_n) >= minimum else "white")
            ax.axvline(0, color="#777777", linestyle="--", linewidth=.8)
            ax.set_yticks(range(len(crossings)), crossings, fontsize=9)
            ax.set_ylim(len(crossings) - .5, -.8)
            ax.set_title(label, loc="left", fontweight="bold")
            ax.text(1.02, 1.01, "Entries\nA / min model", fontsize=8, transform=ax.transAxes)
            ax.grid(axis="x", alpha=.15)
            ax.set_xlabel("p95 difference vs actuated (s)")
            ax.tick_params(labelbottom=True)
        for ax in list(axes.flat)[len(labels):]:
            ax.set_visible(False)
        fig.suptitle("Every crossing: positive means longer waits\n"
                     f"Paired 95% intervals; open circles mean fewer than {minimum} entries in a comparison group",
                     fontsize=14)
        save(fig, "crossing_differences")

    validation = source / "validation.jsonl"
    records = []
    if validation.exists():
        records = [json.loads(line) for line in validation.read_text(encoding="utf-8").splitlines() if line]
        weights = sorted({float(row["controller"].split("-w")[1]) for row in records})
        fig, axes = plt.subplots(len(weights), 1, figsize=(9, 3 * len(weights)),
                                 sharex=True, squeeze=False, layout="constrained")
        for ax, weight in zip(axes.flat, weights):
            groups, mean_groups = defaultdict(list), defaultdict(list)
            for row in records:
                if row["controller"].endswith(f"-w{weight:g}"):
                    method = row["controller"].split("-")[0]
                    groups[(method, row["training_run"])].append((row["episode"], row["score"]))
                    mean_groups[(method, row["episode"])].append(row["score"])
            for (method, _), values in groups.items():
                values.sort()
                ax.plot(*zip(*values), color=color(method), linewidth=.8, alpha=.35)
            for method in ("ippo", "mappo"):
                values = sorted((ep, statistics.mean(scores)) for (name, ep), scores in mean_groups.items() if name == method)
                if values:
                    ax.plot(*zip(*values), color=color(method), linewidth=2.5,
                            linestyle="--" if method == "mappo" else "-",
                            marker="s" if method == "mappo" else "o",
                            markerfacecolor="none" if method == "mappo" else color(method),
                            label=method.upper())
            ax.set_title(f"Pedestrian reward weight {weight:g}", loc="left")
            ax.set_ylabel("Validation score")
            ax.grid(alpha=.15)
            ax.legend()
        axes[-1, 0].set_xticks(sorted({row["episode"] for row in records}))
        axes[-1, 0].set_xlabel("Training episode")
        fig.suptitle("Validation curves: thin lines are replicas, thick lines are group means\nLower is better; scores across different reward weights have different scales")
        save(fig, "learning_curves")
    paper_plots(source, study, rows, records, plt)
    (source / "figure_provenance.json").write_text(json.dumps({
        "library": "matplotlib", "version": matplotlib.__version__,
        "source": "Saved study summary, crossing comparisons and validation scores",
        "interpretation": "Simulation only; read service counts and crossing denominators with the figures"
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    create(parser.parse_args().source)


if __name__ == "__main__":
    main()
