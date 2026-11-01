"""Reproducible tables and standard plotting-library figures from saved runs."""
import csv
import json
import statistics

from .analysis import quantile


def export_figures(root, groups, primary):
    rows = []
    for group in groups:
        for record in group["records"]:
            result = record["result"]
            for crossing, data in result["crossings"].items():
                rows.append({"scenario": group["scenario"], "seed": group["seed"],
                    "controller": record["controller"], "replica": record["replica"], "crossing": crossing,
                    "entries": data["entries"], "mean_wait": data.get("mean"), "p95_wait": data["p95"],
                    "pending_at_cutoff": data["pending_at_cutoff"], "unresolved": data["unresolved_after_followup"],
                    "unknown": data["unknown_cohort_entries"], "lost": data["lost"],
                    **{f"vehicle_{k}": v for k, v in result["vehicles"].items() if not isinstance(v, list)}})
    if rows:
        with (root / "crossing_runs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        (root / "figures-unavailable.txt").write_text("Install matplotlib to render plots; tables and inference remain available.")
        return
    figure, axis = plt.subplots(figsize=(12, 5))
    for label, data in primary.items():
        point = data["point_estimates"]
        if point:
            axis.plot(data["crossing_order"], point[1], marker="o", markersize=3, label=label)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_ylabel("Crossing p95 difference from actuated (s)")
    axis.set_title("Primary scenario: mean replica p95 differences; point estimates")
    axis.tick_params(axis="x", rotation=45)
    if axis.lines:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "crossing_tradeoffs.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 5))
    for label, data in primary.items():
        point = data["point_estimates"]
        if point:
            axis.scatter(point[0], max(point[1]))
            axis.annotate(label, (point[0], max(point[1])), fontsize=8)
    axis.axhline(0, color="black", linewidth=.8)
    axis.axvline(0, color="black", linewidth=.8)
    axis.set_xlabel("Completed vehicle time-loss difference (s)")
    axis.set_ylabel("Worst crossing p95 difference (s)")
    axis.set_title("Primary scenario trade-off; service and uncertainty gates also apply")
    figure.tight_layout()
    figure.savefig(root / "vehicle_pedestrian_tradeoff.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for path in sorted(root.glob("*-r*/training.jsonl")):
        history = [json.loads(line) for line in path.read_text().splitlines()]
        history = list({r["episode"]: r for r in history}.values())
        measured = [r for r in history if r.get("vehicle_mean_time_loss") is not None]
        validation = [r for r in history if r.get("validation_score") is not None]
        axes[0].plot([r["episode"] for r in measured], [r["vehicle_mean_time_loss"] for r in measured], alpha=.5)
        axes[1].plot([r["episode"] for r in validation], [r["validation_score"][1] for r in validation], alpha=.5, label=path.parent.name)
    axes[0].set_title("Training outcome (changing demand curriculum)")
    axes[0].set_ylabel("Completed vehicle time loss (s)")
    axes[1].set_title("Validation worst positive crossing difference")
    axes[1].set_ylabel("Seconds; 1e6 indicates missing crossing estimate")
    for axis in axes:
        axis.set_xlabel("Episode")
    if len(list(root.glob("*-r*/training.jsonl"))) <= 6:
        axes[1].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(root / "learning_curves.png", dpi=160)
    plt.close(figure)
