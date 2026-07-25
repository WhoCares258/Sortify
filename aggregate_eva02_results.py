from pathlib import Path
import re

import pandas as pd


RUNS_DIR = Path("runs")
SEEDS = [0, 1, 2, 3, 4]


def read_overall_metrics(path: Path) -> dict[str, float]:
    """Read final test metrics saved by the training script."""
    text = path.read_text(encoding="utf-8")

    patterns = {
        "Accuracy": r"Test Accuracy\s*:\s*([0-9.]+)",
        "Macro-precision": r"Test Macro Precision\s*:\s*([0-9.]+)",
        "Macro-recall": r"Test Macro Recall\s*:\s*([0-9.]+)",
        "Macro-F1": r"Test Macro F1\s*:\s*([0-9.]+)",
        "Weighted-F1": r"Test Weighted F1\s*:\s*([0-9.]+)",
    }

    metrics = {}

    for name, pattern in patterns.items():
        match = re.search(pattern, text)

        if match is None:
            raise ValueError(f"Could not find {name} in {path}")

        metrics[name] = float(match.group(1)) * 100

    return metrics


overall_rows = []
class_rows = []

for seed in SEEDS:
    run_dir = RUNS_DIR / f"eva02_base_448_split_{seed}"

    overall_path = run_dir / "final_test_metrics.txt"
    class_path = run_dir / "per_class_metrics_test.csv"

    if not overall_path.exists():
        raise FileNotFoundError(f"Missing file: {overall_path}")

    if not class_path.exists():
        raise FileNotFoundError(f"Missing file: {class_path}")

    overall_metrics = read_overall_metrics(overall_path)
    overall_metrics["Split seed"] = seed
    overall_rows.append(overall_metrics)

    class_df = pd.read_csv(class_path)

    required_columns = {"class", "precision", "recall", "f1"}

    if not required_columns.issubset(class_df.columns):
        raise ValueError(
            f"{class_path} is missing one or more required columns: "
            f"{required_columns}"
        )

    class_df = class_df[["class", "precision", "recall", "f1"]].copy()
    class_df[["precision", "recall", "f1"]] *= 100
    class_df["Split seed"] = seed
    class_rows.append(class_df)


# Table 1
overall_df = pd.DataFrame(overall_rows)

column_order = [
    "Split seed",
    "Accuracy",
    "Macro-precision",
    "Macro-recall",
    "Macro-F1",
    "Weighted-F1",
]

overall_df = overall_df[column_order]

metric_columns = column_order[1:]

mean_row = overall_df[metric_columns].mean()
sd_row = overall_df[metric_columns].std(ddof=1)

formatted_summary = {
    "Split seed": "Mean ± SD"
}

for metric in metric_columns:
    formatted_summary[metric] = (
        f"{mean_row[metric]:.2f} ± {sd_row[metric]:.2f}"
    )

table1 = overall_df.copy()

for metric in metric_columns:
    table1[metric] = table1[metric].map(lambda value: f"{value:.2f}")

table1 = pd.concat(
    [table1, pd.DataFrame([formatted_summary])],
    ignore_index=True
)

table1.to_csv("table_1_overall_results.csv", index=False)


# Table 2
all_classes = pd.concat(class_rows, ignore_index=True)

table2_stats = (
    all_classes
    .groupby("class", sort=False)[["precision", "recall", "f1"]]
    .agg(["mean", "std"])
)

table2 = pd.DataFrame({
    "Class": table2_stats.index,
    "Precision (%)": [
        f"{mean:.2f} ± {sd:.2f}"
        for mean, sd in zip(
            table2_stats[("precision", "mean")],
            table2_stats[("precision", "std")]
        )
    ],
    "Recall (%)": [
        f"{mean:.2f} ± {sd:.2f}"
        for mean, sd in zip(
            table2_stats[("recall", "mean")],
            table2_stats[("recall", "std")]
        )
    ],
    "F1-score (%)": [
        f"{mean:.2f} ± {sd:.2f}"
        for mean, sd in zip(
            table2_stats[("f1", "mean")],
            table2_stats[("f1", "std")]
        )
    ],
})

table2.to_csv("table_2_class_results.csv", index=False)

print("\nTable 1")
print(table1.to_string(index=False))

print("\nTable 2")
print(table2.to_string(index=False))

print("\nSaved:")
print("table_1_overall_results.csv")
print("table_2_class_results.csv")