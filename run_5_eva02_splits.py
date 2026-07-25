# run_5_eva02_splits_resume_safe.py
#
# Resume-safe runner for training EVA02 on 5 TrashNet random splits.
#
# Workflow:
# 1. python prepare_5_trashnet_random_splits.py
# 2. python run_5_eva02_splits_resume_safe.py --fresh     # first full clean run
#
# If the terminal/PC stops accidentally:
# python run_5_eva02_splits_resume_safe.py
#
# Important:
# - Do NOT use --fresh when resuming.
# - Completed splits are skipped automatically.
# - Interrupted splits resume from last_checkpoint.pth if AUTO_RESUME=True
#   in train_eva02_base_448_gpu.py.

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


SOURCE_SCRIPT = Path("train_eva02_base_448_gpu.py")

SPLIT_ROOT = Path("trashnet-splits")
SPLIT_INDICES = [0, 1, 2, 3, 4]

TRAINING_SEED = 42

TEMP_DIR = Path("_split_run_scripts")
SUMMARY_CSV = Path("runs/eva02_5split_summary.csv")


def replace_constant(text, name, value):
    pattern = rf"^{name}\s*=\s*.*$"
    replacement = f"{name} = {value}"

    new_text, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.MULTILINE
    )

    if count != 1:
        raise RuntimeError(f"Could not replace constant: {name}")

    return new_text


def make_split_script(split_index):
    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_SCRIPT}. Put this runner in the same folder "
            f"as your training script."
        )

    split_dir = SPLIT_ROOT / f"split_{split_index}"

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Missing split folder: {split_dir}\n"
            f"Run prepare_5_trashnet_random_splits.py first."
        )

    text = SOURCE_SCRIPT.read_text(encoding="utf-8")

    data_dir_value = f'Path("{split_dir.as_posix()}")'
    output_dir_value = f'Path("runs/eva02_base_448_split_{split_index}")'

    text = replace_constant(text, "DATA_DIR", data_dir_value)
    text = replace_constant(text, "OUTPUT_DIR", output_dir_value)
    text = replace_constant(text, "RANDOM_SEED", TRAINING_SEED)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    split_script = TEMP_DIR / f"train_eva02_split_{split_index}.py"
    split_script.write_text(text, encoding="utf-8")

    output_dir = Path(f"runs/eva02_base_448_split_{split_index}")

    return split_script, split_dir, output_dir


def read_metric_file(split_index, split_dir, output_dir, status="completed"):
    metrics_path = output_dir / "final_test_metrics.txt"

    result = {
        "split_index": split_index,
        "split_dir": str(split_dir),
        "training_seed": TRAINING_SEED,
        "status": "missing_metrics",
        "best_val_accuracy": "",
        "best_val_macro_f1": "",
        "test_accuracy": "",
        "test_macro_precision": "",
        "test_macro_recall": "",
        "test_macro_f1": "",
        "test_weighted_f1": "",
        "output_dir": str(output_dir),
    }

    if not metrics_path.exists():
        return result

    text = metrics_path.read_text(encoding="utf-8", errors="ignore")

    keys = {
        "Best Val Accuracy": "best_val_accuracy",
        "Best Val Macro-F1": "best_val_macro_f1",
        "Test Accuracy": "test_accuracy",
        "Test Macro Precision": "test_macro_precision",
        "Test Macro Recall": "test_macro_recall",
        "Test Macro F1": "test_macro_f1",
        "Test Weighted F1": "test_weighted_f1",
    }

    for line in text.splitlines():
        if ":" not in line:
            continue

        left, right = line.split(":", 1)
        left = left.strip()
        right = right.strip()

        if left in keys:
            result[keys[left]] = right

    result["status"] = status
    return result


def write_summary(rows):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split_index",
        "status",
        "training_seed",
        "best_val_accuracy",
        "best_val_macro_f1",
        "test_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "test_weighted_f1",
        "split_dir",
        "output_dir",
    ]

    rows = sorted(rows, key=lambda row: int(row["split_index"]))

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"\nCurrent summary saved to: {SUMMARY_CSV}")


def calculate_mean_sd(rows):
    completed = [
        row for row in rows
        if row["status"] in {"completed", "skipped_completed"}
        and row.get("test_accuracy", "") != ""
    ]

    metrics = [
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
    ]

    print("\n" + "=" * 80)
    print("Mean ± SD across completed splits")
    print("=" * 80)

    if len(completed) == 0:
        print("No completed runs yet.")
        return

    for metric in metrics:
        values = []

        for row in completed:
            value = row.get(metric, "")

            if value == "":
                continue

            values.append(float(value))

        if len(values) == 0:
            print(f"{metric}: no values")
            continue

        mean = sum(values) / len(values)

        if len(values) > 1:
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            sd = variance ** 0.5
        else:
            sd = 0.0

        print(f"{metric:<18}: {mean:.4f} ± {sd:.4f}  n={len(values)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete each split output folder before running."
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Rerun splits even if final_test_metrics.txt already exists."
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop remaining runs if one split fails."
    )
    args = parser.parse_args()

    rows = []

    print("=" * 80)
    print("Running EVA02 on 5 TrashNet random splits")
    print("=" * 80)
    print(f"Training script : {SOURCE_SCRIPT}")
    print(f"Split root      : {SPLIT_ROOT}")
    print(f"Splits          : {SPLIT_INDICES}")
    print(f"Training seed   : {TRAINING_SEED}")
    print(f"Fresh run       : {args.fresh}")
    print(f"Rerun completed : {args.rerun_completed}")

    for split_index in SPLIT_INDICES:
        print("\n" + "=" * 80)
        print(f"Split {split_index}")
        print("=" * 80)

        split_script, split_dir, output_dir = make_split_script(split_index)

        if args.fresh and output_dir.exists():
            print(f"Deleting old output folder: {output_dir}")
            shutil.rmtree(output_dir)

        metrics_path = output_dir / "final_test_metrics.txt"

        if (
            metrics_path.exists()
            and not args.fresh
            and not args.rerun_completed
        ):
            print(f"Already completed. Skipping: {output_dir}")
            row = read_metric_file(
                split_index,
                split_dir,
                output_dir,
                status="skipped_completed"
            )
            rows.append(row)
            write_summary(rows)
            calculate_mean_sd(rows)
            continue

        last_checkpoint = output_dir / "last_checkpoint.pth"

        if last_checkpoint.exists() and not args.fresh:
            print(f"Found checkpoint. Training script should resume from: {last_checkpoint}")
        else:
            print("No resume checkpoint found for this split. Starting normally.")

        command = [sys.executable, str(split_script)]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            status = f"failed_exit_{exc.returncode}"
            print(f"Split {split_index} failed with exit code {exc.returncode}")

            row = {
                "split_index": split_index,
                "split_dir": str(split_dir),
                "training_seed": TRAINING_SEED,
                "status": status,
                "best_val_accuracy": "",
                "best_val_macro_f1": "",
                "test_accuracy": "",
                "test_macro_precision": "",
                "test_macro_recall": "",
                "test_macro_f1": "",
                "test_weighted_f1": "",
                "output_dir": str(output_dir),
            }

            rows.append(row)
            write_summary(rows)
            calculate_mean_sd(rows)

            if args.stop_on_error:
                raise

            continue

        row = read_metric_file(
            split_index,
            split_dir,
            output_dir,
            status="completed"
        )
        rows.append(row)

        write_summary(rows)
        calculate_mean_sd(rows)

    print("\n" + "=" * 80)
    print("All split runs finished or skipped.")
    print("=" * 80)
    print(f"Final summary CSV: {SUMMARY_CSV}")
    print("\nReport mean ± SD using all completed split results.")
    print("Do not remove weak splits.")


if __name__ == "__main__":
    main()