# prepare_5_trashnet_random_splits.py
#
# Creates 5 independent GLOBAL random 70/13/17 TrashNet splits.
# This is NOT stratified. It matches the global random split style you want
# for comparison with literature that only reports a 70/13/17 split ratio.
#
# Expected input:
# trashnet/
#   cardboard/
#   glass/
#   metal/
#   paper/
#   plastic/
#   trash/
#
# Output:
# trashnet-splits/
#   split_0/
#     train/
#     val/
#     test/
#   split_1/
#   split_2/
#   split_3/
#   split_4/
#
# Run:
# python prepare_5_trashnet_random_splits.py

import csv
import shutil
import random
from pathlib import Path
from collections import Counter


# ==========================
# Configuration
# ==========================

DATASET_DIR = Path("trashnet")
OUTPUT_ROOT = Path("trashnet-splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.13
TEST_RATIO = 0.17

SPLIT_SEEDS = [0, 1, 2, 3, 4]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REMOVE_OLD_OUTPUT = True


# ==========================
# Helper functions
# ==========================

def collect_images(dataset_dir, classes):
    all_images = []

    for cls in classes:
        class_dir = dataset_dir / cls

        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                all_images.append((img_path, cls))

    return all_images


def make_split(all_images, seed):
    rng = random.Random(seed)

    items = list(all_images)
    rng.shuffle(items)

    total_images = len(items)

    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)

    train_items = items[:train_end]
    val_items = items[train_end:val_end]
    test_items = items[val_end:]

    return {
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }


def create_split_folders(split_dir, classes):
    for split_name in ["train", "val", "test"]:
        for cls in classes:
            (split_dir / split_name / cls).mkdir(parents=True, exist_ok=True)


def copy_split_items(split_dir, splits):
    for split_name, items in splits.items():
        for img_path, cls in items:
            destination = split_dir / split_name / cls / img_path.name
            shutil.copy2(img_path, destination)


def write_manifest(split_dir, split_index, seed, splits):
    manifest_path = split_dir / "manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split_index",
            "seed",
            "split",
            "class",
            "filename",
            "source_path",
            "destination_path",
        ])

        for split_name, items in splits.items():
            for img_path, cls in items:
                destination = split_dir / split_name / cls / img_path.name
                writer.writerow([
                    split_index,
                    seed,
                    split_name,
                    cls,
                    img_path.name,
                    str(img_path),
                    str(destination),
                ])


def write_summary(split_dir, split_index, seed, splits, classes):
    summary_path = split_dir / "split_summary.csv"

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split_index",
            "seed",
            "split",
            "class",
            "count",
        ])

        for split_name, items in splits.items():
            counter = Counter(cls for _, cls in items)

            for cls in classes:
                writer.writerow([
                    split_index,
                    seed,
                    split_name,
                    cls,
                    counter[cls],
                ])


def print_split_summary(split_index, seed, split_dir, splits, classes):
    print("\n" + "=" * 70)
    print(f"Split {split_index} | Seed {seed}")
    print("=" * 70)
    print(f"Output: {split_dir}")

    for split_name, items in splits.items():
        counter = Counter(cls for _, cls in items)

        print(f"\n{split_name.upper()} split")
        print(f"Total: {len(items)}")

        for cls in classes:
            print(f"{cls:<12}: {counter[cls]:>3}")


def write_all_splits_summary(output_root, all_summary_rows):
    summary_path = output_root / "all_splits_summary.csv"

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split_index",
            "seed",
            "split",
            "class",
            "count",
        ])
        writer.writerows(all_summary_rows)

    return summary_path


# ==========================
# Main
# ==========================

def main():
    print("=" * 70)
    print("Preparing 5 Global Random TrashNet Splits")
    print("=" * 70)

    ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            f"Split ratios must sum to 1.0. Current sum: {ratio_sum}"
        )

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    classes = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])

    if len(classes) == 0:
        raise RuntimeError(f"No class folders found inside: {DATASET_DIR}")

    print(f"Dataset directory : {DATASET_DIR.resolve()}")
    print(f"Output root       : {OUTPUT_ROOT.resolve()}")
    print(f"Classes           : {classes}")
    print(f"Split seeds       : {SPLIT_SEEDS}")
    print(f"Split ratio       : {TRAIN_RATIO:.2f}/{VAL_RATIO:.2f}/{TEST_RATIO:.2f}")
    print("Split type        : global random, not stratified")

    all_images = collect_images(DATASET_DIR, classes)

    if len(all_images) == 0:
        raise RuntimeError("No image files found.")

    total_images = len(all_images)

    print(f"\nTotal images found: {total_images}")

    overall_counter = Counter(cls for _, cls in all_images)

    print("\nOriginal dataset class counts")
    print("-" * 70)

    for cls in classes:
        print(f"{cls:<12}: {overall_counter[cls]:>4}")

    if OUTPUT_ROOT.exists() and REMOVE_OLD_OUTPUT:
        print(f"\nRemoving old output folder: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []

    for split_index, seed in enumerate(SPLIT_SEEDS):
        split_dir = OUTPUT_ROOT / f"split_{split_index}"

        if split_dir.exists():
            shutil.rmtree(split_dir)

        create_split_folders(split_dir, classes)

        splits = make_split(all_images, seed)

        expected_total = sum(len(items) for items in splits.values())

        if expected_total != total_images:
            raise RuntimeError(
                f"Split {split_index} has {expected_total} images, "
                f"expected {total_images}."
            )

        copy_split_items(split_dir, splits)
        write_manifest(split_dir, split_index, seed, splits)
        write_summary(split_dir, split_index, seed, splits, classes)

        print_split_summary(split_index, seed, split_dir, splits, classes)

        for split_name, items in splits.items():
            counter = Counter(cls for _, cls in items)

            for cls in classes:
                all_summary_rows.append([
                    split_index,
                    seed,
                    split_name,
                    cls,
                    counter[cls],
                ])

    all_summary_path = write_all_splits_summary(OUTPUT_ROOT, all_summary_rows)

    print("\n" + "=" * 70)
    print("All splits created successfully.")
    print("=" * 70)
    print(f"Output root          : {OUTPUT_ROOT.resolve()}")
    print(f"Combined summary CSV : {all_summary_path.resolve()}")

    print("\nUse these DATA_DIR values in training:")
    for split_index in range(len(SPLIT_SEEDS)):
        print(f"DATA_DIR = Path(\"{OUTPUT_ROOT / f'split_{split_index}'}\")")


if __name__ == "__main__":
    main()