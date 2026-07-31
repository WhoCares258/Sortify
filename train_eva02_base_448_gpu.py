# train_eva02_base_448_gpu.py

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import timm
from timm.data import create_transform
from torchvision import datasets

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


# =========================================================
# Configuration
# =========================================================

DATA_DIR = Path("trashnet-splits/split_0")

OUTPUT_DIR = Path("runs/eva02_base_448_trashnet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
LOCAL_PRETRAINED_PATH = Path("pretrained/eva02_base_448/model.safetensors")

NUM_CLASSES = 6
IMAGE_SIZE = 448

# Accuracy-first but still fast.
# Effective batch is kept at 8, matching the safer fine-tuning behaviour.
# If BATCH_SIZE = 4 causes CUDA OOM, use BATCH_SIZE = 2 and ACCUMULATION_STEPS = 4.
# If still OOM, use BATCH_SIZE = 1 and ACCUMULATION_STEPS = 8.
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2

# Windows users: start with 2 or 4. If DataLoader hangs, set NUM_WORKERS = 0.
NUM_WORKERS = 4

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 30

PHASE1_LR = 1e-3
PHASE2_BASE_LR = 8e-6
PHASE2_HEAD_LR = 4e-5
PHASE2_WARMUP_EPOCHS = 2
LAYER_DECAY = 0.75

WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0

RANDOM_SEED = 42

USE_AMP = True
AMP_DTYPE = torch.float16
USE_CHANNELS_LAST = True

# True is safer for 448 input on laptop GPUs.
# Set False only if memory is enough; it will usually be faster.
USE_GRAD_CHECKPOINTING = True

AUTO_RESUME = True

BEST_MODEL_PATH = OUTPUT_DIR / "best_eva02_base_448.pth"
LAST_CHECKPOINT_PATH = OUTPUT_DIR / "last_checkpoint.pth"
LOG_PATH = OUTPUT_DIR / "training_log.csv"

FINAL_TEST_METRICS_PATH = OUTPUT_DIR / "final_test_metrics.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix_test.csv"
PER_CLASS_METRICS_PATH = OUTPUT_DIR / "per_class_metrics_test.csv"


# =========================================================
# EVA02 preprocessing
# =========================================================

DATA_CONFIG = {
    "input_size": (3, 448, 448),
    "mean": (0.48145466, 0.4578275, 0.40821073),
    "std": (0.26862954, 0.26130258, 0.27577711),
    "interpolation": "bicubic",
    "crop_pct": 1.0,
    "crop_mode": "squash",
}


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = RANDOM_SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# =========================================================
# Transforms
# =========================================================

def build_transforms():
    train_transform = create_transform(
        input_size=DATA_CONFIG["input_size"],
        is_training=True,
        mean=DATA_CONFIG["mean"],
        std=DATA_CONFIG["std"],
        interpolation=DATA_CONFIG["interpolation"],
        hflip=0.5,
        color_jitter=0.08,
        re_prob=0.10,
        re_mode="pixel",
        re_count=1,
    )

    eval_transform = create_transform(
        input_size=DATA_CONFIG["input_size"],
        is_training=False,
        mean=DATA_CONFIG["mean"],
        std=DATA_CONFIG["std"],
        interpolation=DATA_CONFIG["interpolation"],
        crop_pct=DATA_CONFIG["crop_pct"],
    )

    return train_transform, eval_transform


# =========================================================
# Model loading
# =========================================================

def create_model():
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    print(f"Created model: {MODEL_NAME}")

    if USE_GRAD_CHECKPOINTING and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(enable=True)
        print("Gradient checkpointing enabled.")

    return model


def clean_key_candidates(key):
    candidates = [key]

    for prefix in ["module.", "_orig_mod.", "model."]:
        if key.startswith(prefix):
            candidates.append(key[len(prefix):])

    return candidates


def load_local_pretrained_weights(model, local_path):
    if not local_path.exists():
        raise FileNotFoundError(
            f"Missing pretrained weights: {local_path}\n"
            f"Download with:\n"
            f"python -c \"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download(repo_id='timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', "
            f"filename='model.safetensors', local_dir='pretrained/eva02_base_448')\""
        )

    if local_path.suffix == ".safetensors":
        if load_safetensors is None:
            raise ImportError("Install safetensors using: pip install safetensors")
        pretrained_state = load_safetensors(str(local_path))
    else:
        pretrained_state = torch.load(local_path, map_location="cpu")
        if isinstance(pretrained_state, dict):
            pretrained_state = pretrained_state.get(
                "state_dict",
                pretrained_state.get("model", pretrained_state)
            )

    model_state = model.state_dict()
    filtered_state = {}

    loaded = 0
    skipped = 0

    for key, value in pretrained_state.items():
        matched = False

        for candidate_key in clean_key_candidates(key):
            if (
                candidate_key in model_state
                and value.shape == model_state[candidate_key].shape
            ):
                filtered_state[candidate_key] = value
                loaded += 1
                matched = True
                break

        if not matched:
            skipped += 1

    missing_keys, unexpected_keys = model.load_state_dict(
        filtered_state,
        strict=False
    )

    print(f"Loaded pretrained tensors : {loaded}")
    print(f"Skipped tensors           : {skipped}")
    print(f"Missing model keys        : {len(missing_keys)}")
    print(f"Unexpected keys           : {len(unexpected_keys)}")
    print("Classifier head skipped because TrashNet has 6 classes.")

    return model


def load_model_state_safely(model, state_dict):
    model_state = model.state_dict()
    clean_state = {}

    for key, value in state_dict.items():
        matched = False

        for candidate_key in clean_key_candidates(key):
            if (
                candidate_key in model_state
                and value.shape == model_state[candidate_key].shape
            ):
                clean_state[candidate_key] = value
                matched = True
                break

        if not matched:
            clean_state[key] = value

    model.load_state_dict(clean_state, strict=False)


# =========================================================
# Optimizer helpers
# =========================================================

def freeze_backbone_train_head(model):
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if "head" in name or "classifier" in name or "fc" in name:
                param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_layer_id(name, num_blocks):
    if name.startswith("head"):
        return num_blocks + 2

    if name.startswith("norm"):
        return num_blocks + 1

    if name.startswith("blocks."):
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1]) + 1

    return 0


def build_simple_param_groups(model, lr):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            param.ndim <= 1
            or name.endswith(".bias")
            or "pos_embed" in name
            or "cls_token" in name
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "lr": lr, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]


def build_layer_decay_param_groups(model):
    num_blocks = len(model.blocks) if hasattr(model, "blocks") else 12
    max_layer_id = num_blocks + 2
    groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        no_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or "pos_embed" in name
            or "cls_token" in name
        )

        wd = 0.0 if no_decay else WEIGHT_DECAY

        if name.startswith("head"):
            lr = PHASE2_HEAD_LR
        else:
            layer_id = get_layer_id(name, num_blocks)
            lr_scale = LAYER_DECAY ** (max_layer_id - layer_id)
            lr = PHASE2_BASE_LR * lr_scale

        key = (lr, wd)

        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr,
                "weight_decay": wd,
            }

        groups[key]["params"].append(param)

    return list(groups.values())


def create_scheduler(optimizer, epochs, warmup_epochs=0):
    if warmup_epochs <= 0:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )

    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.10,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs)
    )

    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )


# =========================================================
# Metrics
# =========================================================

def compute_metrics(labels, preds, num_classes):
    labels = torch.tensor(labels)
    preds = torch.tensor(preds)

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for true_label, pred_label in zip(labels, preds):
        confusion[true_label, pred_label] += 1

    tp = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)

    precision = tp.float() / predicted.clamp(min=1).float()
    recall = tp.float() / support.clamp(min=1).float()
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

    accuracy = tp.sum().float() / confusion.sum().clamp(min=1).float()
    weighted_f1 = (f1 * support.float()).sum() / support.sum().clamp(min=1).float()

    return {
        "accuracy": accuracy.item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "weighted_f1": weighted_f1.item(),
        "confusion_matrix": confusion,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_accuracy": recall.tolist(),
        "support": support.tolist(),
    }


def autocast_context(device, amp_enabled):
    if device.type == "cuda":
        try:
            return torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
                dtype=AMP_DTYPE
            )
        except TypeError:
            return torch.cuda.amp.autocast(enabled=amp_enabled)

    return torch.amp.autocast(
        "cpu",
        enabled=False
    )


def move_batch_to_device(images, labels, device):
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    if USE_CHANNELS_LAST and device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)

    return images, labels


# =========================================================
# Train and evaluate
# =========================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, amp_enabled):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="Training", leave=False)

    for batch_idx, (images, labels) in enumerate(progress, start=1):
        images, labels = move_batch_to_device(images, labels, device)

        with autocast_context(device, amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaled_loss = loss / ACCUMULATION_STEPS
        scaler.scale(scaled_loss).backward()

        should_step = (
            batch_idx % ACCUMULATION_STEPS == 0
            or batch_idx == len(loader)
        )

        if should_step:
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, NUM_CLASSES)

    return epoch_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_enabled, split_name):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress = tqdm(loader, desc=split_name, leave=False)

    for images, labels in progress:
        images, labels = move_batch_to_device(images, labels, device)

        with autocast_context(device, amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, NUM_CLASSES)

    return epoch_loss, metrics


# =========================================================
# Saving
# =========================================================

def write_training_log_header():
    if LOG_PATH.exists():
        return

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase",
            "epoch",
            "train_loss",
            "train_acc",
            "train_macro_f1",
            "val_loss",
            "val_acc",
            "val_macro_precision",
            "val_macro_recall",
            "val_macro_f1",
            "val_weighted_f1",
            "lr",
        ])


def append_training_log(
    phase,
    epoch,
    train_loss,
    train_metrics,
    val_loss,
    val_metrics,
    lr
):
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            phase,
            epoch,
            train_loss,
            train_metrics["accuracy"],
            train_metrics["macro_f1"],
            val_loss,
            val_metrics["accuracy"],
            val_metrics["macro_precision"],
            val_metrics["macro_recall"],
            val_metrics["macro_f1"],
            val_metrics["weighted_f1"],
            lr,
        ])


def save_best_checkpoint(
    model,
    epoch,
    phase_index,
    phase_name,
    val_loss,
    val_metrics,
    best_val_accuracy,
    best_macro_f1,
    class_names
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "phase_index": phase_index,
        "phase": phase_name,
        "best_val_accuracy": best_val_accuracy,
        "best_macro_f1": best_macro_f1,
        "val_loss": val_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_precision": val_metrics["macro_precision"],
        "val_macro_recall": val_metrics["macro_recall"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "class_names": class_names,
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "data_config": DATA_CONFIG,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "effective_batch_size": BATCH_SIZE * ACCUMULATION_STEPS,
        "preprocessing": "timm create_transform with EVA02 mean/std and bicubic interpolation",
    }

    torch.save(checkpoint, BEST_MODEL_PATH)


def save_last_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    phase_index,
    phase_name,
    best_val_accuracy,
    best_macro_f1,
    class_names
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "phase_index": phase_index,
        "phase": phase_name,
        "best_val_accuracy": best_val_accuracy,
        "best_macro_f1": best_macro_f1,
        "class_names": class_names,
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "data_config": DATA_CONFIG,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
    }

    torch.save(checkpoint, LAST_CHECKPOINT_PATH)


def save_confusion_matrix(confusion, path, class_names):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + class_names)

        for i, row in enumerate(confusion.tolist()):
            writer.writerow([class_names[i]] + row)


def save_per_class_metrics(metrics, path, class_names):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class",
            "support",
            "precision",
            "recall",
            "f1",
            "per_class_accuracy",
        ])

        for i, cls in enumerate(class_names):
            writer.writerow([
                cls,
                metrics["support"][i],
                metrics["per_class_precision"][i],
                metrics["per_class_recall"][i],
                metrics["per_class_f1"][i],
                metrics["per_class_accuracy"][i],
            ])


def should_save_best(val_metrics, best_val_accuracy, best_macro_f1):
    return (
        val_metrics["accuracy"] >= best_val_accuracy
    )


# =========================================================
# Main
# =========================================================

def main():
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("=" * 70)
    print("Training EVA02-Base 448 on TrashNet")
    print("=" * 70)
    print(f"Device              : {device}")
    print(f"GPU                 : {torch.cuda.get_device_name(0)}")
    print(f"Dataset             : {DATA_DIR}")
    print(f"Output directory    : {OUTPUT_DIR}")
    print(f"Model               : {MODEL_NAME}")
    print(f"Image size          : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Accumulation steps  : {ACCUMULATION_STEPS}")
    print(f"Effective batch     : {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"DataLoader workers  : {NUM_WORKERS}")
    print("Preprocessing        : timm/PIL create_transform")
    print(f"Gradient checkpoint : {USE_GRAD_CHECKPOINTING}")
    print(f"AMP                 : {USE_AMP}")
    print(f"Channels-last       : {USE_CHANNELS_LAST}")

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"

    for folder in [train_dir, val_dir, test_dir]:
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

    train_transform, eval_transform = build_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    class_names = train_dataset.classes

    print("\nClasses:")
    for idx, cls in enumerate(class_names):
        print(f"{idx}: {cls}")

    print("\nDataset size:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val  : {len(val_dataset)}")
    print(f"Test : {len(test_dataset)}")

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "worker_init_fn": seed_worker if NUM_WORKERS > 0 else None,
        "persistent_workers": True if NUM_WORKERS > 0 else False,
    }

    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )

    model = create_model()
    model = load_local_pretrained_weights(model, LOCAL_PRETRAINED_PATH)
    model = model.to(device)

    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    amp_enabled = USE_AMP and device.type == "cuda"

    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    else:
        try:
            scaler = torch.amp.GradScaler("cpu", enabled=False)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=False)

    phases = [
        {
            "name": "phase1_head_only",
            "epochs": PHASE1_EPOCHS,
            "freeze": True,
            "warmup_epochs": 0,
        },
        {
            "name": "phase2_full_finetune",
            "epochs": PHASE2_EPOCHS,
            "freeze": False,
            "warmup_epochs": PHASE2_WARMUP_EPOCHS,
        },
    ]

    resume_checkpoint = None
    start_phase_index = 0
    resume_epoch = 0

    best_val_accuracy = 0.0
    best_macro_f1 = 0.0

    if AUTO_RESUME and LAST_CHECKPOINT_PATH.exists():
        print("\nFound last checkpoint. Resuming training.")
        resume_checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device)
        load_model_state_safely(model, resume_checkpoint["model_state_dict"])

        start_phase_index = resume_checkpoint["phase_index"]
        resume_epoch = resume_checkpoint["epoch"]
        best_val_accuracy = resume_checkpoint.get("best_val_accuracy", 0.0)
        best_macro_f1 = resume_checkpoint.get("best_macro_f1", 0.0)

        print(f"Resume phase      : {resume_checkpoint['phase']}")
        print(f"Resume epoch      : {resume_epoch}")
        print(f"Best Val Accuracy : {best_val_accuracy:.4f}")
        print(f"Best Macro-F1     : {best_macro_f1:.4f}")
    else:
        print("\nStarting new training run.")

    write_training_log_header()

    for phase_index, phase in enumerate(phases):
        if phase_index < start_phase_index:
            continue

        phase_name = phase["name"]
        phase_epochs = phase["epochs"]

        print("\n" + "=" * 70)
        print(phase_name)
        print("=" * 70)

        if phase["freeze"]:
            freeze_backbone_train_head(model)
            param_groups = build_simple_param_groups(model, PHASE1_LR)
            print(f"Learning rate       : {PHASE1_LR}")
        else:
            unfreeze_all(model)
            param_groups = build_layer_decay_param_groups(model)
            print(f"Base LR             : {PHASE2_BASE_LR}")
            print(f"Head LR             : {PHASE2_HEAD_LR}")
            print(f"Layer decay         : {LAYER_DECAY}")

        print(f"Trainable parameters: {count_trainable_params(model):,}")
        print(f"Epochs              : {phase_epochs}")

        optimizer = optim.AdamW(param_groups)

        scheduler = create_scheduler(
            optimizer,
            epochs=phase_epochs,
            warmup_epochs=phase["warmup_epochs"]
        )

        start_epoch = 1

        if resume_checkpoint is not None and phase_index == start_phase_index:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])

            start_epoch = resume_epoch + 1
            print(f"Continuing from epoch {start_epoch}")

        for epoch in range(start_epoch, phase_epochs + 1):
            start_time = time.time()

            train_loss, train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                device,
                amp_enabled
            )

            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled,
                split_name="Validation"
            )

            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - start_time

            print(
                f"{phase_name} | "
                f"Epoch [{epoch}/{phase_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Train F1: {train_metrics['macro_f1']:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
                f"LR: {current_lr:.8f} | "
                f"Time: {epoch_time:.1f}s"
            )

            append_training_log(
                phase_name,
                epoch,
                train_loss,
                train_metrics,
                val_loss,
                val_metrics,
                current_lr
            )

            if should_save_best(val_metrics, best_val_accuracy):
                best_val_accuracy = val_metrics["accuracy"]
                best_macro_f1 = val_metrics["macro_f1"]

                save_best_checkpoint(
                    model,
                    epoch,
                    phase_index,
                    phase_name,
                    val_loss,
                    val_metrics,
                    best_val_accuracy,
                    best_macro_f1,
                    class_names
                )

                print(
                    f"Best model saved. "
                    f"Val Accuracy: {best_val_accuracy:.4f} | "
                    f"Val Macro-F1: {best_macro_f1:.4f}"
                )

            save_last_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                phase_index,
                phase_name,
                best_val_accuracy,
                best_macro_f1,
                class_names
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        resume_checkpoint = None
        resume_epoch = 0

    print("\n" + "=" * 70)
    print("Final Test Evaluation")
    print("=" * 70)

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError("No best model checkpoint found.")

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    load_model_state_safely(model, checkpoint["model_state_dict"])

    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        amp_enabled,
        split_name="Test"
    )

    print(f"Best Val Accuracy    : {checkpoint.get('best_val_accuracy', 0.0):.4f}")
    print(f"Best Val Macro-F1    : {checkpoint.get('best_macro_f1', 0.0):.4f}")
    print(f"Test Loss            : {test_loss:.4f}")
    print(f"Test Accuracy        : {test_metrics['accuracy']:.4f}")
    print(f"Test Macro Precision : {test_metrics['macro_precision']:.4f}")
    print(f"Test Macro Recall    : {test_metrics['macro_recall']:.4f}")
    print(f"Test Macro F1        : {test_metrics['macro_f1']:.4f}")
    print(f"Test Weighted F1     : {test_metrics['weighted_f1']:.4f}")

    print("\nPer-class accuracy:")
    for i, cls in enumerate(class_names):
        print(
            f"{cls:<12}: "
            f"{test_metrics['per_class_accuracy'][i]:.4f} "
            f"({test_metrics['support'][i]} images)"
        )

    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        CONFUSION_MATRIX_PATH,
        class_names
    )

    save_per_class_metrics(
        test_metrics,
        PER_CLASS_METRICS_PATH,
        class_names
    )

    with open(FINAL_TEST_METRICS_PATH, "w") as f:
        f.write(f"Model                : {MODEL_NAME}\n")
        f.write(f"Image size           : {IMAGE_SIZE}\n")
        f.write(f"Batch size           : {BATCH_SIZE}\n")
        f.write(f"Accumulation steps   : {ACCUMULATION_STEPS}\n")
        f.write(f"Effective batch      : {BATCH_SIZE * ACCUMULATION_STEPS}\n")
        f.write("Preprocessing         : timm/PIL create_transform\n")
        f.write(f"Best Val Accuracy    : {checkpoint.get('best_val_accuracy', 0.0):.4f}\n")
        f.write(f"Best Val Macro-F1    : {checkpoint.get('best_macro_f1', 0.0):.4f}\n")
        f.write(f"Test Loss            : {test_loss:.4f}\n")
        f.write(f"Test Accuracy        : {test_metrics['accuracy']:.4f}\n")
        f.write(f"Test Macro Precision : {test_metrics['macro_precision']:.4f}\n")
        f.write(f"Test Macro Recall    : {test_metrics['macro_recall']:.4f}\n")
        f.write(f"Test Macro F1        : {test_metrics['macro_f1']:.4f}\n")
        f.write(f"Test Weighted F1     : {test_metrics['weighted_f1']:.4f}\n")

    print("\nFiles saved:")
    print(f"Best model        : {BEST_MODEL_PATH}")
    print(f"Last checkpoint   : {LAST_CHECKPOINT_PATH}")
    print(f"Training log      : {LOG_PATH}")
    print(f"Final metrics     : {FINAL_TEST_METRICS_PATH}")
    print(f"Confusion matrix  : {CONFUSION_MATRIX_PATH}")
    print(f"Per-class metrics : {PER_CLASS_METRICS_PATH}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
