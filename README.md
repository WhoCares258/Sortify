# TrashNet Waste Classification Using EVA02-Base

This project trains EVA02-Base on the TrashNet dataset using five independent global random splits.

## 1. Install Requirements

```bash
pip install -r requirements.txt
```

## Hardware Recommendation

A CUDA-enabled NVIDIA GPU is strongly recommended for training.

The project will run on CPU if CUDA is not available, but training will be significantly slower and may take many hours or even days depending on your hardware.

To verify that PyTorch detects your GPU, run:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

A return value of `True` indicates that CUDA is available and training will use the GPU.

## 2. Download TrashNet

1. Go to: https://huggingface.co/datasets/garythung/trashnet/tree/main
2. Open the **Files and versions** tab.
3. Download `dataset-original.zip`.
4. Extract the ZIP file.
5. Rename the extracted dataset folder to:

```text
trashnet
```

The folder structure should be:

```text
trashnet/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

## 3. Download EVA02 Pretrained Weights

Run:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', filename='model.safetensors', local_dir='pretrained/eva02_base_448')"
```

The model file should be saved as:

```text
pretrained/eva02_base_448/model.safetensors
```

## 4. Prepare the Five Dataset Splits

Run:

```bash
python prepare_5_trashnet_random_splits.py
```

This creates five global random 70/13/17 splits using seeds 0, 1, 2, 3 and 4.

## 5. Train All Five Splits

For a new clean run:

```bash
python run_5_eva02_splits.py --fresh
```

To resume after interruption:

```bash
python run_5_eva02_splits.py
```

Do not use `--fresh` when resuming.

## 6. Aggregate the Final Results

After all five runs are completed, run:

```bash
python aggregate_eva02_results.py
```

This creates:

```text
table_1_overall_results.csv
table_2_class_results.csv
```

## Main Files

```text
prepare_5_trashnet_random_splits.py
train_eva02_base_448_gpu.py
run_5_eva02_splits_resume_safe.py
aggregate_eva02_results.py
requirements.txt
README.md
```

## Main Output Folders

```text
trashnet-splits/
runs/
```

Each training run saves its model checkpoint, training log, final test metrics, confusion matrix and class-level metrics in a separate output folder.
