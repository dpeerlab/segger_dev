# Segger Training Pipeline

This repository provides a streamlined training pipeline for the Segger model, with built-in support for **consistent inference batch tracking**, **UMAP visualization**, and **cell-type-aware transcript embeddings**.

## Features
- Supports **Cov-weighted** and **No-Cov** experiments.
- Runs **UMAP visualization** on transcript embeddings at each epoch.
- Annotates transcript UMAP plots using **Celltypist labels**.
- **Exports h5ad** files for post-processing.
- Can be easily imported and used for experiments with different parameters.

---
## Running the Script

### 1. **Command-Line Execution**
You can run the script from the command line with customizable parameters:
```sh
python segger_train.py --cov_data_dir /path/to/cov/data \
                       --no_cov_data_dir /path/to/no_cov/data \
                       --epochs 50 \
                       --batch_size_train 3 \
                       --num_tx_tokens 500
                       ...
```

### 2. **Import as a Module**
Alternatively, you can import the training function and customize parameters programmatically:
```python
from segger_train import train_segger_experiments
import argparse

if __name__ == "__main__":
    cov_args = argparse.Namespace(
        data_dir="/my_path_for_cov_data",
        devices=1,
        epochs=100,
        batch_size_train=4,
        batch_size_val=4,
        num_tx_tokens=600,
        model_dir="/my_path_for_cov_models",
        accelerator="cuda",
    )

    no_cov_args = argparse.Namespace(
        data_dir="/my_path_for_no_cov_data",
        devices=1,
        epochs=100,
        batch_size_train=4,
        batch_size_val=4,
        num_tx_tokens=600,
        model_dir="/my_path_for_no_cov_models",
        accelerator="cuda",
    )

    output_dir = "/my_custom_output_dir"
    train_segger_experiments(cov_args, no_cov_args, output_dir=Path(output_dir))
```

## Outputs
The script generates the following outputs:
- **UMAP plots** (`umap_epoch_X.png` and `umap_epoch_X_transcripts_by_celltype.png`).
- **h5ad files** for each epoch (`adata_epoch_X.h5ad`).
- **Training logs** stored under `model_dir`.

## Customization
Modify `cov_args` and `no_cov_args` as needed to change:
- **Epoch count** (`epochs=50`)
- **Batch size** (`batch_size_train=3`)
- **Embedding dimensions** (`hidden_channels=32, out_channels=8`)
- **Covariate weighting** (`global_gene_cov_weight=1` for Cov, `0` for No-Cov)
...
