#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress future warnings from torch.load
warnings.filterwarnings("ignore", category=UserWarning)    # Suppress user warnings (scanpy, matplotlib, etc.)

import argparse
from pathlib import Path
import torch
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from torch.utils.data import DataLoader, Dataset
from typing import Optional

from pytorch_lightning import Trainer, Callback
from lightning.pytorch.loggers import CSVLogger

# segger imports (adjust paths/namespaces as needed)
from segger.training.train import LitSegger
from segger.training.segger_data_module import SeggerDataModule
from segger.data.utils import SpatialTranscriptomicsDataset
from segger.prediction.predict import predict_batch

import anndata
from matplotlib.colors import ListedColormap
import colorcet as cc

import holoviews as hv
hv.extension('bokeh')
# ----------------------------------------------------------------------------
# Global default output directory
# ----------------------------------------------------------------------------
OUTPUT = Path("/data/peer/riffled/segger/development_tools/latent_exploration/ERROR")

# ----------------------------------------------------------------------------
# Helper: Recursively move data to CUDA
# ----------------------------------------------------------------------------
def recursive_to_cuda(data):
    """
    Recursively moves all tensors in `data` to CUDA.
    """
    if torch.is_tensor(data):
        return data.to("cuda")
    elif isinstance(data, dict):
        return {k: recursive_to_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_to_cuda(v) for v in data]
    else:
        return data

# ----------------------------------------------------------------------------
# Custom DataModule for Full Batches but Consistent Inference Batch
# ----------------------------------------------------------------------------
class OneBatchDataModule(SeggerDataModule):
    """
    A DataModule that uses the full training dataset for training,
    but extracts one consistent batch for inference each epoch.
    """

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, train_shuffle=False, enrich_global_edges=False):
        super().__init__(data_dir, batch_size, num_workers, train_shuffle, enrich_global_edges)
        self.inference_batch = None

    def setup(self, stage=None):
        # Call the parent to load the dataset normally
        super().setup(stage=stage)
        # Grab one batch from train_dataloader to use for inference every epoch
        full_train_loader = super().train_dataloader()
        self.inference_batch = next(iter(full_train_loader)).to("cuda")
        print("Consistent inference batch extracted and moved to CUDA for inference.", flush=True)

    def get_inference_batch(self):
        return self.inference_batch

# ----------------------------------------------------------------------------
# Create AnnData from transcript assignments
# ----------------------------------------------------------------------------
def create_anndata(transcripts_parquet: str, transcript_assignments: pd.DataFrame):
    """
    Creates a Scanpy AnnData object from transcript assignments.
    """
    print(">>> Creating AnnData from assignments...", flush=True)

    transcripts_df = pd.read_parquet(transcripts_parquet)
    print(f"    Loaded transcripts dataframe: {transcripts_df.shape}", flush=True)

    # Ensure string type
    transcripts_df["transcript_id"] = transcripts_df["transcript_id"].astype(str)
    transcript_assignments["transcript_id"] = transcript_assignments["transcript_id"].astype(str)

    # Merge transcripts with assignments; drop UNASSIGNED
    merged_df = transcripts_df.merge(transcript_assignments, on="transcript_id", how="left")
    merged_df = merged_df.dropna(subset=['segger_cell_id'])
    print(f"    After dropping UNASSIGNED: {merged_df.shape[0]} transcripts remain", flush=True)

    # Build expression matrix (cell vs. feature_name)
    expr_matrix = merged_df.groupby(["cell_id", "feature_name"]).size().unstack(fill_value=0)

    # Average spatial coords per cell
    spatial_coords = merged_df.groupby("cell_id")[["x_location", "y_location", "z_location"]].mean()

    # Build obs and var dataframes
    obs = pd.DataFrame(index=expr_matrix.index)
    if "nucleus_overlap" in merged_df.columns:
        obs["nucleus_overlap"] = merged_df.groupby("cell_id")["nucleus_overlap"].mean()

    var = pd.DataFrame(index=expr_matrix.columns)

    # Create AnnData
    adata = anndata.AnnData(
        X=expr_matrix.values,
        obs=obs,
        var=var
    )
    # Add spatial info
    adata.obsm["spatial"] = spatial_coords.loc[adata.obs.index].values
    adata.obs["cell_id"] = adata.obs.index.values

    print(f"<<< Created AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}", flush=True)
    return adata

# ----------------------------------------------------------------------------
# Basic Scanpy pipeline
# ----------------------------------------------------------------------------
def run_scanpy_pipeline(adata, label_col=None, out_prefix="umap_sub_sampled"):
    """
    Basic Scanpy steps: neighbors, UMAP, Leiden + silhouette scores.
    """
    print(">>> Running Scanpy pipeline...", flush=True)
    # Optional sub-sample for speed
    if adata.n_obs > 50000:
        adata = adata[adata.obs.sample(50000).index]
    print(f"    Subsampled to {adata.n_obs} cells", flush=True)

    # Build graph & compute UMAP
    n_neighbors = max(1, min(adata.n_obs - 1, 15))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="leiden")

    # Plot UMAP with the provided label_col
    if label_col and label_col in adata.obs:
        print(f"    Plotting UMAP with {label_col}...", flush=True)
        sc.pl.umap(adata, color=label_col, show=False, save=f"_{out_prefix}_{label_col}.png")
    else:
        print("No label_col provided or not found in obs; skipping labeled UMAP plot.", flush=True)

    # Also plot Leiden clusters
    sc.pl.umap(adata, color="leiden", show=False, save=f"_{out_prefix}_leiden.png")

    # Compute silhouette scores
    from sklearn.metrics import silhouette_score
    sil_label, sil_leiden = None, None

    if label_col and label_col in adata.obs:
        n_unique_labels = len(adata.obs[label_col].unique())
        if n_unique_labels > 1:
            sil_label = silhouette_score(adata.obsm["X_umap"], adata.obs[label_col])

    if "leiden" in adata.obs:
        n_unique_leiden = len(adata.obs["leiden"].unique())
        if n_unique_leiden > 1:
            sil_leiden = silhouette_score(adata.obsm["X_umap"], adata.obs["leiden"])

    print(f"<<< Silhouette scores => {label_col}: {sil_label}, leiden: {sil_leiden}", flush=True)
    return sil_label, sil_leiden

# ----------------------------------------------------------------------------
# Add celltypist label
# ----------------------------------------------------------------------------
def add_celltypist_label(adata):
    """
    Adds a 'celltypist_label' metadata column to `adata` from a reference AnnData.
    Adjust path to your reference file accordingly.
    """
    ref = sc.read_h5ad("/data/peer/riffled/segger/development_tools/data/xenium_human_colon/h5ads/segger_embedding_processed.h5ad")
    ref.obs["10x_id"] = ref.obs["10x_id"].astype(str)
    adata.obs["cell_id"] = adata.obs["cell_id"].astype(str)

    common_cells = pd.Index(ref.obs["10x_id"]).intersection(pd.Index(adata.obs["cell_id"]))
    print(f"Found {len(common_cells)} matching cells for celltypist_label.")

    ref_obs_filtered = ref.obs.loc[ref.obs["10x_id"].isin(common_cells), ["10x_id", "celltypist_label"]].copy()

    # If multiple rows share the same 10x_id, pick one label
    def random_select(series):
        return series.sample(1).iloc[0]

    grouped_labels = ref_obs_filtered.groupby("10x_id")["celltypist_label"].agg(random_select)
    grouped_labels = grouped_labels.reset_index().rename(columns={"10x_id": "cell_id"}).set_index("cell_id")

    adata_filtered = adata[adata.obs["cell_id"].isin(grouped_labels.index)].copy()
    adata_filtered.obs = adata_filtered.obs.set_index("cell_id")

    adata_filtered.obs = adata_filtered.obs.join(grouped_labels)
    adata_filtered.obs = adata_filtered.obs.reset_index()
    print(f"Updated `adata` with celltypist_label metadata: {adata_filtered.shape}")
    return adata_filtered

# ----------------------------------------------------------------------------
# Callback for End-of-Epoch Inference & Plotting
# ----------------------------------------------------------------------------
class InferenceCallback(Callback):
    """
    Performs:
      1. UMAP on a consistent inference batch (transcript embeddings).
      2. Full inference + create AnnData + run Scanpy pipeline.
      3. Saves figures & h5ad each epoch (or at chosen intervals).
      4. Also includes a special transcript-level UMAP colored by celltypist label.
    """

    def __init__(self, dm: OneBatchDataModule, output_dir: Path, transcripts_parquet: str, interval=1):
        super().__init__()
        self.dm = dm
        self.output_dir = output_dir
        # IF output_dir is a string, convert to Path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_parquet = transcripts_parquet
        self.interval = interval
        # So that scanpy plots go into the same dir
        sc.settings.figdir = str(self.output_dir / "scanpy_plots")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.interval != 0:
            return  # Skip if not at the chosen interval

        print(f"\n[Callback] Starting inference at end of epoch {epoch}...", flush=True)

        # 1) UMAP on the consistent inference batch
        batch = self.dm.get_inference_batch()
        pl_module.to("cuda")
        with torch.no_grad():
            z = pl_module.get_z(batch)["tx"]
        z_np = z.detach().cpu().numpy()

        # If the batch has any transcript label in 'batch["tx"].label', we can use it
        # for quick coloring. Otherwise, everything is gray.
        labels = None
        if hasattr(batch["tx"], "label"):
            if torch.is_tensor(batch["tx"].label):
                labels = batch["tx"].label.detach().cpu().numpy()
            else:
                labels = np.array(batch["tx"].label)

        # We'll run the transcript-level UMAP plot every 5 epochs, for example.
        if epoch % 5 == 0:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
            embedding = reducer.fit_transform(z_np)

            # Merge transcript assignments with celltypist labels
            score_cut: float = .25,
            use_cc: bool = True,
            receptive_field: dict = {'k_bd': 4, 'dist_bd': 10, 'k_tx': 5, 'dist_tx': 3}
            assignments = predict_batch(pl_module, batch, score_cut, receptive_field, use_cc)

            adata = create_anndata(self.transcripts_parquet, assignments)
            adata = add_celltypist_label(adata)  # This adds "celltypist_label" to cell metadata

            # Merge transcript assignments with cell labels
            assignments = assignments.merge(adata.obs[['cell_id', 'celltypist_label']], 
                                            left_on="segger_cell_id", 
                                            right_on="cell_id", 
                                            how="left")
            assignments = assignments.drop(columns=["cell_id"])  # Remove duplicate column
            # Extract labels for coloring
            if pd.api.types.is_categorical_dtype(assignments["celltypist_label"]):
                # Add the 'Unknown' category if it's missing
                if "Unknown" not in assignments["celltypist_label"].cat.categories:
                    assignments["celltypist_label"] = assignments["celltypist_label"].cat.add_categories(["Unknown"])
                
                # Now it's safe to fill NAs with 'Unknown'
                assignments["celltypist_label"] = assignments["celltypist_label"].fillna("Unknown")

            fig, ax = plt.subplots(figsize=(6, 6))

            unique_labels = np.unique(labels) if labels is not None else ["Unknown"]
            N = len(unique_labels)

            # Map each label to an integer index
            label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}

            # Convert the label array to integer codes
            int_labels = np.array([label2idx[lbl] for lbl in labels])

            # Use colorcet’s glasbey_bw palette, taking as many colors as you have unique labels
            cmap = ListedColormap(cc.glasbey_bw[:N])

            fig, ax = plt.subplots(figsize=(6, 6))
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=int_labels,          # integer-coded labels
                cmap=cmap,             # discrete colormap
                s=8
            )

            ax.set_title(f"Epoch {epoch}")
            plt.tight_layout()
            save_path = self.output_dir / f"umap_epoch_{epoch}.png"
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"[Callback] Saved UMAP plot to {save_path}", flush=True)
            # Create a colormap from unique cell type labels
            transcript_labels = assignments["celltypist_label"]
            unique_labels = np.unique(transcript_labels)
            N = len(unique_labels)

            # Map each label to an integer index
            label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}

            # Convert the label array to integer codes
            int_labels = np.array([label2idx[lbl] for lbl in transcript_labels])

            # Use colorcet’s glasbey_bw palette, taking as many colors as you have unique labels
            cmap = ListedColormap(cc.glasbey_bw[:N])

            fig, ax = plt.subplots(figsize=(6, 6))
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=int_labels,          # integer-coded labels
                cmap=cmap,             # discrete colormap
                s=8
            )

            ax.set_title(f"Transcript UMAP Colored by Cell Type (Epoch {epoch})")
            cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)), label="Cell Type")
            cbar.ax.set_yticklabels(unique_labels, fontsize=6)  # Show string labels instead of numbers

            # Save the plot
            save_path = self.output_dir / f"umap_epoch_{epoch}_transcripts_by_celltype.png"
            plt.savefig(save_path, dpi=150)

        # 2) Full inference + Scanpy pipeline
        print("    Running full inference + Scanpy steps...", flush=True)
        # We do it every epoch
        score_cut = 0.25
        use_cc = True
        receptive_field = {'k_bd': 4, 'dist_bd': 10, 'k_tx': 5, 'dist_tx': 3}
        assignments = predict_batch(pl_module, batch, score_cut, receptive_field, use_cc)
        adata = create_anndata(self.transcripts_parquet, assignments)
        adata = add_celltypist_label(adata)
        sil_label, sil_leiden = run_scanpy_pipeline(adata, label_col="celltypist_label", out_prefix=f"epoch_{epoch}")

        print(
            f"    [Epoch {epoch}] silhouette(label)={sil_label}, silhouette(leiden)={sil_leiden}",
            flush=True
        )

        # 3) Save the AnnData
        h5ad_path = self.output_dir / f"adata_epoch_{epoch}.h5ad"
        adata.write(h5ad_path)
        print(f"[Callback] Wrote AnnData file: {h5ad_path}\n", flush=True)

# ----------------------------------------------------------------------------
# Main function to train Weighted Cov and No Cov models
# ----------------------------------------------------------------------------
def train_segger_experiments(
    cov_args,
    no_cov_args,
    output_dir=OUTPUT,
    enrich_global_edges=False
):
    """
    Trains two models (Weighted Cov and No Cov) with the given arguments.
    Saves intermediate inference plots and final h5ad outputs.
    """

    print(">>> Starting main training script (training on full batches with consistent inference batch)...", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Data for Cov model
    print(f"[Weighted Cov] Initializing OneBatchDataModule with data from: {cov_args.data_dir}", flush=True)
    dm_cov = OneBatchDataModule(
        data_dir=cov_args.data_dir,
        batch_size=cov_args.batch_size_train,
        num_workers=1,
        enrich_global_edges=enrich_global_edges
    )
    dm_cov.setup()  # loads dataset + extracts consistent batch
    # 2. LitSegger for Weighted Cov
    metadata = (
        ["tx", "bd"],
        [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")]
    )
    lit_segger_cov = LitSegger(
        num_tx_tokens=cov_args.num_tx_tokens,
        init_emb=cov_args.init_emb,
        hidden_channels=cov_args.hidden_channels,
        out_channels=cov_args.out_channels,
        heads=cov_args.heads,
        aggr=cov_args.aggr,
        num_mid_layers=cov_args.num_mid_layers,
        metadata=metadata,
        global_gene_cov_weight=cov_args.global_gene_cov_weight,
        bce_loss=cov_args.bce_loss,  
        triplet_loss= cov_args.triplet_loss 
        
    )

    # Callback for Cov model
    cov_callback = InferenceCallback(
        dm_cov,
        output_dir / "weighted_cov",
        transcripts_parquet=cov_args.transcripts_parquet,
        interval=1
    )

    trainer_cov = Trainer(
        accelerator=cov_args.accelerator,
        strategy=cov_args.strategy,
        precision=cov_args.precision,
        devices=cov_args.devices,
        max_epochs=cov_args.epochs,
        default_root_dir=cov_args.model_dir,
        logger=CSVLogger(cov_args.model_dir, name="cov_model_logs"),
        limit_val_batches=0,
        callbacks=[cov_callback],
    )

    print(f">>> [Weighted Cov] Starting training for {cov_args.epochs} epochs...", flush=True)
    trainer_cov.fit(lit_segger_cov, dm_cov)
    print(">>> [Weighted Cov] Done training!\n", flush=True)

    # 3. Prepare Data for No Cov model
    print(f"[No Cov] Initializing OneBatchDataModule with data from: {no_cov_args.data_dir}", flush=True)
    dm_no_cov = OneBatchDataModule(
        data_dir=no_cov_args.data_dir,
        batch_size=no_cov_args.batch_size_train,
        num_workers=1,
    )
    dm_no_cov.setup()

    lit_segger_no_cov = LitSegger(
        num_tx_tokens=no_cov_args.num_tx_tokens,
        init_emb=no_cov_args.init_emb,
        hidden_channels=no_cov_args.hidden_channels,
        out_channels=no_cov_args.out_channels,
        heads=no_cov_args.heads,
        aggr=no_cov_args.aggr,
        num_mid_layers=no_cov_args.num_mid_layers,
        metadata=metadata,
        global_gene_cov_weight=no_cov_args.global_gene_cov_weight,
        bce_loss=cov_args.bce_loss,  
        triplet_loss= cov_args.triplet_loss 
    )

    # Callback for No Cov model
    no_cov_callback = InferenceCallback(
        dm_no_cov,
        output_dir / "no_cov",
        transcripts_parquet=no_cov_args.transcripts_parquet,
        interval=1
    )

    trainer_no_cov = Trainer(
        accelerator=no_cov_args.accelerator,
        strategy=no_cov_args.strategy,
        precision=no_cov_args.precision,
        devices=no_cov_args.devices,
        max_epochs=no_cov_args.epochs,
        default_root_dir=no_cov_args.model_dir,
        logger=CSVLogger(no_cov_args.model_dir, name="no_cov_model_logs"),
        limit_val_batches=0,
        callbacks=[no_cov_callback],
    )

    print(f">>> [No Cov] Starting training for {no_cov_args.epochs} epochs...", flush=True)
    trainer_no_cov.fit(lit_segger_no_cov, dm_no_cov)
    print(">>> [No Cov] Done training!\n", flush=True)

    print(">>> All training complete! Each model was trained for the specified epochs.\n")


# ----------------------------------------------------------------------------
# Optional: main block for command-line usage
# ----------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Weighted Cov and No Cov Segger experiments.")
    parser.add_argument("--cov_data_dir", type=str, default="/path/to/cov/data", help="Cov dataset directory.")
    parser.add_argument("--no_cov_data_dir", type=str, default="/path/to/no_cov/data", help="No Cov dataset directory.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size_train", type=int, default=3, help="Training batch size.")
    parser.add_argument("--num_tx_tokens", type=int, default=500, help="Number of transcript tokens.")
    parser.add_argument("--model_dir", type=str, default="/path/to/model_dir", help="Model directory.")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT), help="Output directory for figures and h5ad files.")
    parser.add_argument("--transcripts_parquet", type=str, default="/path/to/transcripts.parquet", help="Path to transcripts parquet file.")
    parser.add_argument("--bce_loss", type=lambda x: x.lower() == "true", default=False, help="Use BCE loss.")
    parser.add_argument("--triplet_loss", type=lambda x: x.lower() == "true", default=True, help="Use triplet loss.")
    parser.add_argument("--enrich_global_edges", type=lambda x: x.lower() == "true", default=False, help="Enrich global edges.")
    parser.add_argument("--global_gene_cov_weight", type=float, default=1.0, help="Global gene cov weight.")

    args = parser.parse_args()

    cov_args = argparse.Namespace(
        data_dir=args.cov_data_dir,
        devices=1,
        epochs=args.epochs,
        batch_size_train=args.batch_size_train,
        batch_size_val=args.batch_size_train,
        num_tx_tokens=args.num_tx_tokens,
        init_emb=8,
        hidden_channels=32,
        out_channels=8,
        heads=4,
        aggr="sum",
        num_mid_layers=1,
        precision="16-mixed",
        global_gene_cov_weight=args.global_gene_cov_weight,
        model_dir=args.model_dir,
        accelerator="cuda",
        strategy="auto",
        transcripts_parquet= args.transcripts_parquet,
        bce_loss = args.bce_loss,
        triplet_loss = args.triplet_loss
    )

    no_cov_args = argparse.Namespace(
        data_dir=args.no_cov_data_dir,
        devices=1,
        epochs=args.epochs,
        batch_size_train=args.batch_size_train,
        batch_size_val=args.batch_size_train,
        num_tx_tokens=args.num_tx_tokens,
        init_emb=8,
        hidden_channels=32,
        out_channels=8,
        heads=4,
        aggr="sum",
        num_mid_layers=1,
        precision="16-mixed",
        global_gene_cov_weight=0,
        model_dir=args.model_dir,
        accelerator="cuda",
        strategy="auto",
        transcripts_parquet= args.transcripts_parquet,
        bce_loss = args.bce_loss,
        triplet_loss = args.triplet_loss
    )

    # Call the training function
    output_dir = Path(args.output_dir)
    train_segger_experiments(cov_args, no_cov_args, output_dir=output_dir, enrich_global_edges=args.enrich_global_edges)