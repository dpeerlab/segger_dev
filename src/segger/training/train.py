import os
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import F1Score
import lightning as L
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from torch_geometric.typing import Metadata
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean
import torch.nn as nn
from segger.models.segger_model import *
from segger.data.utils import SpatialTranscriptomicsDataset
from typing import Any, List, Tuple, Union
from pytorch_lightning import LightningModule
import inspect
import logging


class LitSegger(LightningModule):
    """
    LitSegger is a PyTorch Lightning module for training and validating the Segger model.

    Attributes
    ----------
    model : Segger
        The Segger model wrapped with PyTorch Geometric's to_hetero for heterogeneous graph support.
    validation_step_outputs : list
        A list to store outputs from the validation steps.
    criterion : torch.nn.Module
        The loss function used for training, specifically BCEWithLogitsLoss.
    """

    def __init__(self, **kwargs):
        """
        Initializes the LitSegger module with the given parameters.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for initializing the module. Specific parameters
            depend on whether the module is initialized with new parameters or components.
        """
        super().__init__()
        new_args = inspect.getfullargspec(self.from_new)[0][1:]
        cmp_args = inspect.getfullargspec(self.from_components)[0][1:]

        # Initialize with new parameters (ensure num_tx_tokens is passed here)
        if set(kwargs.keys()) == set(new_args):
            self.from_new(**kwargs)

        # Initialize with existing components
        elif set(kwargs.keys()) == set(cmp_args):
            self.from_components(**kwargs)

        # Handle invalid arguments
        else:
            raise ValueError(
                f"Supplied kwargs do not match either constructor. Should be one of '{new_args}' or '{cmp_args}'."
            )

        self.validation_step_outputs = []
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def from_new(self, num_tx_tokens: int, init_emb: int, hidden_channels: int, out_channels: int, heads: int, num_mid_layers: int, aggr: str, metadata: Union[Tuple, Metadata], global_gene_cov_weight: float = 0.01, bce_loss=True, triplet_loss=False):
        """
        Initializes the LitSegger module with new parameters.

        Parameters
        ----------
        num_tx_tokens : int
            Number of unique 'tx' tokens for embedding (this must be passed here).
        init_emb : int
            Initial embedding size.
        hidden_channels : int
            Number of hidden channels.
        out_channels : int
            Number of output channels.
        heads : int
            Number of attention heads.
        aggr : str
            Aggregation method for heterogeneous graph conversion.
        num_mid_layers: int
            Number of hidden layers (excluding first and last layers).
        metadata : Union[Tuple, Metadata]
            Metadata for heterogeneous graph structure.
        global_gene_cov_weight : float
            Weight for the global gene covariance penalty.
        bce_loss : bool
            Whether to use binary cross-entropy loss. (Turned off for some debugging)
        """
        # Create the Segger model (ensure num_tx_tokens is passed here)
        model = Segger(
            num_tx_tokens=num_tx_tokens,  # This is required and must be passed here
            init_emb=init_emb,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            num_mid_layers=num_mid_layers,
        )
        # Convert model to handle heterogeneous graphs
        model = to_hetero(model, metadata=metadata, aggr=aggr)
        self.model = model
        # Save hyperparameters
        self.save_hyperparameters()
        self.global_gene_cov_weight = global_gene_cov_weight
        self.bce_loss = bce_loss
        self.triplet_loss = triplet_loss

    def from_components(self, model: Segger):
        """
        Initializes the LitSegger module with existing Segger components.

        Parameters
        ----------
        model : Segger
            The Segger model to be used.
        """
        self.model = model

    def forward(self, batch: SpatialTranscriptomicsDataset) -> torch.Tensor:
        """
        Forward pass for the batch of data.

        Parameters
        ----------
        batch : SpatialTranscriptomicsDataset
            The batch of data, including node features and edge indices.

        Returns
        -------
        torch.Tensor
            The output of the model.
        """
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = torch.matmul(z['tx'], z['bd'].t())  # Example for bipartite graph
        return output
    

    def get_z(self, batch: SpatialTranscriptomicsDataset) -> torch.Tensor:
        """
        Get the latent representation of the model.
        Used for debugging.

        Parameters
        ----------
        batch : SpatialTranscriptomicsDataset
            The batch of data, including node features and edge indices.

        Returns
        -------
        torch.Tensor
            The latent representation of the model.
        """
        z = self.model(batch.x_dict, batch.edge_index_dict)
        return z
    
    def calc_triplet_loss(self, z, triplets, margin=0.2):
        """
        Computes triplet loss.

        Args:
            z (Tensor): Node embeddings.
            triplets (Tensor): Triplet indices [num_triplets, 3].
            margin (float): Triplet loss margin.

        Returns:
            Tensor: Triplet loss value.
        """
        if triplets is None or len(triplets) == 0:
            logging.warning("Triplets are empty or None. Returning zero loss.")
            return torch.tensor(0.0, device=z.device, dtype=torch.float32)

        # Convert list to tensor if needed
        if isinstance(triplets, list):
            triplets = torch.tensor(triplets, dtype=torch.long, device=z.device)

        # Ensure triplets are correctly shaped [num_triplets, 3]
        if triplets.dim() != 2 or triplets.shape[1] != 3:
            logging.error(f"Triplets tensor has incorrect shape {triplets.shape}, expected [num_triplets, 3].")
            return torch.tensor(0.0, device=z.device, dtype=torch.float32)

        # Extract anchor, positive, and negative indices
        anchor, positive, negative = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        # Compute pairwise distances
        pos_dist = F.pairwise_distance(z[anchor], z[positive], p=2)
        neg_dist = F.pairwise_distance(z[anchor], z[negative], p=2)

        # Compute Triplet Loss
        triplet_loss = F.relu(pos_dist - neg_dist + margin).mean()

        return triplet_loss


    def construct_triplets(self, batch, max_anchors=512, k_neg=50):
        """
        Constructs triplets for heterogeneous graphs, focusing on transcript ('tx') nodes.

        - Anchor (A): A transcript node.
        - Positive (P): A transcript node with the same label and closest in feature space.
        - Negative (N): A transcript node with a different label but closest in feature space.

        Returns:
            triplets (Tensor): Shape [3, num_triplets], containing indices of (A, P, N).
        """
        z = self.get_z(batch)
        # Obtain transcript embeddings from the model
        z_tx = z['tx']
        # Ensure embeddings are at least 2D
        if z_tx.dim() == 1:
            z_tx = z_tx.unsqueeze(-1)
        num_nodes = z_tx.size(0)

        # Retrieve transcript labels and move them to CPU.
        y = batch['tx'].label.cpu()

        # Compute pairwise Euclidean distances on CPU.
        with torch.no_grad():
            pairwise_dist = torch.cdist(z_tx.cpu(), z_tx.cpu(), p=2)
            pairwise_dist = pairwise_dist.cpu()

        all_indices = torch.arange(len(z_tx))
        if len(z_tx) > max_anchors:
            anchor_indices = all_indices[torch.randperm(len(z_tx))[:max_anchors]]
        else:
            anchor_indices = all_indices

        triplets = []
        for anchor_idx in anchor_indices:
            anchor_label = y[anchor_idx].item()

            # positives
            same_label_mask = (y == anchor_label)
            possible_pos = all_indices[same_label_mask & (all_indices != anchor_idx)]
            if len(possible_pos) == 0:
                continue
            # pick nearest positive
            pos_dist = F.pairwise_distance(z_tx[anchor_idx].unsqueeze(0), z_tx[possible_pos], p=2)
            positive_idx = possible_pos[torch.argmin(pos_dist)]

            # negatives (sample up to k_neg from all negatives)
            diff_label_mask = (y != anchor_label)
            possible_neg = all_indices[diff_label_mask]
            if len(possible_neg) > k_neg:
                subset_neg = possible_neg[torch.randperm(len(possible_neg))[:k_neg]]
            else:
                subset_neg = possible_neg
            
            neg_dist = F.pairwise_distance(z_tx[anchor_idx].unsqueeze(0), z_tx[subset_neg], p=2)
            negative_idx = subset_neg[torch.argmin(neg_dist)]

            triplets.append([anchor_idx.item(), positive_idx.item(), negative_idx.item()])

        if len(triplets) == 0:
            logging.warning("No valid triplets were created.")
            return None
        logging.warning("Returning Triplets")

        # # Convert list of triplets to a tensor and transpose so that shape is [3, num_triplets]
        # triplet_tensor = torch.tensor(triplets, dtype=torch.long).T

        # if triplet_tensor.shape[0] != 3:
        #     logging.error(f"Constructed triplet tensor has shape {triplet_tensor.shape}, expected [3, num_triplets].")
        #     return None

        return triplets



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Defines the training step.

        Parameters
        ----------
        batch : Any
            The batch of data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The loss value for the current training step.
        """
        # Forward pass to get the logits
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = torch.matmul(z['tx'], z['bd'].t())

        # Get edge labels and logits
        edge_label_index = batch['tx', 'belongs', 'bd'].edge_label_index
        out_values = output[edge_label_index[0], edge_label_index[1]]
        edge_label = batch['tx', 'belongs', 'bd'].edge_label
        
        # Compute binary cross-entropy loss with logits (no sigmoid here)
        bce_loss = self.criterion(out_values, edge_label)
        # Compute tile-level covaraince penalty
        cov_penalty = 0.0
        if not ('meta' in batch and 'global_gene_cov' in batch['meta']):
            logging.warning("Tile gene covariance matrix not found in batch metadata. Proceeding without gene covariance in loss.")
        
        if ('meta' in batch and 'global_gene_cov' in batch['meta']):
                logging.info("Computing gene covariance penalty")
                gene_idx = batch['tx'].label            # shape: [N_tx]
                num_genes = gene_idx.max().item() + 1   # total number of genes

                valid_mask = (gene_idx >= 0)           # Filter out any negative labels
                gene_idx = gene_idx[valid_mask]
                z_tx = z['tx']
                z_tx = z_tx[valid_mask] 

                # Compute gene-gene similarity matrix
                gene_embs = scatter_mean(z_tx, gene_idx, dim=0, dim_size=num_genes)
                gene_gene_sim = torch.matmul(gene_embs, gene_embs.t())  # [G, G]
                gene_gene_sim = gene_gene_sim / (1 + torch.abs(gene_gene_sim))
                # gene_gene_sim = torch.tanh(gene_gene_sim)

                gene_cov_df = batch['meta']['global_gene_cov'][0]  # your global cov as a pandas DF
                gene_cov_tensor = torch.tensor(gene_cov_df.values, dtype=torch.float32, device=z['tx'].device)
                gene_cov_tensor.requires_grad = True 
                cov_penalty = F.mse_loss(gene_gene_sim, gene_cov_tensor)

                s = 5.0  # Tunable Scaling factored
                scaled_penalty = torch.tensor(s * cov_penalty)
                transformed_cov_penalty = torch.sigmoid(scaled_penalty) / torch.sigmoid(torch.tensor(s))
        
        if self.triplet_loss:
            result_triplet_loss = (
                self.calc_triplet_loss(z['tx'], self.construct_triplets(batch))
            )
        else:
            result_triplet_loss = torch.tensor(0.0, device=z['tx'].device, dtype=torch.float32)

        if (self.bce_loss):
            loss = bce_loss + (transformed_cov_penalty * self.global_gene_cov_weight) + (result_triplet_loss)
        else:
            loss = cov_penalty * self.global_gene_cov_weight

        # Log the training loss
        # === 5) Logging ===
        self.log("train_loss", bce_loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("cov_penalty", cov_penalty, prog_bar=True, batch_size=batch.num_graphs)
        self.log("total_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("triplet_loss", result_triplet_loss, prog_bar=True, batch_size=batch.num_graphs)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Defines the validation step.

        Parameters
        ----------
        batch : Any
            The batch of data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The loss value for the current validation step.
        """
        # Forward pass to get the logits
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = torch.matmul(z['tx'], z['bd'].t())

        # Get edge labels and logits
        edge_label_index = batch['tx', 'belongs', 'bd'].edge_label_index
        out_values = output[edge_label_index[0], edge_label_index[1]]
        edge_label = batch['tx', 'belongs', 'bd'].edge_label
        
        # Compute binary cross-entropy loss with logits (no sigmoid here)
        loss = self.criterion(out_values, edge_label)
        
        # Apply sigmoid to logits for AUROC and F1 metrics
        out_values_prob = torch.sigmoid(out_values)

        # Compute metrics
        auroc = torchmetrics.AUROC(task="binary")
        auroc_res = auroc(out_values_prob, edge_label)
        
        f1 = F1Score(task="binary").to(self.device)
        f1_res = f1(out_values_prob, edge_label)
        
        # Log validation metrics
        self.log("validation_loss", loss, batch_size=batch.num_graphs)
        self.log("validation_auroc", auroc_res, prog_bar=True, batch_size=batch.num_graphs)
        self.log("validation_f1", f1_res, prog_bar=True, batch_size=batch.num_graphs)
        
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
