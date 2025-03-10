import os
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import F1Score, AUROC
import lightning as L
from torch_geometric.loader import DataLoader
from torch_geometric.typing import Metadata
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
from segger.models.segger_model import *
from segger.data.utils import SpatialTranscriptomicsDataset
from typing import Any, List, Tuple, Union
from pytorch_lightning import LightningModule
import inspect

class LitSegger(LightningModule):
    """
    LitSegger is a PyTorch Lightning module for training and validating the 
    Segger model.
    """

    def __init__(
        self,
        model: Segger,
        learning_rate: float = 1e-3,
    ):
        """
        Initialize the Segger training module.

        Parameters
        ----------
        model : Segger
            The Segger model to be trained.
        learning_rate : float, optional
            Learning rate for the optimizer. Default is 1e-3.
        """
        super().__init__()
        # Set model and store initialization parameters for reproducibility
        self.model = model
        self.save_hyperparameters(self.model.hparams)

        # Other setup
        self.learning_rate = learning_rate
        self.validation_step_outputs = []

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
        edge_label_index = batch["tx", "belongs", "bd"].edge_label_index
        output = self.model.decode(z, edge_label_index)
        return output

    def get_bce_loss(self, batch: Any) -> torch.Tensor:
        """
        Compute the binary cross-entropy loss for the given batch.

        Parameters
        ----------
        batch : Any
            The input batch containing node features, edge indices,
            and edge labels.

        Returns
        -------
        torch.Tensor
            Computed binary cross-entropy loss.
        """
        # Get edge labels
        edge_label_index = batch['tx', 'belongs', 'bd'].edge_label_index
        edge_label = batch['tx', 'belongs', 'bd'].edge_label

        # Forward pass to get the logits
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = self.model.decode(z, edge_label_index)

        # Compute binary cross-entropy loss with logits (sigmoid in function)
        loss = torch.nn.BCEWithLogitsLoss(output, edge_label)

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        Parameters
        ----------
        batch : Any
            The input batch containing node features, edge indices, 
            and edge labels.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Computed training loss for the current batch.
        """
        # Get loss
        loss = self.get_bce_loss(batch)

        # Log the training loss
        self.log(
            "training_loss",
            loss,
            prog_bar=True,
            batch_size=batch.num_graphs
        )

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
        # Get loss
        loss = self.get_bce_loss(batch)

        # Apply sigmoid to logits for AUROC and F1 metrics
        edge_label = batch['tx', 'belongs', 'bd'].edge_label
        edge_label_index = batch['tx', 'belongs', 'bd'].edge_label_index
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = self.model.decode(z, edge_label_index)
        probs = torch.sigmoid(output)

        # Compute metrics
        auroc = AUROC(task="binary").to(self.device)(probs, edge_label)
        f1_score = F1Score(task="binary").to(self.device)(probs, edge_label)

        # Log validation metrics
        shared_kwargs = dict(prog_bar=True, batch_size=batch.num_graphs)
        self.log("validation_loss", loss, **shared_kwargs)
        self.log("validation_auroc", auroc, **shared_kwargs)
        self.log("validation_f1", f1_score, **shared_kwargs)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
