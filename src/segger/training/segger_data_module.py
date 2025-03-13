from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader
import os
from pathlib import Path
from segger.data.parquet.pyg_dataset import STPyGDataset
import logging
import torch
from torch_geometric.data import Batch


# TODO: Add documentation
class SeggerDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: os.PathLike,
        batch_size: int = 4,
        num_workers: int = 1,
        train_shuffle = True,
        inject_synthetic_extremes = False,
        enrich_global_edges = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle
        self.inject_synthetic_extremes = inject_synthetic_extremes
        self.enrich_global_edges = enrich_global_edges

    # TODO: Add documentation
    def setup(self, stage=None):
        self.train = STPyGDataset(root=self.data_dir / 'train_tiles')
        self.test = STPyGDataset(root=self.data_dir / 'test_tiles')
        self.val = STPyGDataset(root=self.data_dir / 'val_tiles')
        self.loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        self.correlation_threshold = 0.9

    
    def inject_synthetic_extreme_transcripts(self, batch: Batch):
        """
        Inject synthetic nodes based on extreme values in 'global_gene_cov'.
        These synthetic nodes are mapped into Segger's feature space via embeddings.
        """
        if 'global_gene_cov' not in batch.meta:
            logging.warning("global_gene_cov not found in batch['meta'], skipping synthetic injection.")
            return batch
        num_nodes, num_genes = batch.meta['global_gene_cov'].shape
        num_synthetic = int(self.synthetic_fraction * num_nodes)
        if num_synthetic == 0:
            return batch  # Skip if no synthetic nodes should be added
        # Identify extreme values (high variance)
        gene_variance = torch.var(batch.meta['global_gene_cov'], dim=0)
        top_genes = torch.argsort(gene_variance, descending=True)[:num_synthetic]
        # Sample synthetic nodes from extreme genes
        synthetic_nodes = batch.meta['global_gene_cov'][:, top_genes]  # Extract high-variance features
        synthetic_nodes += torch.randn_like(synthetic_nodes) * 0.1  # Add slight noise
        # Assign synthetic `tx` tokens in a valid range
        synthetic_tx_tokens = torch.randint(0, self.segger_model.tx_embedding.num_embeddings, (num_synthetic,))
        # Process through Segger's embedding logic
        synthetic_embeddings = self.segger_model.tx_embedding(synthetic_tx_tokens)
        # Append synthetic nodes
        batch.x = torch.cat([batch.x, synthetic_embeddings], dim=0)
        batch.y = torch.cat([batch.y, batch.y[:num_synthetic]], dim=0)  # Copy labels from real nodes
        batch.meta['is_synthetic'] = torch.cat([torch.zeros(num_nodes), torch.ones(num_synthetic)])
        return batch

    def enrich_global_edges(self, batch: Batch):
        """
        Adds global edges using both 'global_gene_cov' and learned latent representations from Segger.
        """
        if 'global_gene_cov' not in batch.meta:
            logging.warning("global_gene_cov not found in batch['meta'], skipping edge enrichment.")
            return batch
        # Compute correlation-based edges from global_gene_cov
        correlation_matrix = torch.corrcoef(batch.meta['global_gene_cov'].T)
        high_corr_edges = (correlation_matrix > self.correlation_threshold).nonzero(as_tuple=False)

        # Convert to node-node edges based on gene expression
        new_edges = []
        for gene_a, gene_b in high_corr_edges:
            nodes_a = (batch.meta['global_gene_cov'][:, gene_a] > 0).nonzero().squeeze()
            nodes_b = (batch.meta['global_gene_cov'][:, gene_b] > 0).nonzero().squeeze()
            for node_a in nodes_a:
                for node_b in nodes_b:
                    if node_a != node_b:
                        new_edges.append((node_a.item(), node_b.item()))

        # **Learned Edge Enrichment**
        with torch.no_grad():
            z = self.segger_model(batch.x, batch.edge_index)
            similarity_matrix = torch.mm(z, z.T)  # Compute dot-product similarity
        # Find new high-similarity edges
        learned_edges = (similarity_matrix > self.correlation_threshold).nonzero(as_tuple=False)
        new_edges.extend(learned_edges.tolist())
        if len(new_edges) > 0:
            new_edges = torch.tensor(new_edges, dtype=torch.long).T  # Convert to tensor
            batch.edge_index = torch.cat([batch.edge_index, new_edges], dim=1)
        return batch

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=self.train_shuffle, collate_fn=self._collate_batch, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, collate_fn=self._collate_batch, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, collate_fn=self._collate_batch, **self.loader_kwargs)

    def _collate_batch(self, batch_list):
        """
        Custom collate function that injects synthetic nodes and enriches global edges dynamically.
        """
        batch = Batch.from_data_list(batch_list)
        if(self.inject_synthetic_extremes):
            batch = self.inject_synthetic_extreme_transcripts(batch)
        if(self.enrich_global_edges):
            batch = self.enrich_global_edges(batch)
        return batch
