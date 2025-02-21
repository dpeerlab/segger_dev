import torch
from torch_geometric.nn import GATv2Conv, Linear, HeteroDictLinear, HeteroConv
from torch.nn import (
    Embedding,
    ModuleDict,
    ModuleList,
    Module,
    functional as F
)
from torch import Tensor
from typing import Union, Optional


class SkipGAT(Module):
    """
    TODO: Add description.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
    ):
        """
        TODO: Add description.
        """
        super().__init__()

        # Message-passing
        self.conv = HeteroConv(
            convs={
                ('tx','neighbors', 'tx'): GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                ),
                ('tx', 'belongs', 'bd'): GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                    add_self_loops=False,
                ),
            },
            aggr='sum',
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)
        return x_dict


class Segger(torch.nn.Module):
    """
    TODO: Add description.
    """

    def __init__(
        self,
        num_tx_tokens: int,
        in_channels: int = 16,
        hidden_channels: int = 32,
        out_channels: int = 32,
        num_mid_layers: int = 3,
        heads: int = 3,
        embedding_weight: Tensor | None = None,
    ):
        """
        Initialize the Segger model.

        Parameters
        ----------
        num_tx_tokens : int
            Number of unique 'tx' tokens for embedding.
        in_channels : int, optional
            Initial embedding size for both 'tx' and boundary nodes.
            Default is 16.
        hidden_channels : int, optional
            Number of hidden channels. Default is 32.
        out_channels : int, optional
            Number of output channels. Default is 32.
        num_mid_layers : int, optional
            Number of hidden layers (excluding first and last layers).
            Default is 3.
        heads : int, optional
            Number of attention heads. Default is 3.
        embedding_weight : Tensor, optional
            Pretrained embedding weights for 'tx' tokens. If None,
            weights are initialized randomly. Default is None.
        """
        super().__init__()
        # Store hyperparameters for PyTorch Lightning
        self.hparams = locals()
        # First layer: ? -> in
        self.lin_first = ModuleDict(
            {
                'tx': Embedding(
                    num_tx_tokens,
                    in_channels,
                    _weight=embedding_weight,
                ),
                'bd': Linear(-1, in_channels),
            }
        )
        self.conv_layers = ModuleList()
        # First convolution: in -> hidden x heads
        self.conv_layers.append(
            SkipGAT(in_channels, hidden_channels, heads)
        )
        # Middle convolutions: hidden x heads -> hidden x heads
        for _ in range(num_mid_layers):
            self.conv_layers.append(
                SkipGAT((-1, -1), hidden_channels, heads)
            )
        # Last convolution: hidden x heads -> out x heads
        self.conv_layers.append(
            SkipGAT((-1, -1), out_channels, heads)
        )
        # Last layer: out x heads -> out
        self.lin_last = HeteroDictLinear(
            -1,
            out_channels,
            types=("tx", "bd")
        )

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """
        Forward pass for the Segger model.

        Parameters
        ----------
        x_dict : dict[str, Tensor]
            Node features for each node type.
        edge_index_dict : dict[str, Tensor]
            Edge indices for each edge type.

        Returns
        -------
        Tensor
            Output node features after passing through the Segger model.
        """
        # Linearly project embedding to input dim
        x_dict = {k: self.lin_first[k](x) for k, x in x_dict.items()}

        # ReLu for some reason
        x_dict = {k: F.leaky_relu(x) for k, x in x_dict.items()}

        # Graph convolutions with GATv2
        for conv_layer in self.conv_layers:
            x_dict = conv_layer(x_dict, edge_index_dict)
            x_dict = {k: F.leaky_relu(x) for k, x in x_dict.items()}

        # Linearly project to output dim
        x_dict = self.lin_last(x_dict)

        return x_dict

    def decode(
        self,
        z: dict[str, Tensor],
        edge_index: Union[Tensor],
    ) -> Tensor:
        """
        Decode the node embeddings to predict edge values.

        Parameters
        ----------
        z : dict[str, Tensor]
            Node embeddings for each node type.
        edge_index : EdgeIndex
            Edge label indices.

        Returns
        -------
        Tensor
            Predicted edge values.
        """
        z_tx = z["tx"][edge_index[0]]
        z_bd = z["bd"][edge_index[1]]

        return (z_tx * z_bd).sum(dim=-1)
