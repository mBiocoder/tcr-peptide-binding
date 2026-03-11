"""
model.py
========
Neural network architectures for TCR–peptide binding prediction.

Classes
-------
GATEncoder
    3-layer Graph Attention Network (GAT) that maps a protein graph
    (variable size) to a fixed-length embedding vector.

TCRPeptideBindingModel
    Full binding-prediction model that encodes a TCR graph and a peptide
    graph with a shared :class:`GATEncoder`, concatenates the two
    graph-level embeddings, and passes them through an MLP classifier.

Architecture overview::

    TCR graph  ─►  GATEncoder ─►  [128]  ─►  concat [256]  ─►  MLP  ─►  p(bind)
    Pep graph  ─►  GATEncoder ─►  [128]  ─┘

Notes
-----
The :class:`GATEncoder` uses:

- An input linear projection (``embed_dim → hidden_dim``) to allow ESM-2's
  high-dimensional embeddings (480–1280) to be mapped into a tractable space.
- Residual connections after each GAT layer to ease gradient flow in deep
  networks.
- Layer normalisation for training stability.
- Global mean pooling to aggregate variable-size node sets to a single vector.

Examples
--------
>>> from src.model import TCRPeptideBindingModel
>>> model = TCRPeptideBindingModel(input_dim=480, hidden_dim=128)
>>> sum(p.numel() for p in model.parameters())
...
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATEncoder(nn.Module):
    """Graph Attention Network encoder for protein residue graphs.

    Takes a batched protein graph and returns a fixed-size embedding per
    graph in the batch via global mean pooling.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input node features (= embedding dimension of the
        protein language model, e.g. 480 for ESM-2 35M or 1280 for 650M).
    hidden_dim : int, default 128
        Width of all hidden GAT layers.  Must be divisible by ``num_heads``.
    num_layers : int, default 3
        Number of stacked GAT message-passing layers.
    num_heads : int, default 4
        Number of attention heads in each GAT layer.  The output of each
        layer has dimension ``hidden_dim`` (heads are concatenated then
        projected internally via ``concat=True``).
    dropout : float, default 0.2
        Dropout probability applied to edge attention coefficients inside GAT
        and to node features between layers.

    Returns
    -------
    torch.Tensor
        Graph-level embeddings of shape ``(batch_size, hidden_dim)``.

    Examples
    --------
    >>> from torch_geometric.data import Data, Batch
    >>> import torch
    >>> g = Data(x=torch.randn(10, 480),
    ...          edge_index=torch.tensor([[0,1],[1,0]]))
    >>> batch = Batch.from_data_list([g, g])
    >>> enc = GATEncoder(input_dim=480)
    >>> out = enc(batch.x, batch.edge_index, batch.batch)
    >>> out.shape
    torch.Size([2, 128])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        # Project pLM embeddings (high-dim) into the hidden space
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GAT message-passing layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        head_dim = hidden_dim // num_heads
        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    head_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,   # output dim = head_dim * num_heads = hidden_dim
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix ``(total_nodes, input_dim)``.
        edge_index : torch.Tensor
            Edge index tensor ``(2, total_edges)`` in COO format.
        batch : torch.Tensor
            Batch assignment vector ``(total_nodes,)`` mapping each node to
            its graph in the batch.

        Returns
        -------
        torch.Tensor
            Graph-level embeddings ``(batch_size, hidden_dim)``.
        """
        # Project into hidden space
        x = F.relu(self.input_proj(x))

        # Stacked GAT layers with residual connections
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            residual = x
            x = gat(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + residual  # residual connection

        # Aggregate all node embeddings into a single graph-level vector
        x = global_mean_pool(x, batch)
        return x


# ---------------------------------------------------------------------------


class TCRPeptideBindingModel(nn.Module):
    """End-to-end TCR–peptide binding prediction model.

    Architecture
    ------------
    1. A shared :class:`GATEncoder` encodes both the TCR graph and the
       peptide graph independently into 128-dimensional vectors.
    2. The two vectors are concatenated to form a 256-dimensional joint
       representation.
    3. A 3-layer MLP with ReLU activations, dropout, and a final sigmoid
       unit outputs a scalar binding probability in ``[0, 1]``.

    Sharing encoder weights between TCR and peptide is a deliberate design
    choice: it reduces parameter count, enforces a common representational
    space, and improves generalisation when data is limited.

    Parameters
    ----------
    input_dim : int
        Protein language model embedding dimension (e.g. 480 or 1280).
    hidden_dim : int, default 128
        Hidden dimension of the GAT encoder (and first MLP layer).
    num_gat_layers : int, default 3
        Number of GAT message-passing layers.

    Examples
    --------
    >>> model = TCRPeptideBindingModel(input_dim=480, hidden_dim=128)
    >>> n_params = sum(p.numel() for p in model.parameters())
    >>> print(f"Parameters: {n_params:,}")
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_gat_layers: int = 3,
    ) -> None:
        super().__init__()

        # Shared encoder — processes both TCR and peptide graphs
        self.graph_encoder = GATEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers,
        )

        # Fusion MLP classifier
        fusion_dim = hidden_dim * 2  # [TCR_repr || Peptide_repr]
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------

    def forward(self, tcr_batch, peptide_batch) -> torch.Tensor:
        """Compute binding probabilities for a batch of TCR–peptide pairs.

        Parameters
        ----------
        tcr_batch : torch_geometric.data.Batch
            Batched TCR graphs.
        peptide_batch : torch_geometric.data.Batch
            Batched peptide graphs.

        Returns
        -------
        torch.Tensor
            Binding probabilities of shape ``(batch_size, 1)``,
            values in ``[0, 1]``.

        Examples
        --------
        >>> pred = model(tcr_batch, peptide_batch)
        >>> pred.shape
        torch.Size([32, 1])
        """
        tcr_repr = self.graph_encoder(
            tcr_batch.x, tcr_batch.edge_index, tcr_batch.batch
        )
        peptide_repr = self.graph_encoder(
            peptide_batch.x, peptide_batch.edge_index, peptide_batch.batch
        )

        combined = torch.cat([tcr_repr, peptide_repr], dim=1)
        return self.classifier(combined)

    # ------------------------------------------------------------------

    def get_graph_embeddings(
        self, tcr_batch, peptide_batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the intermediate graph-level embeddings (for interpretability).

        Parameters
        ----------
        tcr_batch : torch_geometric.data.Batch
        peptide_batch : torch_geometric.data.Batch

        Returns
        -------
        tcr_repr : torch.Tensor  shape ``(batch_size, hidden_dim)``
        peptide_repr : torch.Tensor  shape ``(batch_size, hidden_dim)``
        """
        with torch.no_grad():
            tcr_repr = self.graph_encoder(
                tcr_batch.x, tcr_batch.edge_index, tcr_batch.batch
            )
            peptide_repr = self.graph_encoder(
                peptide_batch.x, peptide_batch.edge_index, peptide_batch.batch
            )
        return tcr_repr, peptide_repr
