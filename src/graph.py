"""
graph.py
========
Convert amino acid sequences (with pLM embeddings) into
``torch_geometric.data.Data`` graph objects for use with GNN layers.

Graph structure
---------------
**Nodes**: one per residue; node features = ESM-2 / ProtBERT embedding vector.

**Edges** (two types, both bidirectional):

1. *Sequential* edges — connect adjacent residues ``i ↔ i+1``, capturing
   the polypeptide backbone.
2. *k-Nearest-Neighbour* (k-NN) edges — connect residues whose embedding
   vectors are closest in Euclidean space.  Because ESM-2 embeddings capture
   biochemical similarity, k-NN edges approximate functional / spatial
   proximity without requiring a 3-D structure.

Functions
---------
sequence_to_graph
    Main conversion function.
build_sequential_edges
    Returns only backbone connectivity.
build_knn_edges
    Returns k-NN edges based on pLM embedding similarity.

Examples
--------
>>> import torch
>>> embeddings = torch.randn(16, 480)
>>> g = sequence_to_graph("CASSLAPGATNEKLFF", embeddings, k_neighbors=5)
>>> g.num_nodes, g.num_edges
(16, ...)
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def build_sequential_edges(seq_len: int) -> list[list[int]]:
    """Return bidirectional sequential (backbone) edges for a sequence.

    Parameters
    ----------
    seq_len : int
        Number of residues.

    Returns
    -------
    list of [int, int]
        Edge list in COO format (before conversion to tensor).

    Examples
    --------
    >>> edges = build_sequential_edges(4)
    >>> edges
    [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]
    """
    edges: list[list[int]] = []
    for i in range(seq_len - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return edges


def build_knn_edges(embeddings: torch.Tensor, k: int = 5) -> list[list[int]]:
    """Return k-nearest-neighbour edges based on embedding similarity.

    Pairwise Euclidean distances between residue embeddings are computed;
    for each residue the ``k`` closest neighbours (excluding self) are
    connected.  Duplicate edges (arising from mutual neighbourhood) are kept
    because the GAT message-passing step treats them as independent signals.

    Parameters
    ----------
    embeddings : torch.Tensor
        Shape ``(seq_len, embed_dim)``.
    k : int, default 5
        Number of neighbours per node.

    Returns
    -------
    list of [int, int]
        Edge list in COO format.

    Examples
    --------
    >>> emb = torch.randn(10, 480)
    >>> edges = build_knn_edges(emb, k=3)
    >>> len(edges) <= 10 * 3
    True
    """
    seq_len = embeddings.shape[0]
    if seq_len <= 1:
        return []

    # Pairwise distances: (seq_len, seq_len)
    distances = torch.cdist(embeddings.float(), embeddings.float(), p=2)

    edges: list[list[int]] = []
    k_actual = min(k + 1, seq_len)  # +1 because topk includes self

    for i in range(seq_len):
        _, indices = torch.topk(distances[i], k=k_actual, largest=False)
        for j in indices:
            j_int = int(j.item())
            if j_int != i:
                edges.append([i, j_int])

    return edges


def sequence_to_graph(
    sequence: str,
    embeddings: torch.Tensor,
    k_neighbors: int = 5,
) -> Data:
    """Convert an amino acid sequence and its pLM embeddings to a graph.

    Combines sequential backbone edges and k-NN embedding-similarity edges
    into a single :class:`torch_geometric.data.Data` object.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (used only for its length; not stored in the
        graph object).
    embeddings : torch.Tensor
        Per-residue embedding tensor of shape ``(len(sequence), embed_dim)``
        as returned by :meth:`src.embedder.ProteinEmbedder.embed_sequence`.
    k_neighbors : int, default 5
        Number of k-NN edges per node.

    Returns
    -------
    torch_geometric.data.Data
        Graph with:
        - ``x``          — node features  ``(seq_len, embed_dim)``
        - ``edge_index`` — edge indices   ``(2, n_edges)``

    Raises
    ------
    ValueError
        If ``len(sequence) != embeddings.shape[0]``.

    Notes
    -----
    For very short sequences (≤ 2 residues) k-NN edges degenerate to
    sequential edges; the function handles this gracefully.

    Examples
    --------
    >>> import torch
    >>> emb = torch.randn(9, 480)
    >>> g = sequence_to_graph("GILGFVFTL", emb, k_neighbors=3)
    >>> g.x.shape
    torch.Size([9, 480])
    >>> g.edge_index.shape[0]
    2
    """
    seq_len = len(sequence)
    if embeddings.shape[0] != seq_len:
        raise ValueError(
            f"Sequence length {seq_len} does not match "
            f"embeddings shape {embeddings.shape}."
        )

    # Build edge list
    edge_list = build_sequential_edges(seq_len)
    edge_list += build_knn_edges(embeddings, k=k_neighbors)

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # Single-residue sequence — no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=embeddings, edge_index=edge_index)
