"""Unit tests for src/model.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from torch_geometric.data import Batch, Data

from src.model import GATEncoder, TCRPeptideBindingModel


def make_batch(n_graphs: int = 4, n_nodes: int = 10, feat_dim: int = 64) -> Batch:
    graphs = []
    for _ in range(n_graphs):
        x = torch.randn(n_nodes, feat_dim)
        # Simple sequential edges (bidirectional)
        src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
        dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


# ---- GATEncoder ----

def test_gat_encoder_output_shape():
    batch = make_batch(n_graphs=4, n_nodes=10, feat_dim=64)
    enc = GATEncoder(input_dim=64, hidden_dim=128, num_layers=3, num_heads=4)
    out = enc(batch.x, batch.edge_index, batch.batch)
    assert out.shape == (4, 128)


def test_gat_encoder_hidden_not_divisible():
    with pytest.raises(ValueError):
        GATEncoder(input_dim=64, hidden_dim=130, num_heads=4)  # 130 % 4 != 0


def test_gat_encoder_single_graph():
    batch = make_batch(n_graphs=1, n_nodes=5, feat_dim=32)
    enc = GATEncoder(input_dim=32, hidden_dim=64, num_layers=2, num_heads=4)
    out = enc(batch.x, batch.edge_index, batch.batch)
    assert out.shape == (1, 64)


# ---- TCRPeptideBindingModel ----

def test_binding_model_output_shape():
    model = TCRPeptideBindingModel(input_dim=64, hidden_dim=128)
    tcr_b = make_batch(n_graphs=8, feat_dim=64)
    pep_b = make_batch(n_graphs=8, feat_dim=64)
    out = model(tcr_b, pep_b)
    assert out.shape == (8, 1)


def test_binding_model_output_range():
    model = TCRPeptideBindingModel(input_dim=64, hidden_dim=128)
    tcr_b = make_batch(n_graphs=4, feat_dim=64)
    pep_b = make_batch(n_graphs=4, feat_dim=64)
    out = model(tcr_b, pep_b)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_binding_model_get_embeddings():
    model = TCRPeptideBindingModel(input_dim=64, hidden_dim=128)
    tcr_b = make_batch(n_graphs=3, feat_dim=64)
    pep_b = make_batch(n_graphs=3, feat_dim=64)
    tcr_repr, pep_repr = model.get_graph_embeddings(tcr_b, pep_b)
    assert tcr_repr.shape == (3, 128)
    assert pep_repr.shape == (3, 128)


def test_binding_model_param_count():
    model = TCRPeptideBindingModel(input_dim=480, hidden_dim=128)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    assert n_params < 10_000_000  # sanity: not unexpectedly large
