"""Unit tests for src/graph.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from src.graph import (
    build_sequential_edges,
    build_knn_edges,
    sequence_to_graph,
)


def test_sequential_edges_count():
    edges = build_sequential_edges(5)
    # Bidirectional: (n-1) * 2 = 8 edges for n=5
    assert len(edges) == 8


def test_sequential_edges_bidirectional():
    edges = build_sequential_edges(3)
    assert [0, 1] in edges
    assert [1, 0] in edges
    assert [1, 2] in edges
    assert [2, 1] in edges


def test_sequential_edges_single_node():
    edges = build_sequential_edges(1)
    assert edges == []


def test_knn_edges_count():
    emb = torch.randn(10, 32)
    edges = build_knn_edges(emb, k=3)
    # At most n * k edges
    assert len(edges) <= 10 * 3


def test_knn_edges_no_self_loops():
    emb = torch.randn(8, 32)
    edges = build_knn_edges(emb, k=4)
    for src, dst in edges:
        assert src != dst


def test_sequence_to_graph_shapes():
    seq = "CASSLAPG"
    emb = torch.randn(len(seq), 48)
    g = sequence_to_graph(seq, emb, k_neighbors=3)
    assert g.x.shape == (len(seq), 48)
    assert g.edge_index.shape[0] == 2


def test_sequence_to_graph_length_mismatch():
    with pytest.raises(ValueError):
        sequence_to_graph("ABC", torch.randn(5, 48))


def test_sequence_to_graph_single_residue():
    g = sequence_to_graph("A", torch.randn(1, 48))
    assert g.num_nodes == 1
    assert g.num_edges == 0
