"""Unit tests for src/data.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import pytest
import torch

from src.data import create_sample_data, generate_negatives, TCRPeptideDataset


def test_create_sample_data_shape():
    df = create_sample_data(n_samples=100, seed=0)
    assert df.shape == (100, 3)
    assert set(df.columns) == {"tcr_sequence", "peptide_sequence", "label"}


def test_create_sample_data_labels():
    df = create_sample_data(n_samples=200, seed=0)
    assert set(df["label"].unique()).issubset({0, 1})


def test_create_sample_data_reproducible():
    df1 = create_sample_data(n_samples=50, seed=7)
    df2 = create_sample_data(n_samples=50, seed=7)
    assert df1.equals(df2)


def test_generate_negatives_balance():
    positives = create_sample_data(100)
    positives = positives[positives["label"] == 1].reset_index(drop=True)
    full = generate_negatives(positives, ratio=1.0, seed=0)
    n_pos = (full["label"] == 1).sum()
    n_neg = (full["label"] == 0).sum()
    assert n_pos > 0
    assert n_neg > 0


def test_generate_negatives_no_overlap():
    positives = create_sample_data(100)[["tcr_sequence", "peptide_sequence"]].copy()
    positives["label"] = 1
    negatives_df = generate_negatives(positives, ratio=1.0)
    neg_only = negatives_df[negatives_df["label"] == 0]
    pos_pairs = set(zip(positives["tcr_sequence"], positives["peptide_sequence"]))
    for _, row in neg_only.iterrows():
        assert (row["tcr_sequence"], row["peptide_sequence"]) not in pos_pairs


def test_dataset_getitem():
    df = create_sample_data(n_samples=10)
    ds = TCRPeptideDataset(df)
    assert len(ds) == 10
    sample = ds[0]
    assert set(sample.keys()) == {"tcr", "peptide", "label"}
    assert isinstance(sample["label"], torch.Tensor)


def test_dataset_missing_columns():
    with pytest.raises(ValueError):
        TCRPeptideDataset(pd.DataFrame({"a": [1]}))
