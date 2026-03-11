"""
data.py
=======
Dataset utilities for TCR-peptide binding prediction.

Provides:
- TCRPeptideDataset   : PyTorch Dataset wrapping a pandas DataFrame
- create_sample_data  : Synthetic data generator for demos and tests
- download_vdjdb      : Downloads and preprocesses the VDJdb database
- generate_negatives  : Creates shuffled-peptide negative samples
- GraphCollator       : Custom DataLoader collate function for graph batching

Typical usage
-------------
>>> from src.data import download_vdjdb, generate_negatives, TCRPeptideDataset
>>> positives = download_vdjdb()
>>> dataset   = generate_negatives(positives, ratio=1.0)
>>> ds        = TCRPeptideDataset(dataset)
"""

from __future__ import annotations

import hashlib
import io
import logging
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"

VDJDB_URL: str = (
    "https://github.com/antigenomics/vdjdb-db/raw/master/latest/vdjdb.slim.txt"
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TCRPeptideDataset(Dataset):
    """PyTorch Dataset for TCR–peptide pairs with binding labels.

    Each sample is a dict with keys ``tcr``, ``peptide``, and ``label``.
    The collation of raw strings into graph tensors is handled downstream by
    :class:`GraphCollator`.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns ``tcr_sequence``, ``peptide_sequence``, and
        ``label`` (integer 0 / 1).

    Examples
    --------
    >>> df = create_sample_data(n_samples=100)
    >>> ds = TCRPeptideDataset(df)
    >>> sample = ds[0]
    >>> sample.keys()
    dict_keys(['tcr', 'peptide', 'label'])
    """

    def __init__(self, data: pd.DataFrame) -> None:
        required = {"tcr_sequence", "peptide_sequence", "label"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        self.data = data.reset_index(drop=True)
        self.tcr_sequences: np.ndarray = data["tcr_sequence"].values
        self.peptide_sequences: np.ndarray = data["peptide_sequence"].values
        self.labels: np.ndarray = data["label"].values

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return {
            "tcr": self.tcr_sequences[idx],
            "peptide": self.peptide_sequences[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def create_sample_data(n_samples: int = 1_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic TCR–peptide data for demonstration and unit tests.

    Sequences are randomly sampled amino acid strings with lengths matching
    realistic CDR3 (12–18 aa) and MHC-I peptide (8–11 aa) distributions.
    A simple heuristic (shared amino acid alphabet overlap) is used to assign
    binary binding labels — not biologically accurate, but structurally correct
    for pipeline testing.

    Parameters
    ----------
    n_samples : int, default 1000
        Number of TCR–peptide pairs to generate.
    seed : int, default 42
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``tcr_sequence``, ``peptide_sequence``, ``label``.

    Examples
    --------
    >>> df = create_sample_data(n_samples=500)
    >>> df.shape
    (500, 3)
    >>> set(df['label'].unique()) == {0, 1}
    True
    """
    rng = np.random.default_rng(seed)
    aas = list(AMINO_ACIDS)
    rows: list[dict] = []

    for _ in range(n_samples):
        tcr_len = int(rng.integers(12, 19))
        pep_len = int(rng.integers(8, 12))
        tcr_seq = "".join(rng.choice(aas, tcr_len))
        pep_seq = "".join(rng.choice(aas, pep_len))
        label = 1 if len(set(tcr_seq) & set(pep_seq)) > 8 else 0
        rows.append(
            {"tcr_sequence": tcr_seq, "peptide_sequence": pep_seq, "label": label}
        )

    df = pd.DataFrame(rows)
    logger.info(
        "Created synthetic dataset: %d samples (%d positive, %d negative)",
        len(df),
        df["label"].sum(),
        (df["label"] == 0).sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Real data — VDJdb
# ---------------------------------------------------------------------------


def download_vdjdb(save_path: Optional[str] = None) -> pd.DataFrame:
    """Download and preprocess the VDJdb TCR–peptide database.

    Filters for human TCR beta chains with non-null CDR3 and epitope fields.
    All downloaded records are positive binders (label = 1).  To create a
    balanced dataset, pass the result to :func:`generate_negatives`.

    Parameters
    ----------
    save_path : str or None, default None
        If provided, save the processed DataFrame as a CSV to this path.

    Returns
    -------
    pd.DataFrame
        Columns: ``tcr_sequence``, ``peptide_sequence``, ``label``,
        ``mhc_allele``, ``vdjdb_score``.

    Raises
    ------
    requests.HTTPError
        If the download fails.

    Notes
    -----
    The raw file is ~10 MB and downloads in a few seconds.  The processed
    DataFrame typically contains ~55,000–61,000 rows depending on the
    database version.

    Examples
    --------
    >>> df = download_vdjdb()
    >>> df.shape[1]
    5
    """
    logger.info("Downloading VDJdb from %s …", VDJDB_URL)
    response = requests.get(VDJDB_URL, timeout=60)
    response.raise_for_status()

    raw = pd.read_csv(io.StringIO(response.text), sep="\t", low_memory=False)

    # Filter for human TCR-beta entries with complete data
    mask = (
        (raw["species"] == "HomoSapiens")
        & (raw["gene"] == "TRB")
        & raw["cdr3"].notna()
        & raw["antigen.epitope"].notna()
    )
    filtered = raw[mask].copy()

    df = pd.DataFrame(
        {
            "tcr_sequence": filtered["cdr3"].values,
            "peptide_sequence": filtered["antigen.epitope"].values,
            "label": 1,
            "mhc_allele": filtered.get("mhc.a", pd.Series(dtype=str)).values,
            "vdjdb_score": filtered.get("vdjdb.score", pd.Series(dtype=int)).values,
        }
    )

    # Remove duplicates
    df = df.drop_duplicates(subset=["tcr_sequence", "peptide_sequence"]).reset_index(
        drop=True
    )

    logger.info("Downloaded %d unique TCR–peptide pairs from VDJdb.", len(df))

    if save_path:
        df.to_csv(save_path, index=False)
        logger.info("Saved to %s", save_path)

    return df


# ---------------------------------------------------------------------------
# Negative sample generation
# ---------------------------------------------------------------------------


def generate_negatives(
    positive_data: pd.DataFrame,
    ratio: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate negative (non-binding) TCR–peptide pairs by random shuffling.

    Pairs are created by randomly combining TCRs and peptides from the
    positive set.  Any generated pair that already appears in the positive
    set is discarded.  This approach is the standard in the field (used by
    NetTCR, ERGO, etc.) and assumes random pairings are unlikely true binders.

    Parameters
    ----------
    positive_data : pd.DataFrame
        DataFrame of confirmed binders with columns ``tcr_sequence``,
        ``peptide_sequence``, ``label``.
    ratio : float, default 1.0
        Ratio of negatives to positives.  ``1.0`` → balanced dataset.
    seed : int, default 42
        NumPy random seed.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of positives and negatives, shuffled.

    Examples
    --------
    >>> pos = create_sample_data(500)
    >>> pos = pos[pos['label'] == 1].reset_index(drop=True)
    >>> full = generate_negatives(pos, ratio=1.0)
    >>> full['label'].value_counts().to_dict()
    {1: ..., 0: ...}
    """
    rng = np.random.default_rng(seed)
    n_neg = int(len(positive_data) * ratio)

    tcrs = positive_data["tcr_sequence"].values
    peps = positive_data["peptide_sequence"].values

    # Set of existing positive pairs for fast lookup
    positive_pairs: set[tuple[str, str]] = set(
        zip(positive_data["tcr_sequence"], positive_data["peptide_sequence"])
    )

    negatives: list[dict] = []
    attempts = 0
    max_attempts = n_neg * 10

    while len(negatives) < n_neg and attempts < max_attempts:
        tcr = str(rng.choice(tcrs))
        pep = str(rng.choice(peps))
        if (tcr, pep) not in positive_pairs:
            negatives.append({"tcr_sequence": tcr, "peptide_sequence": pep, "label": 0})
            positive_pairs.add((tcr, pep))  # avoid duplicates within negatives
        attempts += 1

    neg_df = pd.DataFrame(negatives)

    combined = (
        pd.concat([positive_data[["tcr_sequence", "peptide_sequence", "label"]], neg_df])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    logger.info(
        "Dataset: %d positives, %d negatives (total %d).",
        (combined["label"] == 1).sum(),
        (combined["label"] == 0).sum(),
        len(combined),
    )
    return combined


# ---------------------------------------------------------------------------
# Graph collator  (used by DataLoader)
# ---------------------------------------------------------------------------


class GraphCollator:
    """DataLoader collate function that converts sequences to graphs on-the-fly.

    This class is used as the ``collate_fn`` argument to
    ``torch.utils.data.DataLoader``.  For each mini-batch it:

    1. Generates per-residue embeddings via the supplied ``embedder``
    2. Constructs a :class:`torch_geometric.data.Data` graph per sequence
    3. Batches TCR graphs and peptide graphs separately using
       :meth:`torch_geometric.data.Batch.from_data_list`

    Parameters
    ----------
    embedder : src.embedder.ProteinEmbedder
        Initialised protein language model embedder.

    Examples
    --------
    >>> from src.embedder import ProteinEmbedder
    >>> from src.data import create_sample_data, TCRPeptideDataset, GraphCollator
    >>> from torch.utils.data import DataLoader
    >>> embedder  = ProteinEmbedder('facebook/esm2_t12_35M_UR50D')
    >>> collator  = GraphCollator(embedder)
    >>> ds        = TCRPeptideDataset(create_sample_data(32))
    >>> loader    = DataLoader(ds, batch_size=4, collate_fn=collator)
    >>> batch     = next(iter(loader))
    >>> batch['tcr_graph'].num_graphs
    4
    """

    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def __call__(self, batch_list: List[dict]) -> dict:
        from src.graph import sequence_to_graph  # local import to avoid circular deps

        tcr_graphs: List[Data] = []
        peptide_graphs: List[Data] = []
        labels: List[torch.Tensor] = []

        for sample in batch_list:
            tcr_emb = self.embedder.embed_sequence(sample["tcr"])
            pep_emb = self.embedder.embed_sequence(sample["peptide"])

            tcr_graphs.append(sequence_to_graph(sample["tcr"], tcr_emb))
            peptide_graphs.append(sequence_to_graph(sample["peptide"], pep_emb))
            labels.append(sample["label"])

        return {
            "tcr_graph": Batch.from_data_list(tcr_graphs),
            "peptide_graph": Batch.from_data_list(peptide_graphs),
            "label": torch.stack(labels),
        }
