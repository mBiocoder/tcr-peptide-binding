"""
embedder.py
===========
Protein language model wrappers for generating per-residue embeddings.

Supports ESM-2 (Facebook AI Research) and ProtBERT (Rostlab).  Both models
produce a floating-point embedding vector for every amino acid residue in the
input sequence, capturing evolutionary patterns and biochemical context learned
from hundreds of millions of protein sequences.

Supported model identifiers
----------------------------
ESM-2 (recommended):
    - ``"facebook/esm2_t6_8M_UR50D"``     (8M params, fast, good for testing)
    - ``"facebook/esm2_t12_35M_UR50D"``   (35M params, demo default)
    - ``"facebook/esm2_t33_650M_UR50D"``  (650M params, best accuracy)

ProtBERT:
    - ``"Rostlab/prot_bert"``             (420M params, 1024-dim embeddings)

Classes
-------
ProteinEmbedder
    Loads a HuggingFace protein model and exposes :meth:`embed_sequence` /
    :meth:`embed_batch`.

Examples
--------
>>> embedder = ProteinEmbedder("facebook/esm2_t12_35M_UR50D")
>>> emb = embedder.embed_sequence("CASSLAPGATNEKLFF")
>>> emb.shape
torch.Size([16, 480])
"""

from __future__ import annotations

import logging
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ProteinEmbedder:
    """Generate per-residue embeddings using a pretrained protein language model.

    The model is loaded once at construction time and kept in evaluation mode.
    All gradient computation is disabled to minimise memory usage during
    inference.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to the 35M ESM-2 model which
        is suitable for CPU demos.  Use ``"facebook/esm2_t33_650M_UR50D"``
        for production runs.
    device : str, default ``"cuda"``
        Target device.  Falls back to ``"cpu"`` automatically if CUDA is not
        available.

    Attributes
    ----------
    embed_dim : int
        Dimensionality of the output embeddings (e.g. 480 for ESM-2 35M,
        1280 for ESM-2 650M, 1024 for ProtBERT).

    Notes
    -----
    Embedding generation is the computational bottleneck when training from
    scratch.  In production, pre-compute and cache all embeddings once before
    training rather than computing them on-the-fly inside the DataLoader.

    Examples
    --------
    >>> embedder = ProteinEmbedder("facebook/esm2_t12_35M_UR50D", device="cpu")
    Embedding dimension: 480
    >>> t = embedder.embed_sequence("GILGFVFTL")
    >>> t.shape
    torch.Size([9, 480])
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"

        logger.info("Loading protein language model: %s", model_name)
        logger.info("Using device: %s", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.embed_dim: int = self.model.config.hidden_size
        print(f"Embedding dimension: {self.embed_dim}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """Embed a single amino acid sequence.

        Parameters
        ----------
        sequence : str
            Single-letter amino acid string, e.g. ``"CASSLAPGATNEKLFF"``.
            Non-standard residues are handled by the tokenizer's UNK token.

        Returns
        -------
        torch.Tensor
            Shape ``(seq_len, embed_dim)``.  Always returned on CPU.

        Examples
        --------
        >>> emb = embedder.embed_sequence("CASSLAPGATNEKLFF")
        >>> emb.shape
        torch.Size([16, 480])
        """
        # ESM-2 expects space-separated residues; ProtBERT expects the same
        spaced = " ".join(list(sequence))
        inputs = self.tokenizer(spaced, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # Strip BOS and EOS special tokens → pure residue embeddings
        embeddings: torch.Tensor = outputs.last_hidden_state[0, 1:-1, :]
        return embeddings.cpu()

    def embed_batch(self, sequences: List[str]) -> List[torch.Tensor]:
        """Embed a list of sequences, returning one tensor per sequence.

        Parameters
        ----------
        sequences : list of str
            Amino acid sequences of (potentially) varying length.

        Returns
        -------
        list of torch.Tensor
            Each tensor has shape ``(len(seq), embed_dim)``.

        Examples
        --------
        >>> seqs = ["CASSLAPGATNEKLFF", "GILGFVFTL"]
        >>> embs = embedder.embed_batch(seqs)
        >>> [e.shape for e in embs]
        [torch.Size([16, 480]), torch.Size([9, 480])]
        """
        return [self.embed_sequence(seq) for seq in sequences]
