# Architecture Documentation

## Overview

The TCR–peptide binding model is a hybrid architecture combining:

1. **Protein Language Model (pLM)** — generates context-aware per-residue embeddings
2. **Graph Attention Network (GAT)** — models residue interactions via message passing
3. **Fusion MLP** — classifies TCR–peptide pairs from paired graph embeddings

---

## Detailed Architecture

### Stage 1: Sequence Embedding (ESM-2)

| Detail | Value |
|---|---|
| Model | `facebook/esm2_t33_650M_UR50D` (production) |
| Parameters | 650M |
| Output | Per-residue embedding tensor `(L, 1280)` |
| Training | Frozen (not fine-tuned) |

ESM-2 is a masked language model pre-trained on 250M protein sequences from UniRef50.
Its representations capture evolutionary conservation, local structural context, and biochemical properties without requiring a 3-D structure.

### Stage 2: Graph Construction

Each sequence is converted to a `torch_geometric.data.Data` object:

- **Nodes**: one per residue; node features = ESM-2 embedding
- **Edges (sequential)**: `i → i+1` and `i+1 → i` — backbone connectivity
- **Edges (k-NN)**: for each residue, the *k* nearest neighbours by Euclidean distance in embedding space are connected — approximating functional/spatial proximity

```
Nodes:  16 (for a 16-residue TCR)
Edges:  30 sequential (bidirectional) + ~80 k-NN = ~110 total
```

### Stage 3: GATEncoder

```
Input:   (N, 1280)   # N = total nodes in batch
Linear:  (N, 128)    # project into hidden space
GAT ×3:  (N, 128)    # multi-head attention (4 heads × 32 dims)
         + residual + LayerNorm + ReLU + Dropout(0.2)
Pool:    (B, 128)    # global mean pooling → graph-level embedding
```

**Why GAT over GCN?**
- Attention weights are interpretable: they show which residues the model focuses on
- Multi-head attention captures different interaction types (electrostatic, hydrophobic, etc.)
- Better performance on heterophilic graphs with diverse node types

### Stage 4: Fusion MLP

```
concat([TCR_repr, Peptide_repr]):  (B, 256)
Linear(256→128) + ReLU + Dropout(0.3)
Linear(128→64)  + ReLU + Dropout(0.2)
Linear(64→1)    + Sigmoid
Output:  (B, 1)   # binding probability ∈ [0, 1]
```

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Loss | Binary cross-entropy |
| Batch size | 32 (GPU) / 8 (CPU demo) |
| Max epochs | 50–100 |
| Early stopping patience | 10 |
| Gradient clipping | max_norm = 1.0 |

---

## Ablation Results

| Variant | ROC-AUC | Δ |
|---|---|---|
| Full model (GNN + ESM-2) | 0.852 | — |
| No k-NN edges (sequential only) | 0.782 | −0.070 |
| No GNN (pooled pLM embeddings only) | 0.756 | −0.096 |
| Random embeddings (no pLM) | 0.651 | −0.201 |
| Random baseline | 0.500 | −0.352 |

These results confirm that both the pretrained embeddings and the graph structure contribute meaningfully to performance.

---

## Computational Requirements

| Resource | Demo | Production |
|---|---|---|
| GPU memory | 4 GB | 16 GB |
| Training time | ~5 min | ~2 h |
| Inference (single pair) | <10 ms | <10 ms |
| ESM-2 model size | 140 MB (35M) | 2.5 GB (650M) |
