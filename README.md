# TCR-Peptide Binding Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch_Geometric-2.3+-orange" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

> **End-to-end machine learning pipeline for predicting T-cell receptor (TCR)–peptide binding specificity using hybrid Graph Neural Networks and protein language models.**

---

## Overview

T-cell receptors (TCRs) recognise peptides presented by MHC molecules — a fundamental step in adaptive immunity and the basis of personalised cancer immunotherapy. Computationally predicting which TCR will bind which peptide is extraordinarily challenging: the combined sequence space exceeds 10²⁴ possibilities, yet only tens of thousands of experimentally validated pairs exist.

This project implements a **hybrid deep learning architecture** that combines:

- **ESM-2** (650M parameter protein language model) for rich, context-aware residue embeddings
- **Graph Attention Networks (GAT)** to model spatial and functional residue interactions
- A **fusion MLP** to produce a final binding probability from the paired TCR + peptide representations

The pipeline achieves ~**85% ROC-AUC** on held-out test data, competitive with published methods such as NetTCR-2.1 and ERGO-II.

---

## Key Results

| Method | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| **This work (GNN + ESM-2)** | **0.852** | **0.803** | **0.821** |
| NetTCR-2.1 (LSTM + BLOSUM) | 0.836 | 0.789 | 0.801 |
| ERGO-II (AE + attention) | 0.823 | 0.776 | 0.793 |
| DeepTCR (CNN + embedding) | 0.814 | 0.768 | 0.785 |

---

## Repository Structure

```
tcr-peptide-binding/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb        # EDA on VDJdb / IEDB datasets
│   ├── 02_embeddings_and_graphs.ipynb   # ESM-2 embeddings & graph construction
│   ├── 03_model_training.ipynb          # Full training loop with live plots
│   ├── 04_evaluation_and_results.ipynb  # Metrics, ROC, confusion matrix
│   └── 05_interpretability.ipynb        # Attention heatmaps, UMAP, ablation
│
├── src/
│   ├── __init__.py
│   ├── data.py          # Dataset classes, data loading, negative generation
│   ├── embedder.py      # ESM-2 / ProtBERT protein language model wrappers
│   ├── graph.py         # Sequence → PyTorch Geometric graph conversion
│   ├── model.py         # GATEncoder, TCRPeptideBindingModel
│   ├── train.py         # Training loop, checkpointing, early stopping
│   └── evaluate.py      # Metrics, plots, interpretability utilities
│
│
├── tests/
│   ├── test_data.py
│   ├── test_graph.py
│   └── test_model.py
│
├── docs/
│   └── architecture.md 
│
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md          
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/tcr-peptide-binding.git
cd tcr-peptide-binding

# Option A — pip
pip install -r requirements.txt

# Option B — Conda (recommended)
conda env create -f environment.yml
conda activate tcr-binding
```

### 2. Run the demo pipeline (synthetic data, no downloads needed)

```bash
# Runs the full pipeline in ~5 min on CPU, ~2 min on GPU
cd notebooks
jupyter lab
# Open 03_model_training.ipynb and run all cells
```

### 3. Run with real VDJdb data

```python
# Inside any notebook or script:
from src.data import download_vdjdb, generate_negatives

positives = download_vdjdb()          # ~60k pairs, downloads automatically
dataset   = generate_negatives(positives, ratio=1.0)  # balanced dataset
```

---

## Architecture

```
TCR sequence ──► ESM-2 ──► [L_tcr × 1280] ──► Graph (nodes=residues, edges=seq+kNN)
                                                      │
                                                   3× GAT ──► GlobalMeanPool ──► [128]
                                                                                    │
                                                                              Concat [256]
                                                                                    │
Peptide seq  ──► ESM-2 ──► [L_pep × 1280] ──► Graph ──► 3× GAT ──► [128] ──►  MLP ──► p(bind)
```

**Why this design?**

| Component | Choice | Reason |
|---|---|---|
| Sequence encoder | ESM-2 (650M) | Best protein LM; context-aware residue embeddings |
| Graph edges | Sequential + k-NN | Backbone connectivity + long-range functional similarity |
| GNN type | GAT (4 heads) | Attention weights → interpretable binding residues |
| Pooling | Global mean | Variable-length graphs → fixed 128-dim embeddings |
| Fusion | Concatenation + MLP | Simple, effective; no cross-attention needed for this scale |

---

## Notebooks Guide

| Notebook | What is it about? |
|---|---|
| `01_data_exploration` | Class balance, sequence length distributions, amino acid composition, VDJdb metadata |
| `02_embeddings_and_graphs` | How ESM-2 tokenises proteins, what embeddings capture, how k-NN edges are constructed |
| `03_model_training` | Full training loop, loss curves, early stopping, GPU optimisation |
| `04_evaluation_and_results` | ROC-AUC, F1, precision-recall, confusion matrix, benchmark comparison |
| `05_interpretability` | GAT attention heatmaps, UMAP of embedding space, ablation study, counterfactual mutations |

---

## Datasets

| Dataset | Size | Source |
|---|---|---|
| VDJdb | ~60,000 TCR–peptide pairs | https://vdjdb.cdr3.net |
| IEDB | ~15,000 TCR–pMHC records | https://www.iedb.org |
| NetTCR benchmark | ~5,000 curated pairs | https://services.healthtech.dtu.dk/NetTCR |

> Data files are **not** committed to this repository (see `.gitignore`).  
> Run `src/data.py` or notebook `01` to download and preprocess automatically.

---

## Requirements

```
Python  >= 3.10
PyTorch >= 2.0
torch-geometric >= 2.3
transformers >= 4.30
```

See `requirements.txt` for the full pinned list.

---

## Reproducing Results

All notebooks set a global random seed (`SEED = 42`). To reproduce the benchmark table exactly:

```bash
jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_evaluation_and_results.ipynb
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
