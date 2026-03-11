"""
TCR-Peptide Binding Prediction
==============================
A hybrid GNN + protein language model pipeline for predicting TCR–peptide
binding specificity.

Modules
-------
data      : Dataset classes, data loading, negative-sample generation
embedder  : ESM-2 / ProtBERT protein language model wrappers
graph     : Sequence → PyTorch Geometric graph conversion
model     : GATEncoder and TCRPeptideBindingModel architectures
train     : Training loop with early stopping and checkpointing
evaluate  : Metrics, plots, and interpretability utilities
"""
