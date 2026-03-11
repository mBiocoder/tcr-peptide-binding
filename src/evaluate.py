"""
evaluate.py
===========
Model evaluation, visualisation, and interpretability utilities.

Functions
---------
evaluate_model
    Compute ROC-AUC, accuracy, and F1 on a DataLoader.
plot_training_curves
    Plot training/validation loss and validation AUC over epochs.
plot_roc_curve
    Plot the ROC curve with AUC annotation.
plot_confusion_matrix
    Heatmap of true-positive / false-positive rates.
plot_attention_heatmap
    Visualise GAT attention weights for a single TCR–peptide pair.
plot_embedding_umap
    UMAP projection of graph-level embeddings coloured by binding label.
run_ablation_study
    Measure AUC impact of removing graph structure or pretrained embeddings.

Examples
--------
>>> from src.evaluate import evaluate_model
>>> results = evaluate_model(model, test_loader, device="cuda")
>>> print(results['auc'])
0.852
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Publication-quality style
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: str = "cuda",
    threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Run model inference over a DataLoader and compute evaluation metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Trained binding model in eval mode.
    test_loader : DataLoader
        Yields batches with keys ``"tcr_graph"``, ``"peptide_graph"``,
        ``"label"``.
    device : str, default ``"cuda"``
    threshold : float, default 0.5
        Probability threshold for converting predictions to binary labels.
    verbose : bool, default True
        Print the metrics table to stdout.

    Returns
    -------
    dict with keys:
        - ``"auc"``         : float
        - ``"accuracy"``    : float
        - ``"f1"``          : float
        - ``"predictions"`` : np.ndarray of shape (n,)
        - ``"labels"``      : np.ndarray of shape (n,)
    """
    model.eval()
    all_preds: list[float] = []
    all_labels: list[float] = []
    device = device if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for batch in test_loader:
            tcr_batch = batch["tcr_graph"].to(device)
            peptide_batch = batch["peptide_graph"].to(device)
            labels = batch["label"]

            preds = model(tcr_batch, peptide_batch).squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    binary_preds = (preds_arr >= threshold).astype(int)

    auc = roc_auc_score(labels_arr, preds_arr)
    acc = accuracy_score(labels_arr, binary_preds)
    f1 = f1_score(labels_arr, binary_preds)

    if verbose:
        print("\n" + "=" * 50)
        print("  TEST SET EVALUATION")
        print("=" * 50)
        print(f"  ROC-AUC   : {auc:.4f}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  Threshold : {threshold}")
        print("=" * 50 + "\n")

    return {
        "auc": auc,
        "accuracy": acc,
        "f1": f1,
        "predictions": preds_arr,
        "labels": labels_arr,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training/validation loss and validation AUC vs. epoch.

    Parameters
    ----------
    history : dict
        Dictionary with keys ``"train_loss"``, ``"val_loss"``, ``"val_auc"``
        as returned by :class:`src.train.TCRBindingTrainer`.
    save_path : str or None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss panel
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(
        epochs, history["val_loss"], label="Validation", linewidth=2, linestyle="--"
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Training and Validation Loss", fontweight="bold")
    axes[0].legend()

    # AUC panel
    axes[1].plot(
        epochs, history["val_auc"], color="#2ca02c", label="Val AUC", linewidth=2
    )
    axes[1].axhline(
        y=max(history["val_auc"]),
        color="gray",
        linestyle=":",
        linewidth=1,
        label=f"Best AUC = {max(history['val_auc']):.3f}",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("Validation ROC-AUC", fontweight="bold")
    axes[1].legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved training curves to %s", save_path)

    return fig


def plot_roc_curve(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the ROC curve with AUC annotation.

    Parameters
    ----------
    labels : np.ndarray  — true binary labels
    predictions : np.ndarray  — predicted probabilities
    save_path : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    fpr, tpr, _ = roc_curve(labels, predictions)
    auc = roc_auc_score(labels, predictions)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved ROC curve to %s", save_path)

    return fig


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    labels : np.ndarray
    predictions : np.ndarray  — predicted probabilities
    threshold : float, default 0.5
    save_path : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    binary_preds = (predictions >= threshold).astype(int)
    cm = confusion_matrix(labels, binary_preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-binding", "Binding"],
        yticklabels=["Non-binding", "Binding"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved confusion matrix to %s", save_path)

    return fig


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tcr_sequence: str,
    peptide_sequence: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualise per-residue GAT attention weights for a TCR–peptide pair.

    Parameters
    ----------
    attention_weights : np.ndarray
        Shape ``(len(tcr), len(peptide))`` — cross-attention between TCR
        and peptide residues.  Can be approximated by taking the mean
        attention across all edges incident to each TCR node.
    tcr_sequence : str
    peptide_sequence : str
    save_path : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(max(6, len(peptide_sequence)), max(4, len(tcr_sequence) * 0.5)))
    sns.heatmap(
        attention_weights,
        xticklabels=list(peptide_sequence),
        yticklabels=list(tcr_sequence),
        cmap="RdYlGn",
        ax=ax,
        cbar_kws={"label": "Attention weight"},
        annot=len(tcr_sequence) <= 20,
        fmt=".2f",
    )
    ax.set_xlabel("Peptide residues", fontsize=12)
    ax.set_ylabel("TCR CDR3 residues", fontsize=12)
    ax.set_title(
        f"TCR–Peptide Predicted Contact Map\n"
        f"TCR: {tcr_sequence}  |  Peptide: {peptide_sequence}",
        fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved attention heatmap to %s", save_path)

    return fig


def plot_embedding_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Project graph-level embeddings to 2-D via UMAP and colour by binding label.

    Parameters
    ----------
    embeddings : np.ndarray  shape ``(n_samples, hidden_dim)``
    labels : np.ndarray  shape ``(n_samples,)``  — binary 0/1
    save_path : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise ImportError("Install umap-learn: pip install umap-learn") from exc

    reducer = umap.UMAP(n_components=2, random_state=42)
    projected = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    colours = {0: "#d62728", 1: "#1f77b4"}
    label_names = {0: "Non-binding", 1: "Binding"}

    for lbl in [0, 1]:
        mask = labels == lbl
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=colours[lbl],
            label=label_names[lbl],
            alpha=0.6,
            s=20,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of Graph-Level Embeddings", fontweight="bold")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved UMAP plot to %s", save_path)

    return fig
