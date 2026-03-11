"""
train.py
========
Training loop for the TCR–peptide binding prediction model.

Provides :class:`TCRBindingTrainer`, which wraps the model with:

- AdamW optimiser with weight decay
- Binary cross-entropy loss
- Epoch-level validation with ROC-AUC scoring
- Automatic checkpointing of the best model weights
- Early stopping to prevent overfitting
- Training-history dict for post-hoc plotting

Usage
-----
>>> from src.model import TCRPeptideBindingModel
>>> from src.train import TCRBindingTrainer
>>> model   = TCRPeptideBindingModel(input_dim=480)
>>> trainer = TCRBindingTrainer(model, device="cuda", lr=1e-4)
>>> best_auc = trainer.train(train_loader, val_loader, num_epochs=50)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class TCRBindingTrainer:
    """Training harness for :class:`src.model.TCRPeptideBindingModel`.

    Parameters
    ----------
    model : nn.Module
        Instantiated binding model.
    device : str, default ``"cuda"``
        Target device.  Falls back to CPU if CUDA is unavailable.
    lr : float, default 1e-4
        Initial learning rate for AdamW.
    weight_decay : float, default 1e-5
        L2 regularisation coefficient for AdamW.
    checkpoint_path : str, default ``"best_tcr_model.pt"``
        File path where the best model weights will be saved.
    patience : int, default 10
        Number of epochs without validation AUC improvement before
        early stopping is triggered.

    Attributes
    ----------
    history : dict
        Keys ``"train_loss"``, ``"val_loss"``, ``"val_auc"`` — each a list
        of per-epoch floats, suitable for plotting.

    Examples
    --------
    >>> trainer = TCRBindingTrainer(model, device="cpu", lr=1e-4)
    >>> best_auc = trainer.train(train_loader, val_loader, num_epochs=20)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        checkpoint_path: str = "best_tcr_model.pt",
        patience: int = 10,
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.checkpoint_path = Path(checkpoint_path)
        self.patience = patience

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
        }

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def train_epoch(self, train_loader) -> float:
        """Run a single training epoch.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader yielding batches with keys ``"tcr_graph"``,
            ``"peptide_graph"``, ``"label"``.

        Returns
        -------
        float
            Mean training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            tcr_batch = batch["tcr_graph"].to(self.device)
            peptide_batch = batch["peptide_graph"].to(self.device)
            labels = batch["label"].to(self.device)

            predictions = self.model(tcr_batch, peptide_batch).squeeze()
            loss = self.criterion(predictions, labels)

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(
        self, val_loader
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate the model on a validation / test DataLoader.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader

        Returns
        -------
        avg_loss : float
        auc : float
        all_preds : np.ndarray  shape ``(n_samples,)``
        all_labels : np.ndarray  shape ``(n_samples,)``
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: list[float] = []
        all_labels: list[float] = []

        for batch in val_loader:
            tcr_batch = batch["tcr_graph"].to(self.device)
            peptide_batch = batch["peptide_graph"].to(self.device)
            labels = batch["label"].to(self.device)

            predictions = self.model(tcr_batch, peptide_batch).squeeze()
            loss = self.criterion(predictions, labels)
            total_loss += loss.item()

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        auc = roc_auc_score(all_labels, all_preds)
        return avg_loss, auc, np.array(all_preds), np.array(all_labels)

    # ------------------------------------------------------------------

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 50,
        verbose: bool = True,
    ) -> float:
        """Run the full training loop with early stopping.

        Saves the best model weights (highest validation AUC) to
        ``self.checkpoint_path`` automatically.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        num_epochs : int, default 50
            Maximum number of training epochs.
        verbose : bool, default True
            Print per-epoch metrics every 5 epochs.

        Returns
        -------
        float
            Best validation ROC-AUC achieved during training.
        """
        best_auc = 0.0
        epochs_without_improvement = 0

        if verbose:
            print("Starting training …")
            print(f"  Training samples  : {len(train_loader.dataset)}")
            print(f"  Validation samples: {len(val_loader.dataset)}")
            print(f"  Max epochs        : {num_epochs}")
            print(f"  Early-stop patience: {self.patience}")
            print("-" * 60)

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc, _, _ = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)

            # Checkpoint best model
            if val_auc > best_auc:
                best_auc = val_auc
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                logger.debug("Saved checkpoint (AUC %.4f) to %s", best_auc, self.checkpoint_path)
            else:
                epochs_without_improvement += 1

            if verbose and (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1:>3}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_auc:.4f} | "
                    f"Best AUC: {best_auc:.4f}"
                )

            # Early stopping
            if epochs_without_improvement >= self.patience:
                if verbose:
                    print(
                        f"\nEarly stopping at epoch {epoch + 1} "
                        f"(no improvement for {self.patience} epochs)."
                    )
                break

        if verbose:
            print("-" * 60)
            print(f"Training complete.  Best validation AUC: {best_auc:.4f}")

        return best_auc

    # ------------------------------------------------------------------

    def load_best_model(self) -> None:
        """Load the best checkpoint weights back into ``self.model``."""
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        logger.info("Loaded best model from %s", self.checkpoint_path)
