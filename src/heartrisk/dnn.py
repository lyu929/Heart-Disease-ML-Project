"""Optional PyTorch MLP with focal loss, wrapped as a scikit-learn classifier.

Kept from the original course project (where it was the "Deep Neural Network" model) but
rewritten so it can live inside a Pipeline, be cloned by CV and be pickled into a bundle.
Weights are stored as NumPy arrays, so loading a bundle does not need a GPU.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden=(64, 32),
        dropout=0.2,
        epochs=200,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        focal_gamma=2.0,
        patience=20,
        val_fraction=0.15,
        random_state=0,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.focal_gamma = focal_gamma
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def _module(self, n_in: int):
        import torch.nn as nn

        layers, prev = [], n_in
        for width in self.hidden:
            layers += [nn.Linear(prev, int(width)), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = int(width)
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def _focal(self, logits, target, alpha: float):
        import torch
        import torch.nn.functional as F

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)
        a_t = alpha * target + (1 - alpha) * (1 - target)
        return (a_t * (1 - p_t) ** self.focal_gamma * bce).mean()

    def fit(self, X, y):
        import torch

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        torch.manual_seed(self.random_state)
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=self.val_fraction, stratify=y, random_state=self.random_state
        )
        alpha = float(1.0 - ytr.mean())  # up-weights the minority class
        net = self._module(X.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
        xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva)
        gen = torch.Generator().manual_seed(self.random_state)
        best, best_state, wait, history = np.inf, None, 0, []
        for _ in range(int(self.epochs)):
            net.train()
            perm = torch.randperm(len(xt), generator=gen)
            for i in range(0, len(xt), int(self.batch_size)):
                idx = perm[i : i + int(self.batch_size)]
                opt.zero_grad()
                loss = self._focal(net(xt[idx]).squeeze(1), yt[idx], alpha)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = float(self._focal(net(xv).squeeze(1), yv, alpha))
            history.append(val)
            if val < best - 1e-5:
                best, wait = val, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.loss_history_ = history
        self.weights_ = {k: v.numpy().copy() for k, v in (best_state or net.state_dict()).items()}
        return self

    def predict_proba(self, X):
        import torch

        net = self._module(self.n_features_in_)
        net.load_state_dict({k: torch.from_numpy(v) for k, v in self.weights_.items()})
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(torch.from_numpy(np.asarray(X, dtype=np.float32))).squeeze(1)).numpy()
        return np.column_stack([1 - p, p]).astype(float)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
