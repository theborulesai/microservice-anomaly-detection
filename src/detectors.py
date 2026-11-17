"""
detectors.py
============
Unsupervised anomaly detectors for microservice telemetry.

Implements six detectors, each exposing a common interface:
  .fit(X_train)      – train on normal data only
  .score(X_test)     – return anomaly scores (higher = more anomalous)
  .predict(X_test, threshold) – binary predictions

Detectors
---------
1. IsolationForest       (IF)    – tree-based, O(n log n)
2. LocalOutlierFactor    (LOF)   – density-based, kNN
3. OneClassSVM           (OCSVM) – kernel SVM boundary
4. AutoencoderDetector   (AE)    – reconstruction-error MLP
5. LSTMAutoencoder       (LSTM)  – sequence reconstruction (needs 3-D input)
6. EnsembleDetector      (ENS)   – rank-averaging ensemble of IF+LOF+AE
"""

import numpy as np
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore")


# ═══════════════════════════ Base Interface ══════════════════════════════════

class BaseDetector(ABC):
    """Common interface for all anomaly detectors."""

    def __init__(self, name: str):
        self.name = name
        self.scaler = StandardScaler()
        self._fitted = False

    @abstractmethod
    def _fit_impl(self, X: np.ndarray) -> None: ...

    @abstractmethod
    def _score_impl(self, X: np.ndarray) -> np.ndarray: ...

    def fit(self, X: np.ndarray) -> "BaseDetector":
        X_s = self.scaler.fit_transform(X)
        self._fit_impl(X_s)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        X_s = self.scaler.transform(X)
        return self._score_impl(X_s)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        scores = self.score(X)
        return (scores >= threshold).astype(int)

    def auto_threshold(self, X_val: np.ndarray,
                        contamination: float = 0.05) -> float:
        """Pick threshold as (1-contamination) quantile of val scores."""
        scores = self.score(X_val)
        return float(np.quantile(scores, 1 - contamination))


# ═══════════════════════════ 1. Isolation Forest ════════════════════════════

class IsolationForestDetector(BaseDetector):
    """
    Extended Isolation Forest.
    Anomaly score = − decision_function (flipped so higher = more anomalous).
    """
    def __init__(self, n_estimators: int = 200, contamination: float = 0.05,
                 random_state: int = 42):
        super().__init__("IsolationForest")
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def _fit_impl(self, X: np.ndarray) -> None:
        self.model.fit(X)

    def _score_impl(self, X: np.ndarray) -> np.ndarray:
        # decision_function: negative = anomalous → flip sign
        return -self.model.decision_function(X)


# ═══════════════════════════ 2. Local Outlier Factor ════════════════════════

class LOFDetector(BaseDetector):
    """
    Local Outlier Factor (novelty=True for out-of-sample scoring).
    """
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        super().__init__("LOF")
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            contamination=contamination,
            n_jobs=-1,
        )

    def _fit_impl(self, X: np.ndarray) -> None:
        self.model.fit(X)

    def _score_impl(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)


# ═══════════════════════════ 3. One-Class SVM ════════════════════════════════

class OCSVMDetector(BaseDetector):
    """
    One-Class SVM with RBF kernel.
    """
    def __init__(self, nu: float = 0.05, kernel: str = "rbf", gamma: str = "scale"):
        super().__init__("OneClassSVM")
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def _fit_impl(self, X: np.ndarray) -> None:
        self.model.fit(X)

    def _score_impl(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)


# ═══════════════════════════ 4. Autoencoder (MLP) ════════════════════════════

class AutoencoderDetector(BaseDetector):
    """
    Fully-connected autoencoder trained with MSE reconstruction loss.
    Anomaly score = mean squared reconstruction error.
    """
    def __init__(self, hidden_dims=None, latent_dim: int = 8,
                 epochs: int = 50, batch_size: int = 256,
                 lr: float = 1e-3, random_state: int = 42):
        super().__init__("Autoencoder")
        self.hidden_dims = hidden_dims or [64, 32]
        self.latent_dim  = latent_dim
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.random_state= random_state
        self._ae = None

    def _build_model(self, input_dim: int):
        try:
            import torch
            import torch.nn as nn

            class AE(nn.Module):
                def __init__(self, in_d, hidden, lat):
                    super().__init__()
                    enc_layers, dec_layers = [], []
                    prev = in_d
                    for h in hidden:
                        enc_layers += [nn.Linear(prev, h), nn.ReLU()]
                        prev = h
                    enc_layers += [nn.Linear(prev, lat)]
                    self.encoder = nn.Sequential(*enc_layers)

                    prev = lat
                    hidden_rev = list(reversed(hidden))
                    for h in hidden_rev:
                        dec_layers += [nn.Linear(prev, h), nn.ReLU()]
                        prev = h
                    dec_layers += [nn.Linear(prev, in_d)]
                    self.decoder = nn.Sequential(*dec_layers)

                def forward(self, x):
                    return self.decoder(self.encoder(x))

            return AE(input_dim, self.hidden_dims, self.latent_dim), "torch"
        except ImportError:
            return None, "numpy"

    def _fit_impl(self, X: np.ndarray) -> None:
        input_dim = X.shape[1]
        model_obj, backend = self._build_model(input_dim)

        if backend == "torch":
            import torch
            import torch.nn as nn
            torch.manual_seed(self.random_state)
            device = torch.device("cpu")
            model_obj.to(device)
            opt = torch.optim.Adam(model_obj.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            X_t = torch.FloatTensor(X)
            dataset = torch.utils.data.TensorDataset(X_t)
            loader  = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True)

            model_obj.train()
            for epoch in range(self.epochs):
                for (batch,) in loader:
                    opt.zero_grad()
                    recon = model_obj(batch)
                    loss  = criterion(recon, batch)
                    loss.backward()
                    opt.step()
            self._ae = model_obj
            self._backend = "torch"
        else:
            # Fallback: PCA-based reconstruction
            from sklearn.decomposition import PCA
            self._pca = PCA(n_components=min(self.latent_dim, X.shape[1]))
            self._pca.fit(X)
            self._backend = "pca"

    def _score_impl(self, X: np.ndarray) -> np.ndarray:
        if self._backend == "torch":
            import torch
            self._ae.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X)
                recon = self._ae(X_t).numpy()
        else:
            recon = self._pca.inverse_transform(self._pca.transform(X))
        return np.mean((X - recon) ** 2, axis=1)


# ═══════════════════════════ 5. LSTM Autoencoder ════════════════════════════

class LSTMAutoencoderDetector:
    """
    LSTM-based sequence autoencoder for windowed telemetry.
    Input shape: (N, window_size, n_features).
    Anomaly score = mean reconstruction error per sequence.
    """
    def __init__(self, hidden_size: int = 32, num_layers: int = 2,
                 epochs: int = 30, batch_size: int = 64,
                 lr: float = 1e-3, random_state: int = 42):
        self.name = "LSTM-AE"
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.random_state= random_state
        self.scaler      = StandardScaler()
        self._fitted     = False
        self._model      = None
        self._backend    = None

    def _build_lstm(self, n_features: int):
        try:
            import torch
            import torch.nn as nn

            class LSTMAutoEncoder(nn.Module):
                def __init__(self, n_feat, hidden, n_layers):
                    super().__init__()
                    self.encoder = nn.LSTM(n_feat, hidden, n_layers,
                                           batch_first=True)
                    self.decoder = nn.LSTM(hidden, hidden, n_layers,
                                           batch_first=True)
                    self.output  = nn.Linear(hidden, n_feat)

                def forward(self, x):
                    # Encode
                    _, (h, c) = self.encoder(x)
                    # Decode: repeat last hidden state
                    seq_len = x.size(1)
                    decoder_in = h[-1].unsqueeze(1).repeat(1, seq_len, 1)
                    out, _ = self.decoder(decoder_in)
                    return self.output(out)

            return LSTMAutoEncoder(n_features, self.hidden_size, self.num_layers)
        except ImportError:
            return None

    def fit(self, X: np.ndarray) -> "LSTMAutoencoderDetector":
        """X shape: (N, T, F)"""
        N, T, F = X.shape
        X_2d = X.reshape(-1, F)
        self.scaler.fit(X_2d)
        X_s = self.scaler.transform(X_2d).reshape(N, T, F)

        model = self._build_lstm(F)
        if model is not None:
            import torch
            import torch.nn as nn
            torch.manual_seed(self.random_state)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            X_t = torch.FloatTensor(X_s)
            dataset = torch.utils.data.TensorDataset(X_t)
            loader  = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True)

            model.train()
            for epoch in range(self.epochs):
                for (batch,) in loader:
                    opt.zero_grad()
                    recon = model(batch)
                    loss  = criterion(recon, batch)
                    loss.backward()
                    opt.step()
            self._model   = model
            self._backend = "torch"
        else:
            # PCA fallback
            from sklearn.decomposition import PCA
            self._pca     = PCA(n_components=min(16, F))
            self._pca.fit(X_s.reshape(-1, F))
            self._backend = "pca"
            self._T, self._F = T, F

        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        N, T, F = X.shape
        X_2d = X.reshape(-1, F)
        X_s  = self.scaler.transform(X_2d).reshape(N, T, F)

        if self._backend == "torch":
            import torch
            self._model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_s)
                recon = self._model(X_t).numpy()
        else:
            recon = self._pca.inverse_transform(
                self._pca.transform(X_s.reshape(-1, F))).reshape(N, T, F)

        return np.mean((X_s - recon) ** 2, axis=(1, 2))

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)

    def auto_threshold(self, X_val: np.ndarray,
                        contamination: float = 0.05) -> float:
        scores = self.score(X_val)
        return float(np.quantile(scores, 1 - contamination))


# ═══════════════════════════ 6. Ensemble Detector ═══════════════════════════

class EnsembleDetector:
    """
    Rank-averaging ensemble of IsolationForest + LOF + Autoencoder.
    Each detector's scores are rank-normalised [0,1], then averaged.
    """
    def __init__(self):
        self.name = "Ensemble"
        self.detectors = [
            IsolationForestDetector(n_estimators=200),
            LOFDetector(n_neighbors=20),
            AutoencoderDetector(epochs=50),
        ]
        self._fitted = False

    def _rank_normalize(self, scores: np.ndarray) -> np.ndarray:
        from scipy.stats import rankdata
        return rankdata(scores) / len(scores)

    def fit(self, X: np.ndarray) -> "EnsembleDetector":
        for d in self.detectors:
            d.fit(X)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted
        all_scores = np.stack(
            [self._rank_normalize(d.score(X)) for d in self.detectors],
            axis=1
        )
        return all_scores.mean(axis=1)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)

    def auto_threshold(self, X_val: np.ndarray,
                        contamination: float = 0.05) -> float:
        scores = self.score(X_val)
        return float(np.quantile(scores, 1 - contamination))


# ═══════════════════════════ Registry ════════════════════════════════════════

def get_tabular_detectors():
    return [
        IsolationForestDetector(),
        LOFDetector(),
        OCSVMDetector(),
        AutoencoderDetector(epochs=50),
        EnsembleDetector(),
    ]


def get_sequence_detector():
    return LSTMAutoencoderDetector(epochs=30)
