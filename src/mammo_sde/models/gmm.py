"""Gaussian Mixture Model from scratch in PyTorch.

Implements the EM algorithm for fitting a GMM to data, with full / diag /
spherical / tied covariance options, BIC/AIC for model selection, and
sampling from the fitted mixture.

Uses log-sum-exp throughout for numerical stability. GPU-friendly.

Reference: Bishop, "Pattern Recognition and Machine Learning", Chapter 9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch

from mammo_sde.models.kmeans_pp import kmeans_pp_init

CovType = Literal["full", "diag", "spherical", "tied"]


@dataclass
class GMMFitHistory:
    log_likelihoods: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0
    final_log_likelihood: float = -float("inf")


class GMM:
    """Gaussian Mixture Model fit by EM.

    Parameters
    ----------
    n_components : int
        Number of mixture components K.
    cov_type : {"full", "diag", "spherical", "tied"}
        Covariance parameterization.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on relative change in log-likelihood.
    reg_covar : float
        Diagonal regularization added to covariance for numerical stability.
    init : {"kmeans++", "random"}
        Initialization strategy.
    seed : int
        Random seed.
    device : str
        Torch device.
    """

    def __init__(
        self,
        n_components: int,
        cov_type: CovType = "diag",
        max_iter: int = 200,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        init: Literal["kmeans++", "random"] = "kmeans++",
        seed: int = 42,
        device: str = "cpu",
    ):
        self.K = int(n_components)
        self.cov_type: CovType = cov_type
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.init = init
        self.seed = int(seed)
        self.device = torch.device(device)

        # Fitted parameters (set by .fit)
        self.weights_: torch.Tensor | None = None  # (K,)
        self.means_: torch.Tensor | None = None  # (K, D)
        self.covariances_: torch.Tensor | None = None
        # shape depends on cov_type:
        #   full       -> (K, D, D)
        #   diag       -> (K, D)
        #   spherical  -> (K,)
        #   tied       -> (D, D)
        self.precisions_cholesky_: torch.Tensor | None = None  # cached for log_prob
        self.D: int | None = None

        self.history_: GMMFitHistory = GMMFitHistory()

    # ----------------------------------------------------------------- public API

    def fit(self, X: torch.Tensor) -> GMM:
        """Fit GMM to X via EM. Returns self."""
        X = X.to(self.device, dtype=torch.float32)
        N, D = X.shape
        self.D = int(D)

        self._initialize(X)
        prev_ll = -float("inf")
        history: list[float] = []

        for it in range(self.max_iter):
            log_resp, ll = self._e_step(X)
            self._m_step(X, log_resp)
            history.append(float(ll))
            if it > 0:
                rel = abs(ll - prev_ll) / (abs(prev_ll) + 1e-12)
                if rel < self.tol:
                    self.history_ = GMMFitHistory(
                        log_likelihoods=history,
                        converged=True,
                        n_iter=it + 1,
                        final_log_likelihood=float(ll),
                    )
                    return self
            prev_ll = ll

        self.history_ = GMMFitHistory(
            log_likelihoods=history,
            converged=False,
            n_iter=self.max_iter,
            final_log_likelihood=float(prev_ll),
        )
        return self

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return responsibilities r_nk = p(component k | x_n), shape (N, K)."""
        X = X.to(self.device, dtype=torch.float32)
        log_resp, _ = self._e_step(X)
        return log_resp.exp()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Hard cluster assignments, shape (N,)."""
        return self.predict_proba(X).argmax(dim=1)

    def score(self, X: torch.Tensor) -> float:
        """Mean log-likelihood per sample under the fitted model."""
        X = X.to(self.device, dtype=torch.float32)
        _, ll = self._e_step(X)
        return float(ll / X.shape[0])

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        """Per-sample log p(x), shape (N,)."""
        X = X.to(self.device, dtype=torch.float32)
        weighted = self._weighted_log_prob(X)  # (N, K)
        return torch.logsumexp(weighted, dim=1)

    def sample(self, n: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw n samples from the fitted mixture. Returns (samples, component_assignments)."""
        if self.means_ is None:
            raise RuntimeError("GMM not fitted yet.")
        g = torch.Generator(device="cpu").manual_seed(self.seed if seed is None else seed)
        weights_cpu = self.weights_.detach().cpu()
        comps = torch.multinomial(weights_cpu, n, replacement=True, generator=g).to(self.device)
        samples = torch.empty(n, self.D, device=self.device, dtype=torch.float32)
        for k in range(self.K):
            mask = comps == k
            if not mask.any():
                continue
            nk = int(mask.sum().item())
            chol = self._component_cov_chol(k)  # (D, D)
            eps = torch.randn(nk, self.D, device=self.device, dtype=torch.float32)
            samples[mask] = self.means_[k] + eps @ chol.T
        return samples, comps

    def n_parameters(self) -> int:
        """Number of free parameters (used for BIC/AIC)."""
        K, D = self.K, self.D
        n_weights = K - 1  # weights sum to 1
        n_means = K * D
        if self.cov_type == "full":
            n_cov = K * D * (D + 1) // 2
        elif self.cov_type == "diag":
            n_cov = K * D
        elif self.cov_type == "spherical":
            n_cov = K
        elif self.cov_type == "tied":
            n_cov = D * (D + 1) // 2
        else:
            raise ValueError(f"Unknown cov_type {self.cov_type}")
        return n_weights + n_means + n_cov

    def bic(self, X: torch.Tensor) -> float:
        """Bayesian Information Criterion: -2 log L + p log N. Lower is better."""
        X = X.to(self.device, dtype=torch.float32)
        N = X.shape[0]
        _, ll = self._e_step(X)
        return float(-2.0 * ll + self.n_parameters() * math.log(N))

    def aic(self, X: torch.Tensor) -> float:
        """Akaike Information Criterion: -2 log L + 2p. Lower is better."""
        X = X.to(self.device, dtype=torch.float32)
        _, ll = self._e_step(X)
        return float(-2.0 * ll + 2.0 * self.n_parameters())

    def state_dict(self) -> dict:
        return {
            "n_components": self.K,
            "cov_type": self.cov_type,
            "embedding_dim": self.D,
            "weights": self.weights_.detach().cpu() if self.weights_ is not None else None,
            "means": self.means_.detach().cpu() if self.means_ is not None else None,
            "covariances": self.covariances_.detach().cpu() if self.covariances_ is not None else None,
            "history": {
                "log_likelihoods": self.history_.log_likelihoods,
                "converged": self.history_.converged,
                "n_iter": self.history_.n_iter,
                "final_log_likelihood": self.history_.final_log_likelihood,
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict, device: str = "cpu") -> GMM:
        gmm = cls(
            n_components=state["n_components"],
            cov_type=state["cov_type"],
            device=device,
        )
        gmm.D = int(state["embedding_dim"])
        gmm.weights_ = state["weights"].to(device) if state["weights"] is not None else None
        gmm.means_ = state["means"].to(device) if state["means"] is not None else None
        gmm.covariances_ = state["covariances"].to(device) if state["covariances"] is not None else None
        h = state.get("history", {})
        gmm.history_ = GMMFitHistory(
            log_likelihoods=h.get("log_likelihoods", []),
            converged=h.get("converged", False),
            n_iter=h.get("n_iter", 0),
            final_log_likelihood=h.get("final_log_likelihood", -float("inf")),
        )
        gmm._refresh_precisions_cholesky()
        return gmm

    # ----------------------------------------------------------------- internals

    def _initialize(self, X: torch.Tensor) -> None:
        N, D = X.shape
        if self.init == "kmeans++":
            self.means_ = kmeans_pp_init(X, self.K, seed=self.seed)
        else:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            idx = torch.randperm(N, generator=g)[: self.K]
            self.means_ = X[idx].clone()

        self.weights_ = torch.full((self.K,), 1.0 / self.K, device=self.device, dtype=torch.float32)

        var = X.var(dim=0, unbiased=False) + self.reg_covar  # (D,)
        if self.cov_type == "full":
            cov = torch.eye(D, device=self.device, dtype=torch.float32) * var.mean()
            self.covariances_ = cov.unsqueeze(0).expand(self.K, D, D).clone()
        elif self.cov_type == "diag":
            self.covariances_ = var.unsqueeze(0).expand(self.K, D).clone()
        elif self.cov_type == "spherical":
            self.covariances_ = torch.full((self.K,), float(var.mean().item()), device=self.device, dtype=torch.float32)
        elif self.cov_type == "tied":
            self.covariances_ = torch.diag(var)
        else:
            raise ValueError(f"Unknown cov_type {self.cov_type}")

        self._refresh_precisions_cholesky()

    def _refresh_precisions_cholesky(self) -> None:
        """Compute Cholesky factors of the precision matrix(es) for fast log-prob."""
        if self.cov_type == "full":
            K, D, _ = self.covariances_.shape
            chol = torch.empty_like(self.covariances_)
            for k in range(K):
                cov_k = self.covariances_[k] + self.reg_covar * torch.eye(D, device=self.device)
                L = torch.linalg.cholesky(cov_k)  # cov = L L^T
                chol[k] = torch.linalg.solve_triangular(L, torch.eye(D, device=self.device), upper=False).T
            self.precisions_cholesky_ = chol
        elif self.cov_type == "diag":
            self.precisions_cholesky_ = 1.0 / (self.covariances_.sqrt() + 1e-12)
        elif self.cov_type == "spherical":
            self.precisions_cholesky_ = 1.0 / (self.covariances_.sqrt() + 1e-12)
        elif self.cov_type == "tied":
            D = self.covariances_.shape[0]
            cov = self.covariances_ + self.reg_covar * torch.eye(D, device=self.device)
            L = torch.linalg.cholesky(cov)
            self.precisions_cholesky_ = torch.linalg.solve_triangular(
                L, torch.eye(D, device=self.device), upper=False
            ).T

    def _log_prob_per_component(self, X: torch.Tensor) -> torch.Tensor:
        """Compute log N(x_n | μ_k, Σ_k) for each n, k. Returns (N, K)."""
        N, D = X.shape
        K = self.K
        log_two_pi = math.log(2.0 * math.pi)

        if self.cov_type == "full":
            # log_det(Σ) = -2 sum(log diag(precisions_cholesky_)) ... but precisions_cholesky_ here is L^{-T}
            # Standard approach: y_k = (x - μ_k) @ precisions_cholesky_k → (N, D)
            log_det = self.precisions_cholesky_.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # (K,)
            out = torch.empty(N, K, device=self.device, dtype=torch.float32)
            for k in range(K):
                diff = X - self.means_[k]  # (N, D)
                y = diff @ self.precisions_cholesky_[k]  # (N, D)
                quad = (y**2).sum(dim=1)  # (N,)
                out[:, k] = -0.5 * (D * log_two_pi + quad) + log_det[k]
            return out

        if self.cov_type == "diag":
            # precisions_cholesky_ is 1/sqrt(σ²_kd), shape (K, D)
            log_det = self.precisions_cholesky_.log().sum(dim=1)  # (K,)
            diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (N, K, D)
            quad = ((diff * self.precisions_cholesky_.unsqueeze(0)) ** 2).sum(dim=2)  # (N, K)
            return -0.5 * (D * log_two_pi + quad) + log_det.unsqueeze(0)

        if self.cov_type == "spherical":
            # precisions_cholesky_ is 1/sqrt(σ²_k), shape (K,)
            log_det = self.precisions_cholesky_.log() * D  # (K,)
            diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (N, K, D)
            quad = (diff**2).sum(dim=2) * (self.precisions_cholesky_**2).unsqueeze(0)  # (N, K)
            return -0.5 * (D * log_two_pi + quad) + log_det.unsqueeze(0)

        if self.cov_type == "tied":
            log_det = self.precisions_cholesky_.diagonal().log().sum()
            out = torch.empty(N, K, device=self.device, dtype=torch.float32)
            for k in range(K):
                diff = X - self.means_[k]
                y = diff @ self.precisions_cholesky_
                quad = (y**2).sum(dim=1)
                out[:, k] = -0.5 * (D * log_two_pi + quad) + log_det
            return out

        raise ValueError(f"Unknown cov_type {self.cov_type}")

    def _weighted_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """log π_k + log N(x | μ_k, Σ_k), shape (N, K)."""
        return self._log_prob_per_component(X) + self.weights_.log().unsqueeze(0)

    def _e_step(self, X: torch.Tensor) -> tuple[torch.Tensor, float]:
        weighted = self._weighted_log_prob(X)  # (N, K)
        log_norm = torch.logsumexp(weighted, dim=1)  # (N,)
        log_resp = weighted - log_norm.unsqueeze(1)
        total_ll = float(log_norm.sum().item())
        return log_resp, total_ll

    def _m_step(self, X: torch.Tensor, log_resp: torch.Tensor) -> None:
        N, D = X.shape
        K = self.K
        resp = log_resp.exp()  # (N, K)
        Nk = resp.sum(dim=0) + 10 * torch.finfo(resp.dtype).eps  # (K,)
        self.weights_ = Nk / N
        self.means_ = (resp.T @ X) / Nk.unsqueeze(1)  # (K, D)

        if self.cov_type == "full":
            new_cov = torch.empty(K, D, D, device=self.device, dtype=torch.float32)
            for k in range(K):
                diff = X - self.means_[k]
                weighted_diff = resp[:, k].unsqueeze(1) * diff
                cov_k = (weighted_diff.T @ diff) / Nk[k]
                cov_k = cov_k + self.reg_covar * torch.eye(D, device=self.device)
                new_cov[k] = cov_k
            self.covariances_ = new_cov

        elif self.cov_type == "diag":
            new_cov = torch.empty(K, D, device=self.device, dtype=torch.float32)
            for k in range(K):
                diff = X - self.means_[k]
                new_cov[k] = (resp[:, k].unsqueeze(1) * diff**2).sum(dim=0) / Nk[k] + self.reg_covar
            self.covariances_ = new_cov

        elif self.cov_type == "spherical":
            new_cov = torch.empty(K, device=self.device, dtype=torch.float32)
            for k in range(K):
                diff = X - self.means_[k]
                new_cov[k] = (resp[:, k].unsqueeze(1) * diff**2).sum() / (Nk[k] * D) + self.reg_covar
            self.covariances_ = new_cov

        elif self.cov_type == "tied":
            cov = torch.zeros(D, D, device=self.device, dtype=torch.float32)
            for k in range(K):
                diff = X - self.means_[k]
                weighted_diff = resp[:, k].unsqueeze(1) * diff
                cov = cov + weighted_diff.T @ diff
            self.covariances_ = cov / N + self.reg_covar * torch.eye(D, device=self.device)

        self._refresh_precisions_cholesky()

    def _component_cov_chol(self, k: int) -> torch.Tensor:
        """Return Cholesky factor of Σ_k for sampling. Shape (D, D)."""
        D = self.D
        if self.cov_type == "full":
            cov = self.covariances_[k] + self.reg_covar * torch.eye(D, device=self.device)
            return torch.linalg.cholesky(cov)
        if self.cov_type == "diag":
            return torch.diag(self.covariances_[k].sqrt())
        if self.cov_type == "spherical":
            return torch.eye(D, device=self.device) * self.covariances_[k].sqrt()
        if self.cov_type == "tied":
            cov = self.covariances_ + self.reg_covar * torch.eye(D, device=self.device)
            return torch.linalg.cholesky(cov)
        raise ValueError(f"Unknown cov_type {self.cov_type}")
