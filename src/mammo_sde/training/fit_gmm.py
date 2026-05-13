"""GMM fitting orchestration: sweep K, pick best by BIC/AIC, refit, save."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from mammo_sde.models.gmm import GMM, CovType
from mammo_sde.utils.io import save_json, save_torch_checkpoint
from mammo_sde.utils.logger import StepLogger


@dataclass
class GMMSweepResult:
    k_values: list[int]
    bic_per_k: list[float]
    aic_per_k: list[float]
    log_likelihood_per_k: list[float]
    n_iter_per_k: list[int]
    converged_per_k: list[bool]
    best_k_bic: int
    best_k_aic: int
    selection_metric: str
    best_k: int
    best_gmm_state: dict = field(default_factory=dict)


def sweep_k(
    X: torch.Tensor,
    k_min: int,
    k_max: int,
    cov_type: CovType = "diag",
    max_iter: int = 200,
    tol: float = 1e-4,
    reg_covar: float = 1e-6,
    seed: int = 42,
    device: str = "cpu",
    selection_metric: Literal["bic", "aic"] = "bic",
    logger: StepLogger | None = None,
) -> GMMSweepResult:
    """Fit GMM for K in [k_min..k_max], record BIC/AIC, pick best, return result."""
    k_values = list(range(k_min, k_max + 1))
    bic_per_k: list[float] = []
    aic_per_k: list[float] = []
    ll_per_k: list[float] = []
    iter_per_k: list[int] = []
    converged_per_k: list[bool] = []
    best_gmm: GMM | None = None
    best_value = float("inf")
    best_k = k_values[0]

    iterator = logger.track(k_values, description=f"K sweep ({cov_type} cov)") if logger else iter(k_values)
    for k in iterator:
        gmm = GMM(
            n_components=k,
            cov_type=cov_type,
            max_iter=max_iter,
            tol=tol,
            reg_covar=reg_covar,
            seed=seed,
            device=device,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        ll = gmm.history_.final_log_likelihood
        bic_per_k.append(bic)
        aic_per_k.append(aic)
        ll_per_k.append(ll)
        iter_per_k.append(gmm.history_.n_iter)
        converged_per_k.append(gmm.history_.converged)
        if logger is not None:
            logger.metric(f"K={k}", f"BIC={bic:.2f}  AIC={aic:.2f}  LL={ll:.2f}  iter={gmm.history_.n_iter}")
        value = bic if selection_metric == "bic" else aic
        if value < best_value:
            best_value = value
            best_gmm = gmm
            best_k = k

    best_k_bic = k_values[int(torch.tensor(bic_per_k).argmin().item())]
    best_k_aic = k_values[int(torch.tensor(aic_per_k).argmin().item())]

    return GMMSweepResult(
        k_values=k_values,
        bic_per_k=bic_per_k,
        aic_per_k=aic_per_k,
        log_likelihood_per_k=ll_per_k,
        n_iter_per_k=iter_per_k,
        converged_per_k=converged_per_k,
        best_k_bic=best_k_bic,
        best_k_aic=best_k_aic,
        selection_metric=selection_metric,
        best_k=best_k,
        best_gmm_state=best_gmm.state_dict() if best_gmm is not None else {},
    )


def save_sweep_result(result: GMMSweepResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "results" / "sweep.json",
        {
            "k_values": result.k_values,
            "bic": result.bic_per_k,
            "aic": result.aic_per_k,
            "log_likelihood": result.log_likelihood_per_k,
            "n_iter": result.n_iter_per_k,
            "converged": result.converged_per_k,
            "best_k_bic": result.best_k_bic,
            "best_k_aic": result.best_k_aic,
            "selection_metric": result.selection_metric,
            "best_k": result.best_k,
        },
    )
    save_torch_checkpoint(output_dir / "checkpoints" / "best_gmm.pt", result.best_gmm_state)
