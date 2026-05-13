"""VAE training loop — works for vanilla, σ-VAE, and VampPrior variants."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mammo_sde.models.base_vae import BaseVAE
from mammo_sde.utils.io import save_json, save_torch_checkpoint
from mammo_sde.utils.logger import StepLogger


@dataclass
class TrainHistory:
    train_total: list[float] = field(default_factory=list)
    train_recon: list[float] = field(default_factory=list)
    train_kl: list[float] = field(default_factory=list)
    val_total: list[float] = field(default_factory=list)
    val_recon: list[float] = field(default_factory=list)
    val_kl: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)


def train_vae(
    model: BaseVAE,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    beta_schedule: Callable[[int], float] | None = None,
    grad_clip: float = 1.0,
    device: str = "cpu",
    logger: StepLogger | None = None,
    log_every: int = 1,
) -> TrainHistory:
    """Train a VAE (any variant) and return loss history.

    Parameters
    ----------
    model : BaseVAE
        Model to train.
    beta_schedule : Optional[Callable]
        Function epoch → β (KL weight). Default: linear warmup 0 → 1 over first
        20 epochs, then constant 1. Useful to avoid posterior collapse.
    """
    if beta_schedule is None:

        def beta_schedule(epoch: int) -> float:
            return min(1.0, (epoch + 1) / 20.0)

    device_t = torch.device(device)
    model.to(device_t)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = TrainHistory()

    for epoch in range(epochs):
        import time

        t0 = time.time()
        beta = float(beta_schedule(epoch))
        model.train()
        sum_total = sum_recon = sum_kl = 0.0
        n_seen = 0
        for batch in train_loader:
            x, _ = batch
            x = x.to(device_t)
            optimizer.zero_grad()
            out = model.loss(x, beta=beta)
            out.total.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            bs = x.size(0)
            sum_total += float(out.total.item()) * bs
            sum_recon += float(out.recon.item()) * bs
            sum_kl += float(out.kl.item()) * bs
            n_seen += bs
        train_total = sum_total / n_seen
        train_recon = sum_recon / n_seen
        train_kl = sum_kl / n_seen
        history.train_total.append(train_total)
        history.train_recon.append(train_recon)
        history.train_kl.append(train_kl)

        if val_loader is not None:
            model.eval()
            v_total = v_recon = v_kl = 0.0
            n_v = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, _ = batch
                    x = x.to(device_t)
                    out = model.loss(x, beta=beta)
                    bs = x.size(0)
                    v_total += float(out.total.item()) * bs
                    v_recon += float(out.recon.item()) * bs
                    v_kl += float(out.kl.item()) * bs
                    n_v += bs
            val_total = v_total / n_v
            val_recon = v_recon / n_v
            val_kl = v_kl / n_v
            history.val_total.append(val_total)
            history.val_recon.append(val_recon)
            history.val_kl.append(val_kl)
        else:
            val_total = val_recon = val_kl = float("nan")

        history.epoch_times.append(time.time() - t0)

        if logger is not None and (epoch % log_every == 0 or epoch == epochs - 1):
            logger.metric(
                f"Epoch {epoch+1}/{epochs}",
                f"β={beta:.3f}  train: tot={train_total:.3f} recon={train_recon:.3f} KL={train_kl:.3f}  "
                f"val: tot={val_total:.3f} recon={val_recon:.3f} KL={val_kl:.3f}",
            )

    return history


def save_vae_run(
    model: BaseVAE,
    history: TrainHistory,
    config: dict,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    save_torch_checkpoint(
        output_dir / "checkpoints" / "final_model.pt",
        {"state_dict": model.state_dict(), "config": config},
    )
    save_json(
        output_dir / "results" / "history.json",
        {
            "train_total": history.train_total,
            "train_recon": history.train_recon,
            "train_kl": history.train_kl,
            "val_total": history.val_total,
            "val_recon": history.val_recon,
            "val_kl": history.val_kl,
            "epoch_times": history.epoch_times,
        },
    )
    save_json(output_dir / "results" / "config.json", config)
