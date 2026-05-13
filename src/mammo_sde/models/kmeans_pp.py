"""K-means++ initialization (Arthur & Vassilvitskii 2007) from scratch in PyTorch.

Used to initialize the GMM EM algorithm to good starting cluster centers.
"""

from __future__ import annotations

import torch


def kmeans_pp_init(X: torch.Tensor, k: int, seed: int = 42) -> torch.Tensor:
    """K-means++ initialization for k cluster centers.

    Parameters
    ----------
    X : torch.Tensor
        Data, shape (N, D).
    k : int
        Number of centers to pick.
    seed : int
        Random seed.

    Returns
    -------
    centers : torch.Tensor
        Shape (k, D), each row is a sampled center from X.

    Algorithm
    ---------
    1. Pick first center uniformly at random from X.
    2. For each remaining slot:
        a. Compute D(x) = squared distance from each point x to its NEAREST already-chosen center.
        b. Sample the next center from X with probability ∝ D(x).
    """
    N, D = X.shape
    device = X.device
    g = torch.Generator(device="cpu").manual_seed(seed)

    centers = torch.empty(k, D, dtype=X.dtype, device=device)
    first_idx = int(torch.randint(0, N, (1,), generator=g).item())
    centers[0] = X[first_idx]

    sq_dists = torch.full((N,), float("inf"), dtype=X.dtype, device=device)

    for i in range(1, k):
        new_sq = (X - centers[i - 1]).pow(2).sum(dim=1)
        sq_dists = torch.minimum(sq_dists, new_sq)
        total = sq_dists.sum()
        if total.item() == 0.0:
            # All points already covered exactly — fill remaining centers with random points
            remaining_indices = torch.randint(0, N, (k - i,), generator=g)
            for j, idx in enumerate(remaining_indices):
                centers[i + j] = X[int(idx.item())]
            break
        probs = sq_dists / total
        probs_cpu = probs.detach().cpu()
        next_idx = int(torch.multinomial(probs_cpu, 1, generator=g).item())
        centers[i] = X[next_idx]

    return centers
