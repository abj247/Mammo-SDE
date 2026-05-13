"""Models: GMM, Vanilla VAE, σ-VAE, VampPrior VAE — all from scratch."""

from mammo_sde.models.base_vae import BaseVAE, VAELossOutput
from mammo_sde.models.gmm import GMM
from mammo_sde.models.sigma_vae import SigmaVAE
from mammo_sde.models.vae import VanillaVAE
from mammo_sde.models.vamp_vae import VampPriorVAE, init_pseudoinputs_from_gmm

__all__ = [
    "GMM",
    "BaseVAE",
    "VAELossOutput",
    "VanillaVAE",
    "SigmaVAE",
    "VampPriorVAE",
    "init_pseudoinputs_from_gmm",
]


def build_vae(variant: str, **kwargs):
    """Factory: build a VAE variant by name."""
    variant = variant.lower()
    if variant in ("vanilla", "vae"):
        return VanillaVAE(**kwargs)
    if variant in ("sigma", "sigma_vae", "sigma-vae", "σ-vae"):
        return SigmaVAE(**kwargs)
    if variant in ("vamp", "vamp_vae", "vampprior"):
        return VampPriorVAE(**kwargs)
    raise ValueError(f"Unknown VAE variant {variant!r}. Choose: vanilla, sigma, vamp")
