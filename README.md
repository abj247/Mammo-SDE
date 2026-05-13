# Mammo_SDE: Generative Models on Mammogram Embeddings for Latent SDE Risk Prediction

**Status:** Phase 1 (Gaussian Mixture Model) and Phase 2 (three Variational Autoencoder variants) implemented from scratch in PyTorch. 21 unit tests passing. Verified end-to-end on synthetic data.

This repository implements the `g(·)` generator block of a longitudinal mammographic risk prediction pipeline. Given a frozen image encoder `E*` (such as Mammo-CLIP or Mammo-FM) producing per-exam embeddings `h`, the generator `g(·)` learns a probability distribution over the embedding space from which new embeddings `ĥ` can be sampled. Downstream, these samples feed a survival head `f(·)` that produces a cancer risk score. This codebase covers only `g(·)`; the encoder and the downstream survival head are intentionally out of scope.

The work is organised around a five-phase model ladder. The Phase 1 model (GMM) is a non-parametric encoder sanity check that does not require neural training. Phases 2 through 5 (Vanilla VAE, σ-VAE, VampPrior VAE, Latent SDE) progressively add structure: stochastic encoding, calibrated decoder variance, learned multimodal priors, and eventually continuous-time dynamics. This repository implements Phases 1 and 2 in full; Phases 3 to 5 are scheduled as follow-up work.

---

## Pipeline

```
                       (frozen)
   mammogram B  --->   E*(·)   --->   h ∈ R^d
                                       │
                                       v
                                     g(·)        ←  this repository
                                       │
                                       v
                                  D(μ, σ²)
                                       │
                                  sample ĥ
                                       │
                                       v
                                     f(·)         ←  downstream (out of scope)
                                       │
                                       v
                              cancer score ∈ [0, 1]
```

`g(·)` is implemented as four interchangeable models, fit or trained on pre-extracted encoder embeddings:

| Model | Role | Training |
|---|---|---|
| **GMM** | Encoder sanity check; baseline generator | EM (no gradients) |
| **Vanilla VAE** | Probabilistic generator baseline | Backprop |
| **σ-VAE** | Calibrated decoder variance | Backprop |
| **VampPrior VAE** | Learned mixture prior; pseudoinputs initialised from GMM centres | Backprop |

All four models are implemented from scratch in PyTorch. `scikit-learn` is used only for downstream visualisation utilities (t-SNE, UMAP), not for the model fitting itself.

---

## Method

### Notation

Let `h ∈ R^d` be a single exam-level embedding produced by the frozen encoder. We are given a dataset `{h_n}_{n=1..N}`. The goal of `g(·)` is to fit a probability density `p(h)` so that we can both score new embeddings and sample from the learned density.

### GMM (Phase 1)

The Gaussian Mixture Model parameterises the density as a weighted sum of K Gaussians:

```
p(h) = Σ_k π_k N(h ; μ_k, Σ_k),   Σ_k π_k = 1
```

Fitted by Expectation-Maximisation. Four covariance parameterisations are supported (`full`, `diag`, `spherical`, `tied`). Model selection over K uses Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC).

Numerical stability is handled throughout via the log-sum-exp identity. The covariance Cholesky factor is cached after each M-step so that the log-likelihood evaluation in the next E-step is `O(K · D²)` per sample without explicit inversion. K-means++ initialisation is used by default.

The role of GMM in this codebase is twofold:

1. **Encoder sanity check.** If the embedding space `h` has clinically meaningful structure (clusters by BI-RADS density, age strata, cancer outcome), GMM should recover it without supervision. If the GMM finds no structure, the encoder is broken and no downstream model will rescue it.
2. **Initialisation source for Phase 2.** The cluster centres `{μ_k}` are passed to the VampPrior VAE as initial pseudoinputs.

### Vanilla VAE (Kingma and Welling 2014)

An MLP encoder maps `h → (μ, log σ²)` over a latent `z ∈ R^L`. The reparameterisation trick `z = μ + σ ⊙ ε` with `ε ~ N(0, I)` enables backpropagation through the sampling step. An MLP decoder maps `z` back to `ĥ`. The training objective is the negative ELBO:

```
L(h) = E_{z ~ q(z|h)}[ -log p(h|z) ] + KL(q(z|h) || N(0, I))
```

The reconstruction term `log p(h|z)` is Gaussian with implicit unit variance, which collapses to a mean-squared-error loss up to a constant. The KL term has a closed form against the standard normal prior. β-annealing (linearly from 0 to 1 over the first 20 epochs) is applied to mitigate posterior collapse.

### σ-VAE (Rybkin, Daniilidis, and Levine, ICML 2021)

The Vanilla VAE objective has a hidden bias: the implicit assumption of `σ²_x = 1` for the decoder noise creates a constant scaling of the KL term relative to the reconstruction term. σ-VAE makes the decoder variance explicit and either learns it as a scalar parameter `log σ_x` or computes the analytically optimal value per batch:

```
σ²_x*(batch) = mean_{n, d}[ (h_n,d − ĥ_n,d)² ]
```

The optimal mode is hyperparameter-free and is the default. The Gaussian log-likelihood is then evaluated with the calibrated variance:

```
log p(h | z) = -½ [ D log(2π σ²_x) + ‖h − ĥ‖² / σ²_x ]
```

### VampPrior VAE (Tomczak and Welling, AISTATS 2018)

The standard normal prior `p(z) = N(0, I)` is unimodal and cannot match the multimodal structure of the aggregated posterior `q(z) = E_h[q(z|h)]` when embeddings cluster. The VampPrior replaces it with a mixture over the encoder evaluated at K learned pseudoinputs `u_1, ..., u_K ∈ R^d`:

```
p(z) = (1/K) Σ_k q(z | u_k)
```

The KL term has no closed form; it is estimated via Monte Carlo using the single sample `z` already drawn during the forward pass:

```
KL(q(z|h) || p(z)) ≈ log q(z|h) − log [ (1/K) Σ_k q(z | u_k) ]
```

In this codebase, the pseudoinputs `u_k` are initialised from the GMM cluster centres fitted in Phase 1 (`--vamp-init-from-gmm <gmm_checkpoint>`). This is a deliberate experimental design: the GMM independently identifies modes of the embedding distribution, and the VampPrior is then seeded with those modes rather than learning them from random initialisation. Empirically (see Smoke test results below), this initialisation produces dramatically better prior-sample fidelity than either the Vanilla or σ-VAE on clustered data.

### Encoder contract

The downstream Latent SDE (Phase 5) makes three assumptions about the encoder's output. These are documented for the upstream encoder team and are independent of which encoder is plugged in:

1. **One continuous d-dimensional vector per exam, encoded independently.** Attention pool over the four mammogram views happens inside the encoder. Outputs are continuous (no softmax, no discrete tokens). Each exam is encoded in its own forward pass; no cross-exam context.
2. **LayerNorm at the output.** The final operation must be `LayerNorm(d)` so that embeddings have bounded, consistent norm across exams.
3. **Acquisition invariance.** The encoder is trained with augmentations and a contrastive objective so that the same exam under different acquisition conditions (compression, positioning, machine model, year, exposure) produces nearly identical features.

This codebase enforces these requirements through the `BaseEncoder` abstract interface in `src/mammo_sde/encoders/base_encoder.py`.

---

## Installation

This package targets Python 3.10 and PyTorch 2.0 or newer.

```bash
git clone git@github.com:abj247/mammo-sde.git
cd mammo-sde
python -m pip install -e .[dev]
```

The `[dev]` extra installs pytest, pre-commit, and ruff in addition to the runtime dependencies.

On the development HPC cluster, PyTorch threading interacts poorly with the σ-VAE backward pass. Set the following environment variables before running any script:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

This is a known issue specific to the cluster's BLAS configuration; it does not affect output correctness on machines without the issue.

---

## Usage

### 1. Extract embeddings from your encoder

The extraction script is encoder-agnostic and accepts a checkpoint path. A `synthetic` mode is provided so that the full pipeline can be exercised without any real encoder weights.

```bash
python scripts/extract_embeddings.py \
    --encoder-type synthetic \
    --scope dev10k \
    --embedding-dim 64 \
    --n-clusters-synthetic 5 \
    --output-dir outputs/embeddings/synthetic/dev10k
```

For a real encoder, replace `--encoder-type synthetic` with `mammo_clip` or `generic_vit` and supply `--checkpoint-path <weights.pt>`. See `src/mammo_sde/encoders/` for the wrapper implementations.

### 2. Run GMM analysis

```bash
python scripts/run_gmm_analysis.py \
    --embeddings-path outputs/embeddings/synthetic/dev10k/embeddings.h5 \
    --k-min 2 --k-max 20 --cov-type diag \
    --output-dir outputs/gmm/synthetic_dev10k
```

This sweeps K from 2 to 20, selects the best K by BIC, fits the corresponding GMM, computes cluster-to-label alignment statistics against any clinical metadata present in the embeddings file, and produces 16 diagnostic plots plus 9 JSON metric files.

### 3. Train the three VAE variants

```bash
# Vanilla VAE
python scripts/train_vae.py --variant vanilla \
    --embeddings-path outputs/embeddings/synthetic/dev10k/embeddings.h5 \
    --latent-dim 16 --epochs 20 \
    --output-dir outputs/vae/synthetic_dev10k

# σ-VAE (optimal decoder variance mode)
python scripts/train_vae.py --variant sigma --sigma-mode optimal \
    --embeddings-path outputs/embeddings/synthetic/dev10k/embeddings.h5 \
    --latent-dim 16 --epochs 20 \
    --output-dir outputs/sigma_vae/synthetic_dev10k

# VampPrior VAE with pseudoinputs initialised from the fitted GMM
python scripts/train_vae.py --variant vamp \
    --vamp-init-from-gmm outputs/gmm/synthetic_dev10k/checkpoints/best_gmm.pt \
    --n-pseudoinputs 50 \
    --embeddings-path outputs/embeddings/synthetic/dev10k/embeddings.h5 \
    --latent-dim 16 --epochs 20 \
    --output-dir outputs/vamp_vae/synthetic_dev10k
```

### 4. Compare variants

```bash
python scripts/compare_vae_variants.py \
    --runs outputs/vae/synthetic_dev10k \
           outputs/sigma_vae/synthetic_dev10k \
           outputs/vamp_vae/synthetic_dev10k \
    --names vanilla sigma vamp \
    --output-dir outputs/comparison/synthetic_dev10k
```

Produces side-by-side training curves, a grouped bar chart of recon MSE / active dimension fraction / total KL / prior-sample distance, and a KL-per-dimension overlay plot.

### Data scopes

All scripts accept `--scope {dev10k, longitudinal, full}`. The scope controls how many exams are processed:

- `dev10k`: 10,000 randomly sampled exams. Use for pipeline validation.
- `longitudinal`: patients with two or more exams (target population for downstream SDE phases).
- `full`: the entire available cohort.

---

## Project structure

```
mammo-sde/
├── README.md                              this file
├── LICENSE                                MIT
├── CITATION.cff                           machine-readable citation metadata
├── pyproject.toml                         project metadata, ruff config, dev extras
├── requirements.txt                       runtime dependencies
├── .pre-commit-config.yaml                pre-commit hooks
├── .github/workflows/ci.yml               GitHub Actions: lint + test
├── configs/                               YAML configs
├── src/mammo_sde/
│   ├── data/
│   │   ├── embedding_dataset.py           PyTorch Dataset over an HDF5 of embeddings
│   │   └── embedding_loader.py            DataLoader splits + synthetic generator
│   ├── encoders/
│   │   ├── base_encoder.py                Abstract interface (frozen, exam-level)
│   │   ├── mammo_clip_encoder.py          Mammo-CLIP wrapper template
│   │   └── generic_vit_encoder.py         Generic ViT wrapper (I-JEPA, timm, HF ViT)
│   ├── models/
│   │   ├── kmeans_pp.py                   K-means++ initialisation
│   │   ├── gmm.py                         GMM with EM, 4 cov types, BIC/AIC, sample
│   │   ├── base_vae.py                    Shared MLP encoder/decoder and ELBO
│   │   ├── vae.py                         Vanilla VAE
│   │   ├── sigma_vae.py                   σ-VAE (calibrated decoder)
│   │   └── vamp_vae.py                    VampPrior VAE + GMM-init helper
│   ├── training/
│   │   ├── fit_gmm.py                     K-sweep + BIC selection + save
│   │   └── train_vae.py                   Training loop for all VAE variants
│   ├── analysis/
│   │   ├── embedding_stats.py             Per-dim stats, PCA scree, pairwise cosine
│   │   ├── gmm_analysis.py                Cluster purity, sizes, sample fidelity
│   │   └── vae_analysis.py                Recon MSE, active dims, KL per dim, prior fidelity
│   ├── visualization/
│   │   ├── plots_embeddings.py
│   │   ├── plots_gmm.py
│   │   ├── plots_vae.py
│   │   └── plots_compare.py
│   └── utils/
│       ├── logger.py                      rich-based StepLogger
│       ├── seed.py
│       ├── timer.py
│       └── io.py                          HDF5, JSON, YAML, torch checkpoints
├── scripts/                               CLI entry points
│   ├── extract_embeddings.py
│   ├── run_gmm_analysis.py
│   ├── train_vae.py
│   └── compare_vae_variants.py
├── tests/                                 21 unit tests (pytest)
├── outputs/                               per-run outputs (plots, results, checkpoints, logs)
└── notebooks/                             exploratory analysis
```

A more detailed per-file walkthrough is available in the project's Notion page; see the "Codebase Walkthrough" subpage of the Jump-SDE page for the API of every module and the design rationale.

---

## Reproducibility

Every script accepts a `--seed` flag and sets all sources of randomness (Python, NumPy, PyTorch CPU and CUDA) via `mammo_sde.utils.seed.set_seed`. PyTorch's `cudnn.deterministic` is enabled by default.

Every run writes to a self-contained directory of the form `outputs/<model>/<run>/` containing:

```
plots/         all matplotlib figures as PNG
results/       JSON metric files (one per analysis step)
checkpoints/   model state dicts
logs/          plain-text log file mirrored from the StepLogger
```

The HDF5 embedding file produced by `extract_embeddings.py` carries the encoder type, scope, seed, and other run metadata as HDF5 attributes, so any downstream artifact can be traced back to the exact extraction settings.

---

## Smoke test results

A synthetic dataset of 10,000 embeddings drawn from a five-component Gaussian mixture in 64 dimensions is included as a regression test of the full pipeline. All four models are exercised end-to-end on this dataset.

| Step | Outcome | Wall time |
|---|---|---|
| Synthetic extraction | 10,000 × 64, mean norm 23.7 | 0.16 s |
| GMM K-sweep (2..10, diag covariance) | BIC selects K = 5, matching ground truth | 2.9 s |
| GMM cluster purity vs ground-truth labels | 1.000 (perfect recovery) | — |
| Vanilla VAE (latent 16, 20 epochs) | Recon MSE 0.99, 16/16 active dimensions | 21 s |
| σ-VAE (optimal mode) | Recon MSE 0.99, 13/16 active dimensions | 17 s |
| VampPrior VAE (initialised from GMM centres) | Recon MSE 1.01, 16/16 active dimensions | 18 s |
| Prior-sample mean L2 distance to real | Vanilla 1.65, σ-VAE 1.77, **VampPrior 0.75** | — |

The VampPrior VAE attains roughly half the prior-sample distance of the other two variants on clustered data. This validates the GMM-to-VampPrior initialisation pathway: the multimodal prior matches the multimodal data distribution, whereas the unimodal N(0, I) prior used by the Vanilla and σ-VAE cannot.

---

## Tests

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest tests/
```

21 unit tests covering:

- GMM correctness on synthetic data for all four covariance types
- Monotone log-likelihood per EM iteration
- BIC-based K recovery on data with known K
- Responsibilities summing to one
- Sample fidelity to the fitted mixture
- State-dict roundtrip preserving predictions exactly
- Loss decrease over training for all three VAE variants
- Pseudoinput initialisation from a GMM checkpoint
- Tensor shapes through forward and sampling paths
- KL non-negativity against the standard normal prior
- HDF5 embedding I/O roundtrip with metadata and attrs
- DataLoader train/val/test split sizes

---

## Development workflow

Pre-commit hooks (formatting, linting, file hygiene) are enforced locally. GitHub Actions runs the full test suite in a clean Ubuntu environment on every push and pull request to `main`.

```bash
# One-time setup after cloning
python -m pip install -e .[dev]
pre-commit install

# Optional: run all hooks against all files
pre-commit run --all-files
```

The pre-commit configuration intentionally excludes the test suite for low commit latency. Tests are run by CI in a clean environment, which is the appropriate venue for them.

---

## References

1. Kingma, D. P. and Welling, M. (2014). Auto-Encoding Variational Bayes. International Conference on Learning Representations.
2. Tomczak, J. M. and Welling, M. (2018). VAE with a VampPrior. Artificial Intelligence and Statistics (AISTATS).
3. Rybkin, O., Daniilidis, K. and Levine, S. (2021). Simple and Effective VAE Training with Calibrated Decoders. International Conference on Machine Learning (ICML).
4. Burda, Y., Grosse, R. and Salakhutdinov, R. (2016). Importance Weighted Autoencoders. International Conference on Learning Representations. Source of the "active dimensions" definition used in `analysis/vae_analysis.py`.
5. Arthur, D. and Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. ACM-SIAM Symposium on Discrete Algorithms (SODA).
6. Ghosh, S. et al. (2024). Mammo-CLIP: A Vision Language Foundation Model for Mammography. MICCAI. Supported encoder type via `mammo_clip_encoder.py`.

---

## Citation

If you use this code, please cite:

```bibtex
@software{mammo_sde_2026,
  author  = {Jain, Abhishek},
  title   = {Mammo\_SDE: Generative Models on Mammogram Embeddings for Latent SDE Risk Prediction},
  year    = {2026},
  url     = {https://github.com/abj247/mammo-sde},
  version = {0.1.0}
}
```

GitHub will also surface this via the "Cite this repository" button using the `CITATION.cff` file at the repository root.

---

## License

MIT. See `LICENSE` for the full text.

---

## Acknowledgments

This work is part of a larger research program on latent stochastic differential equations for breast cancer risk prediction. The codebase is encoder-agnostic; the Mammo-CLIP and Mammo-FM foundation models are the intended upstream encoders, supported via wrapper templates in `src/mammo_sde/encoders/`.
