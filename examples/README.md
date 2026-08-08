# ZeroGrad Examples

Demos for the ZeroGrad evolutionary optimizer — especially non-differentiable
forwards and distributed / cluster population evaluation. Shared helpers live
alongside the scripts (`_xor_model.py`, `_vision_mlp_train.py`, `_data.py`, …).

## Quick start

```bash
# CPU default — add a GPU extra if needed: --extra cuda12 | cuda13 | rocm7
uv sync
uv pip install -e ".[dev]"
uv pip install Pillow  # only for CIFAR-10 image loading
```

## Scripts

| Script | Dataset | Model | Key takeaway |
|--------|---------|-------|--------------|
| `train_xor.py` | XOR (synthetic) | 2→16→1 MLP | Non-linear separation without gradients |
| `train_sine.py` | sin(x) | 1→32→1 MLP | Continuous regression without gradients |
| `train_qat_xor.py` | XOR | 2→16→1 MLP, **4-bit** | Trains through `jnp.round()` — **no STE** |
| `train_vision_mlp.py` | MNIST / CIFAR-10 | 2-layer MLP | `--dataset {mnist,cifar10}` |
| `train_cnn_int8_mlp.py` | CIFAR-10 | IntConv + IntLUT | Pure-int8 CNN + ±1 bin updates |
| `train_vit_cifar10.py` | CIFAR-10 | ViT | ZeroGrad vs AdamW / QAT — [findings](vit_findings.md) |
| `train_distributed_xor.py` | XOR | 2→16→1 MLP | `--layout {cpu_gpu,dual_gpu}` or `--devices …` |
| `train_distributed_asymmetric.py` | Synthetic | MLP | Weighted / auto-calibrated splits |
| `train_cluster_xor.py` | XOR | 2→16→1 MLP | `--mode {seed,multiprocess,unreliable}` |

## Running

```bash
# Fast (seconds) — synthetic data, no downloads
uv run python examples/train_xor.py
uv run python examples/train_sine.py
uv run python examples/train_qat_xor.py

# Distributed — split population across devices
uv run python examples/train_distributed_xor.py --layout cpu_gpu
uv run python examples/train_distributed_xor.py --layout dual_gpu
uv run python examples/train_distributed_asymmetric.py

# Cluster — seed-derived params, only fitnesses shared
uv run python examples/train_cluster_xor.py --mode seed
uv run python examples/train_cluster_xor.py --mode multiprocess
uv run python examples/train_cluster_xor.py --mode unreliable

# Vision (downloads on first run → ~/.cache/zerograd/)
uv run python examples/train_vision_mlp.py --dataset mnist --steps 200
uv run python examples/train_vision_mlp.py --dataset cifar10 --steps 300
uv run python examples/train_cnn_int8_mlp.py --steps 200
uv run python examples/train_vit_cifar10.py --steps 200
```

## How ZeroGrad differs from backprop

Standard training computes `∂loss/∂params` via reverse-mode autodiff and feeds
it to an optimizer (AdamW, SGD, ...). ZeroGrad instead:

1. Generates N perturbed parameter sets using deterministic low-rank factors.
2. Evaluates the loss for each candidate (forward pass only — **no backward**).
3. Shapes the losses into a descent direction (centered-rank weighting).
4. Replays the same factors into a pseudo-gradient via einsum.
5. Feeds the pseudo-gradient to a standard Optax transform (AdamW, ...).

This means **any differentiable or non-differentiable forward pass works** —
the optimizer never differentiates through your model.

## The QAT example: why zero-gradient matters

`train_qat_xor.py` applies 4-bit quantization (`jnp.round()`) to every layer's
output. With backprop, `round()` has zero gradient almost everywhere, so
practitioners use the Straight-Through Estimator (STE):

```python
# Backprop + STE hack
x_q = x + jax.lax.stop_gradient(jnp.round(x) - x)
```

With ZeroGrad, this hack is unnecessary. The pseudo-gradient comes from ES
perturbations of the *parameters*, not from differentiating through the forward
pass. So you write the quantization naturally:

```python
# ZeroGrad — no STE needed
x_q = jnp.round(jnp.clip(x / scale, -8, 7)) * scale
```

## Distributed / cluster

Population evaluation is embarrassingly parallel. Workers share only 1D loss
arrays. Seed-derived clusters never communicate parameters — each node
replays the same fitness history from a shared seed.

```bash
uv run python examples/train_distributed_xor.py --layout cpu_gpu
uv run python examples/train_cluster_xor.py --mode seed
uv run python examples/train_cluster_xor.py --mode multiprocess
uv run python examples/train_cluster_xor.py --mode unreliable
```
