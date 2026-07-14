# ZeroGrad Examples

End-to-end training scripts that demonstrate the ZeroGrad evolutionary optimizer
on real and synthetic tasks. Each script is self-contained — no shared training
loop or model builder.

## Quick start

```bash
uv sync
uv pip install -e ".[dev]"
uv pip install Pillow  # only for CIFAR-10 image loading
```

## Scripts

| Script | Dataset | Model | Key takeaway |
|--------|---------|-------|--------------|
| `train_xor.py` | XOR gate (synthetic) | 2→16→1 MLP | Non-linear separation without gradients |
| `train_sine.py` | sin(x) (synthetic) | 1→32→1 MLP | Continuous regression without gradients |
| `train_qat_xor.py` | XOR gate (synthetic) | 2→16→1 MLP, **4-bit quantized** | Trains through `jnp.round()` — **no STE needed** |
| `train_mnist.py` | MNIST | 784→64→10 MLP | Real image classification |
| `train_cifar10.py` | CIFAR-10 | 3072→128→10 MLP | Harder image classification |
| `train_vit_cifar10.py` | CIFAR-10 | ViT (2L, 4H, d=64) | ZeroGrad vs AdamW, bf16 vs 4-bit QAT — see [findings](vit_findings.md) |
| `train_distributed_cpu_gpu.py` | XOR gate (synthetic) | 2→16→1 MLP | Population split across **CPU + GPU** workers |
| `train_distributed_dual_worker.py` | XOR gate (synthetic) | 2→16→1 MLP | Two workers on the **same GPU** |

## Running

```bash
# Fast (seconds) — synthetic data, no downloads
uv run python examples/train_xor.py
uv run python examples/train_sine.py
uv run python examples/train_qat_xor.py

# Distributed — split population across devices
uv run python examples/train_distributed_cpu_gpu.py
uv run python examples/train_distributed_dual_worker.py

# Slower (minutes) — downloads data on first run
uv run python examples/train_mnist.py --steps 200
uv run python examples/train_cifar10.py --steps 300
```

Datasets are cached in `~/.cache/zerograd/` after first download.

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

The quantization scale is a learnable parameter (manifest VECTOR entry) that
ZeroGrad optimizes alongside the weights — all without ever backpropagating
through `round()`.

## Distributed ZeroGrad: multi-device population evaluation

Because each candidate in the ES population is evaluated independently (forward
pass only), the population is embarrassingly parallel. Workers only need to
share their 1D loss arrays — a few hundred floats per step — to coordinate.

```
   Worker 0 (CPU)          Worker 1 (GPU)
  ┌──────────────┐       ┌──────────────┐
  │ candidates   │       │ candidates   │
  │   0..15      │       │  16..31      │
  │ ↓ loss_fn    │       │ ↓ loss_fn    │
  │ losses[0:16] │       │ losses[16:32]│
  └──────┬───────┘       └──────┬───────┘
         │                      │
         └──────┬───────────────┘
                ↓
         concatenate losses
                ↓
         shape + replay + optax step
```

The `DistributedZeroGrad` coordinator handles this automatically:

```python
from zerograd import ZeroGrad, DistributedZeroGrad

cpu = jax.devices('cpu')[0]
gpu = jax.devices('gpu')[0]

opt = ZeroGrad(manifest, optax.adamw(1e-2), population_size=32, ...)
dist_opt = DistributedZeroGrad(opt, devices=[cpu, gpu], loss_fn=loss_fn)

state = dist_opt.init(params)
for step in range(steps):
    params, state, metrics = dist_opt.step(state, params, batch)
```

For a 4× GPU node, pass all four GPU devices — each gets a quarter of the
population. For two workers on one GPU, pass the same GPU device twice. The
protocol is identical in all cases.
