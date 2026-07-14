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
| `train_distributed_asymmetric.py` | Synthetic (512→512→10) | MLP | **Asymmetric compute**: manual weights vs auto-calibration |
| `train_cluster_seed_derived.py` | XOR gate (synthetic) | 2→16→1 MLP | **Seed-derived cluster**: params never communicated, only losses |
| `train_cluster_multiprocess.py` | XOR gate (synthetic) | 2→16→1 MLP | **True multi-process**: 4 isolated processes, params verified identical |

## Running

```bash
# Fast (seconds) — synthetic data, no downloads
uv run python examples/train_xor.py
uv run python examples/train_sine.py
uv run python examples/train_qat_xor.py

# Distributed — split population across devices
uv run python examples/train_distributed_cpu_gpu.py
uv run python examples/train_distributed_dual_worker.py
uv run python examples/train_distributed_asymmetric.py

# Cluster — seed-derived params, only fitnesses shared
uv run python examples/train_cluster_seed_derived.py
uv run python examples/train_cluster_multiprocess.py

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

### Asymmetric compute (weighted partitioning)

When devices have different speeds, an even split wastes the fast device.
Pass ``weights`` to assign more candidates to faster devices:

```python
# 1 slow CPU + 1 fast GPU: GPU gets 4× more candidates
dist_opt = DistributedZeroGrad(opt, devices=[cpu, gpu], loss_fn=loss_fn, weights=[1, 4])

# 4 slow GPUs + 1 fast GPU
dist_opt = DistributedZeroGrad(
    opt, devices=[gpu0, gpu1, gpu2, gpu3, gpu4], loss_fn=loss_fn,
    weights=[1, 1, 1, 1, 4],
)
```

Or let auto-calibration measure each device and set weights automatically:

```python
dist_opt = DistributedZeroGrad(opt, devices=[cpu, gpu], loss_fn=loss_fn)
results = dist_opt.calibrate(params, batch)  # times each device, sets weights
print(dist_opt.weights)      # e.g. [1.0, 3.7]
print(dist_opt.partition_sizes)  # e.g. [14, 50]
```

Calibration runs a few timed evaluations per device (with JIT warmup) and
sets weights inversely proportional to per-candidate time.  Call it once
before training starts.

### Seed-derived cluster (params never communicated)

For multi-node clusters, ZeroGrad has an even stronger property: **parameters
are never sent over the network**. Each node computes its own params locally
from a shared seed and the sequence of fitness arrays.

```
  Node 0                Node 1                Node 2
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ seed → params│  │ seed → params│  │ seed → params│  (computed locally)
│ evaluate     │  │ evaluate     │  │ evaluate     │
│ ↓ losses     │  │ ↓ losses     │  │ ↓ losses     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └────────┬────────┴────────┬────────┘
                ↓                 ↓
         gather losses (O(pop) bytes)
                ↓
         broadcast to all nodes
                ↓
   each node: step_from_losses → identical params
```

The per-step communication is O(population) bytes — a few hundred floats —
**regardless of model size**. A 1B-parameter model and a 1M-parameter model
have the same network cost.

```python
from zerograd import ClusterZeroGrad

cluster = ClusterZeroGrad(
    optimizer, build_params_fn, loss_fn,
    seed=42, num_nodes=4,
)
for step in range(steps):
    params, state, metrics = cluster.step(batch)
    assert cluster.verify_sync()  # all nodes have identical params
```

`train_cluster_seed_derived.py` runs this in-process with `verify_sync()`
after every step. `train_cluster_multiprocess.py` spawns 4 separate Python
processes communicating via pipes — proving true process isolation with
zero param communication.
