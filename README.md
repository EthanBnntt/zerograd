# zerograd

`zerograd` is a JAX + Optax library for zero-gradient evolutionary optimization. It evaluates a pure loss callback against deterministic, factor-only candidate perturbations, replays those same low-rank factors into pseudo-gradients, and applies an Optax transform transactionally.

## Design principles

- **No objective gradients:** fitness-only Evolution Strategies primitives; arbitrary JAX-compatible objectives may be evaluated.
- **No dense perturbation materialization:** candidate perturbations are PRNG-derived A/B low-rank factors, never materialized as dense matrices or tables during forward evaluation.
- **Manifest identity:** Flax NNX modules get layouts via graph surgery; dict params use an explicit `Manifest` (layouts and tie groups are never inferred from PyTree leaf order alone).
- **Transactional lifecycle:** `init` / `step` return new state only after candidate evaluation, factor replay, and Optax application complete.

## Installation

Default install is **CPU-only** JAX. Pick exactly one accelerator extra for GPU:

| Extra | Backend | Notes |
|-------|---------|-------|
| *(none)* / `cpu` | CPU | Portable default; good for CI and laptops |
| `cuda` / `cuda12` | NVIDIA CUDA 12 | Pip wheels include CUDA/cuDNN runtime libs |
| `cuda13` | NVIDIA CUDA 13 | Pip wheels include CUDA/cuDNN runtime libs |
| `cuda12-local` / `cuda13-local` | NVIDIA CUDA 12/13 | Use a system CUDA toolkit already on `LD_LIBRARY_PATH` |
| `rocm7` | AMD ROCm 7 | Requires a local ROCm 7.x install; installs the JAX ROCm plugin only |

```bash
# CPU (default)
uv sync
uv pip install -e ".[dev]"

# NVIDIA CUDA 12 (pip-bundled runtime)
uv sync --extra cuda12
uv pip install -e ".[cuda12,dev]"

# NVIDIA CUDA 13
uv sync --extra cuda13

# System CUDA toolkit (no NVIDIA pip libs)
uv sync --extra cuda12-local   # or cuda13-local

# AMD ROCm 7 (ROCm must already be installed on the host)
uv sync --extra rocm7
uv pip install -e ".[rocm7,dev]"
```

With plain pip:

```bash
pip install -e ".[cpu]"          # or just: pip install -e .
pip install -e ".[cuda12]"
pip install -e ".[cuda13]"
pip install -e ".[cuda12-local]"  # system CUDA
pip install -e ".[rocm7]"        # system ROCm 7
```

### ROCm 6

Upstream JAX no longer ships a `rocm6` / `jax[rocm]` extra on current releases
(`>=0.9` only publish `jax[rocm7-local]`). For ROCm 6.x, install a ROCm-6-compatible
JAX stack first (see [AMD's JAX on ROCm docs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html)),
then install zerograd without a GPU extra so it reuses that JAX:

```bash
# after AMD/ROCm-6 jax + jaxlib + jax-rocm60-* are installed
pip install -e ".[dev]"
```

Do not combine GPU extras (for example `cuda12` and `rocm7`) in one environment.

## Quick start (NNX)

Build a normal Flax NNX module. `opt.init(model)` replaces layers with factor-aware ones and builds the manifest automatically; the loss sees the model under candidate perturbations.

```python
import jax.numpy as jnp
import optax
from flax import nnx
from zerograd import ZeroGrad

class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(2, 16, rngs=rngs)
        self.l2 = nnx.Linear(16, 1, rngs=rngs)

    def __call__(self, x):
        return self.l2(nnx.relu(self.l1(x)))

model = MLP(nnx.Rngs(0))
opt = ZeroGrad(
    optax.adamw(1e-2, weight_decay=0.0),
    population_size=32,
    rank=4,
    sigma=0.1,
    seed=0,
    run_id="xor",
)
state = opt.init(model)

def loss_fn(model, batch):
    x, y = batch
    logits = jnp.squeeze(model(x), -1)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y.astype(jnp.float32))), None

x = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = jnp.array([0, 1, 1, 0])
model, state, metrics = opt.step(state, model, (x, y), loss_fn)
print(f"gen {metrics.generation}: mean_loss={metrics.mean_loss:.4f}")
```

## Manifest (auto-built)

`opt.init(model)` runs graph surgery and builds a `Manifest` automatically.
Inspect it after init when you need layouts / tie groups:

```python
state = opt.init(model)
print(opt.manifest.entries)
```

Dict params with an explicit `Manifest` remain supported for `init` /
`step_from_losses` (custom evaluation). Prefer the NNX path for `step`.

## Distributed multi-device evaluation

The ES population is embarrassingly parallel — each candidate's loss is
computed independently, and workers only share 1D fitness arrays.

```python
from zerograd import DistributedZeroGrad, ZeroGrad

cpu = jax.devices('cpu')[0]
gpu = jax.devices('gpu')[0]

opt = ZeroGrad(optax.adamw(1e-2), population_size=64, rank=4, sigma=0.1, seed=0, run_id="dist")
dist_opt = DistributedZeroGrad(opt, devices=[cpu, gpu], loss_fn=loss_fn)

state = dist_opt.init(model)
for step in range(steps):
    model, state, metrics = dist_opt.step(state, model, batch)
```

Workers can be mixed across CPU and GPU, or multiple shards can share a
single GPU. For a 4× GPU node, pass all four devices — each gets a quarter
of the population. See `examples/train_distributed_cpu_gpu.py` and
`examples/train_distributed_dual_worker.py` for runnable demos (both are
thin wrappers around `examples/train_distributed_xor.py --devices ...`,
which takes an arbitrary comma-separated device/weight topology).

For asymmetric compute (slow CPU + fast GPU, or mixed GPU generations),
pass ``weights`` to assign more candidates to faster devices:

```python
# GPU gets 4× more candidates than CPU
dist_opt = DistributedZeroGrad(opt, devices=[cpu, gpu], loss_fn=loss_fn, weights=[1, 4])

# Or auto-calibrate from measured device speed
dist_opt.calibrate(model, batch)
```

For custom multi-process setups, use `optimizer.evaluate_shard()` and
`optimizer.step_from_losses()` directly — the only data that crosses process
boundaries is the 1D loss array.

### Seed-derived cluster (params never communicated)

For multi-node clusters, parameters are never sent over the network. Each
node computes its own params from a shared seed and the sequence of fitness
arrays — only the 1D loss array (O(population) bytes, model-size independent)
is communicated.

```python
from zerograd import ClusterZeroGrad

cluster = ClusterZeroGrad(optimizer, build_params_fn, loss_fn, seed=42, num_nodes=4)
for step in range(steps):
    params, state, metrics = cluster.step(batch)
    assert cluster.verify_sync()  # all nodes have identical params
```

See `examples/train_cluster_seed_derived.py` (in-process) and
`examples/train_cluster_multiprocess.py` (true multi-process) for demos.

### Fault-tolerant cluster (node death, late join, pause/resume)

For decentralized compute where nodes go offline unpredictably:

```python
from zerograd import FaultTolerantCluster

cluster = FaultTolerantCluster(optimizer, build_params_fn, loss_fn, seed=42, initial_nodes=4)

# Late join: new node replays loss history to catch up
cluster.add_node(weight=2.0)

# Pause/resume: work redistributed, node catches up on resume
cluster.pause_node(0)
cluster.resume_node(0)

# Death: work redistributed to survivors
cluster.remove_node(2)

assert cluster.verify_sync()  # all nodes have identical params
```

The coordinator stores a loss history log so any node can catch up by
replaying missed generations — no param communication needed. See
`examples/train_cluster_unreliable.py` for a simulated churn scenario.

## Development

```bash
uv run pytest
```

## License

MIT
