# zerograd

`zerograd` is a JAX + Optax library for zero-gradient evolutionary optimization. It evaluates a pure loss callback against deterministic, factor-only candidate perturbations, replays those same low-rank factors into pseudo-gradients, and applies an Optax transform transactionally.

## Design principles

- **No objective gradients:** fitness-only Evolution Strategies primitives; arbitrary JAX-compatible objectives may be evaluated.
- **No dense perturbation materialization:** candidate perturbations are PRNG-derived A/B low-rank factors, never materialized as dense matrices or tables during forward evaluation.
- **Explicit manifest identity:** parameter layouts and tie groups are user-controlled, not inferred from PyTree leaf order.
- **Transactional lifecycle:** `init` / `step` return new state only after candidate evaluation, factor replay, and Optax application complete.

## Installation

```bash
uv sync
uv pip install -e ".[dev]"
```

## Quick start

```python
import jax
import jax.numpy as jnp
import optax
from zerograd import Manifest, ManifestEntry, ParameterLayout, ZeroGrad

params = {
    "embed": {"weight": jnp.ones((128, 32))},
    "norm": {"scale": jnp.ones((32,))},
    "head": {"weight": jnp.ones((32, 128))},
}
manifest = Manifest(
    version=1,
    entries=(
        ManifestEntry(("embed", "weight"), ParameterLayout.TABLE, "token_embed"),
        ManifestEntry(("norm", "scale"), ParameterLayout.VECTOR, "norm_scale"),
        ManifestEntry(("head", "weight"), ParameterLayout.MATRIX, "head"),
    ),
)
optimizer = ZeroGrad(
    manifest,
    optax.adamw(learning_rate=3e-4),
    population_size=64,
    rank=8,
    sigma=0.01,
    seed=0,
    run_id="experiment-1",
)
state = optimizer.init(params)

def model_loss(params, candidate, batch, rng):
    # Use candidate helpers — never receive a perturbed parameter tree
    x = candidate.table_lookup(params, ("embed", "weight"), batch)
    x = candidate.vector(params, ("norm", "scale")) * x  # simplified
    logits = candidate.tied_logits(params, ("embed", "weight"), x)
    return jnp.mean(logits ** 2), None

params, state, metrics = optimizer.step(state, params, jnp.array([0, 1, 2]), model_loss)
print(f"gen {metrics.generation}: mean_loss={metrics.mean_loss:.4f}")
```

## Distributed multi-device evaluation

The ES population is embarrassingly parallel — each candidate's loss is
computed independently, and workers only share 1D fitness arrays.

```python
from zerograd import DistributedZeroGrad

cpu = jax.devices('cpu')[0]
gpu = jax.devices('gpu')[0]

opt = ZeroGrad(manifest, optax.adamw(1e-2), population_size=64, ...)
dist_opt = DistributedZeroGrad(opt, devices=[cpu, gpu], loss_fn=model_loss)

state = dist_opt.init(params)
for step in range(steps):
    params, state, metrics = dist_opt.step(state, params, batch)
```

Workers can be mixed across CPU and GPU, or multiple shards can share a
single GPU. For a 4× GPU node, pass all four devices — each gets a quarter
of the population. See `examples/train_distributed_cpu_gpu.py` and
`examples/train_distributed_dual_worker.py` for runnable demos.

For asymmetric compute (slow CPU + fast GPU, or mixed GPU generations),
pass ``weights`` to assign more candidates to faster devices:

```python
# GPU gets 4× more candidates than CPU
dist_opt = DistributedZeroGrad(opt, devices=[cpu, gpu], loss_fn=loss_fn, weights=[1, 4])

# Or auto-calibrate from measured device speed
dist_opt.calibrate(params, batch)
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
