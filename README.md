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

## Development

```bash
uv run pytest
```

## License

MIT
