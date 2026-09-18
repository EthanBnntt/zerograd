"""A/B: int8 vs ternary hidden layers in the int-CNN CIFAR pipeline (dev tool).

Sweeps (kind, hidden, depth) on the SAME CNN frontend + training recipe to
isolate whether ternary weights, width, or depth drive ES learning. Prints a
test-accuracy trajectory per config.

    uv run python examples/_ternary_ab.py --steps 300
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import optax
from _data import load_cifar10
from flax import nnx

from zerograd import (
    IntConv,
    IntLinear,
    IntLinearLUT,
    IntLUT,
    TernaryLinearLUT,
    ZeroGrad,
    int_avg_pool2d,
)
from zerograd._nnx import disable_candidates

NC = 10
LS = 256.0


def images_to_int8(images):
    x = images.reshape(-1, 32, 32, 3).astype(jnp.float32)
    return jnp.clip(jnp.rint(x * 127.0), -128, 127).astype(jnp.int8)


def loss_fn(model, batch):
    x, y = batch
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(model(x), y)), None


def evaluate(model, x, y, bs=512):
    disable_candidates(model)
    correct, total = 0.0, 0
    for s in range(0, x.shape[0], bs):
        pred = jnp.argmax(model(x[s : s + bs]), axis=-1)
        correct += float(jnp.sum(pred == y[s : s + bs]))
        total += int(y[s : s + bs].shape[0])
    return correct / max(total, 1)


def make(kind, rngs, hidden, depth):
    class M(nnx.Module):
        def __init__(s, rngs):
            s.conv1 = IntConv(3, 32, 3, rngs=rngs)
            s.act1 = IntLUT(init="identity", rngs=rngs)
            s.conv2 = IntConv(32, 64, 3, rngs=rngs)
            s.act2 = IntLUT(init="identity", rngs=rngs)
            flat = 8 * 8 * 64
            dims = [flat] + [hidden] * depth
            cls = TernaryLinearLUT if kind == "ternary" else IntLinearLUT
            s.mlp = nnx.List([cls(dims[i], dims[i + 1], rngs=rngs) for i in range(depth)])
            s.head = IntLinear(hidden, NC, act_dtype=jnp.int32, rngs=rngs)

        def __call__(s, images):
            b = images.shape[0]
            x = images_to_int8(images)
            x = s.act1(s.conv1(x))
            x = int_avg_pool2d(x, (2, 2), (2, 2))
            x = s.act2(s.conv2(x))
            x = int_avg_pool2d(x, (2, 2), (2, 2))
            x = jnp.reshape(x, (b, -1))
            for layer in s.mlp:
                x = layer(x)
            return s.head(x).astype(jnp.float32) / LS

    return M(rngs)


def run(kind, hidden, depth, steps, batch, pop, data):
    xtr, ytr, xte, yte = data
    m = make(kind, nnx.Rngs(0), hidden, depth)
    opt = ZeroGrad(
        optax.adamw(1e-2),
        population_size=pop,
        rank=8,
        seed=0,
        run_id=f"ab-{kind}-{hidden}-{depth}",
        integer_es=True,
        sigma_shift=2,
        int_bits=8,
        candidate_chunk_size=16,
    )
    st = opt.init(m)
    key = jax.random.key(0)
    traj = []
    for i in range(steps):
        idx = jax.random.randint(jax.random.fold_in(key, i), (batch,), 0, xtr.shape[0])
        m, st, met = opt.step(st, m, (jax.device_put(xtr[idx]), jax.device_put(ytr[idx])), loss_fn)
        if i % 50 == 0 or i == steps - 1:
            traj.append((i, float(met.mean_loss), float(evaluate(m, xte, yte))))
    out = " ".join(f"({i},{l:.3f},{a:.3f})" for i, l, a in traj)
    print(f"{kind:8s} hidden={hidden:4d} depth={depth}: {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--population", type=int, default=256)
    args = ap.parse_args()
    xtr, ytr, xte, yte = load_cifar10()
    data = (jnp.asarray(xtr), jnp.asarray(ytr), jnp.asarray(xte), jnp.asarray(yte))
    print(f"backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    for kind, hidden, depth in [
        ("int8", 128, 1),
        ("ternary", 128, 1),
        ("int8", 256, 2),
        ("ternary", 256, 2),
    ]:
        run(kind, hidden, depth, args.steps, args.batch, args.population, data)


if __name__ == "__main__":
    main()
