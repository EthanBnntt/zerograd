"""Train a Vision Transformer on CIFAR-10: ZeroGrad vs AdamW, bf16 vs 4-bit QAT.

Four-way comparison:

  1. ZeroGrad + bfloat16 — ES pseudo-gradients, bf16 forward pass.
  2. ZeroGrad + 4-bit — ES pseudo-gradients, 4-bit quantized activations, NO STE.
  3. AdamW + bfloat16 — standard backprop gradients, bf16 forward pass.
  4. AdamW + 4-bit — standard backprop, 4-bit quantized activations, WITH STE.

The key finding: ZeroGrad trains through jnp.round() without the Straight-Through
Estimator because it never differentiates the forward pass. AdamW requires STE
for 4-bit QAT because round() has zero gradient. We compare whether the
quantization penalty differs between the two optimization methods.

ViT: 4×4 patches (64+CLS=65 tokens), 2 layers, 4 heads, embed 64, MLP 128.
~75K parameters. Trained on CIFAR-10 (50K train, 10K test).

    uv run python examples/train_vit_cifar10.py [--steps N] [--batch N]
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import optax

from zerograd import Manifest, ManifestEntry, ParameterLayout, ZeroGrad

from _checkpoint import save_checkpoint
from _data import load_cifar10

# ── ViT config ────────────────────────────────────────────────────────────────
PATCH_SIZE = 4
IMG_SIZE = 32
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3
EMBED_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
MLP_DIM = 128
NUM_CLASSES = 10
NUM_TOKENS = NUM_PATCHES + 1

# ── 4-bit quantization ───────────────────────────────────────────────────────
QMIN = -8
QMAX = 7


def quantize_4bit(x: jax.Array) -> jax.Array:
    """Per-tensor 4-bit symmetric quantization with dynamic scale."""
    scale = jnp.max(jnp.abs(x)) / QMAX + 1e-8
    return jnp.round(jnp.clip(x / scale, QMIN, QMAX)) * scale


def quantize_4bit_ste(x: jax.Array) -> jax.Array:
    """4-bit quantization with Straight-Through Estimator for backprop."""
    scale = jnp.max(jnp.abs(x)) / QMAX + 1e-8
    x_q = jnp.round(jnp.clip(x / scale, QMIN, QMAX)) * scale
    return x + jax.lax.stop_gradient(x_q - x)


# ── Patch extraction ──────────────────────────────────────────────────────────
def extract_patches(images: jax.Array) -> jax.Array:
    imgs = images.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    B = imgs.shape[0]
    patches = imgs.reshape(B, IMG_SIZE // PATCH_SIZE, PATCH_SIZE,
                           IMG_SIZE // PATCH_SIZE, PATCH_SIZE, 3)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    return patches.reshape(B, NUM_PATCHES, PATCH_DIM)


def layer_norm(x, scale, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


# ── Parameter construction ────────────────────────────────────────────────────
def build_params(key):
    keys = jax.random.split(key, 20)
    init = 0.02
    params = {
        "patch_embed": {"weight": jax.random.normal(keys[0], (PATCH_DIM, EMBED_DIM)) * init},
        "cls_token": jax.random.normal(keys[1], (EMBED_DIM,)) * init,
        "pos_embed": jax.random.normal(keys[2], (NUM_TOKENS, EMBED_DIM)) * init,
        "ln_f": {"scale": jnp.ones((EMBED_DIM,)), "bias": jnp.zeros((EMBED_DIM,))},
        "head": {"weight": jax.random.normal(keys[3], (EMBED_DIM, NUM_CLASSES)) * init,
                 "bias": jnp.zeros((NUM_CLASSES,))},
    }
    for i in range(NUM_LAYERS):
        lk = keys[4 + i * 2]
        params[f"layer{i}"] = {
            "q": {"weight": jax.random.normal(jax.random.fold_in(lk, 0), (EMBED_DIM, EMBED_DIM)) * init},
            "k": {"weight": jax.random.normal(jax.random.fold_in(lk, 1), (EMBED_DIM, EMBED_DIM)) * init},
            "v": {"weight": jax.random.normal(jax.random.fold_in(lk, 2), (EMBED_DIM, EMBED_DIM)) * init},
            "o": {"weight": jax.random.normal(jax.random.fold_in(lk, 3), (EMBED_DIM, EMBED_DIM)) * init},
            "mlp1": {"weight": jax.random.normal(jax.random.fold_in(lk, 4), (EMBED_DIM, MLP_DIM)) * init},
            "mlp2": {"weight": jax.random.normal(jax.random.fold_in(lk, 5), (MLP_DIM, EMBED_DIM)) * init},
            "ln1": {"scale": jnp.ones((EMBED_DIM,)), "bias": jnp.zeros((EMBED_DIM,))},
            "ln2": {"scale": jnp.ones((EMBED_DIM,)), "bias": jnp.zeros((EMBED_DIM,))},
        }
    return params


def build_manifest():
    entries = [
        ManifestEntry(("patch_embed", "weight"), ParameterLayout.MATRIX, "patch_embed"),
        ManifestEntry(("cls_token",), ParameterLayout.VECTOR, "cls_token"),
        ManifestEntry(("pos_embed",), ParameterLayout.TABLE, "pos_embed"),
        ManifestEntry(("ln_f", "scale"), ParameterLayout.VECTOR, "ln_f_scale"),
        ManifestEntry(("ln_f", "bias"), ParameterLayout.VECTOR, "ln_f_bias"),
        ManifestEntry(("head", "weight"), ParameterLayout.MATRIX, "head_w"),
        ManifestEntry(("head", "bias"), ParameterLayout.VECTOR, "head_b"),
    ]
    for i in range(NUM_LAYERS):
        p = f"layer{i}"
        entries.extend([
            ManifestEntry((p, "q", "weight"), ParameterLayout.MATRIX, f"{p}_q"),
            ManifestEntry((p, "k", "weight"), ParameterLayout.MATRIX, f"{p}_k"),
            ManifestEntry((p, "v", "weight"), ParameterLayout.MATRIX, f"{p}_v"),
            ManifestEntry((p, "o", "weight"), ParameterLayout.MATRIX, f"{p}_o"),
            ManifestEntry((p, "mlp1", "weight"), ParameterLayout.MATRIX, f"{p}_mlp1"),
            ManifestEntry((p, "mlp2", "weight"), ParameterLayout.MATRIX, f"{p}_mlp2"),
            ManifestEntry((p, "ln1", "scale"), ParameterLayout.VECTOR, f"{p}_ln1_s"),
            ManifestEntry((p, "ln1", "bias"), ParameterLayout.VECTOR, f"{p}_ln1_b"),
            ManifestEntry((p, "ln2", "scale"), ParameterLayout.VECTOR, f"{p}_ln2_s"),
            ManifestEntry((p, "ln2", "bias"), ParameterLayout.VECTOR, f"{p}_ln2_b"),
        ])
    return Manifest(version=1, entries=tuple(entries))


# ── ViT forward: ZeroGrad (candidate-based) ───────────────────────────────────
def vit_forward_zg(params, candidate, images, quantize, bf16):
    """Forward pass using CandidateContext for perturbed params."""
    p = params
    if bf16:
        p = jax.tree_util.tree_map(
            lambda v: v.astype(jnp.bfloat16) if isinstance(v, jax.Array) else v, params)

    B = images.shape[0]
    q = quantize_4bit if quantize else (lambda x: x)

    patches = extract_patches(images)
    if bf16:
        patches = patches.astype(jnp.bfloat16)
    x = candidate.linear(p, ("patch_embed", "weight"), patches)
    x = q(x)

    cls = candidate.vector(p, ("cls_token",))
    x = jnp.concatenate([jnp.broadcast_to(cls, (B, 1, EMBED_DIM)), x], axis=1)
    x = x + q(candidate.table_lookup(p, ("pos_embed",), jnp.arange(NUM_TOKENS)))

    for i in range(NUM_LAYERS):
        layer = f"layer{i}"
        residual = x
        h = layer_norm(x, candidate.vector(p, (layer, "ln1", "scale")),
                       candidate.vector(p, (layer, "ln1", "bias")))

        q_proj = q(candidate.linear(p, (layer, "q", "weight"), h))
        k_proj = q(candidate.linear(p, (layer, "k", "weight"), h))
        v_proj = q(candidate.linear(p, (layer, "v", "weight"), h))

        qh = q_proj.reshape(B, NUM_TOKENS, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        kh = k_proj.reshape(B, NUM_TOKENS, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        vh = v_proj.reshape(B, NUM_TOKENS, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        attn = jax.nn.softmax(
            jnp.einsum("bhnd,bhmd->bhnm", qh, kh) / jnp.sqrt(float(HEAD_DIM)), axis=-1)
        attn_out = q(candidate.linear(
            p, (layer, "o", "weight"),
            jnp.einsum("bhnm,bhmd->bhnd", attn, vh)
            .transpose(0, 2, 1, 3).reshape(B, NUM_TOKENS, EMBED_DIM)))

        x = residual + attn_out
        residual = x
        h = layer_norm(x, candidate.vector(p, (layer, "ln2", "scale")),
                       candidate.vector(p, (layer, "ln2", "bias")))
        h = q(candidate.linear(p, (layer, "mlp1", "weight"), h))
        h = jax.nn.gelu(h)
        h = q(candidate.linear(p, (layer, "mlp2", "weight"), h))
        x = residual + h

    x = layer_norm(x, candidate.vector(p, ("ln_f", "scale")),
                   candidate.vector(p, ("ln_f", "bias")))
    logits = candidate.linear(p, ("head", "weight"), x[:, 0, :])
    logits = logits + candidate.vector(p, ("head", "bias"))
    return logits.astype(jnp.float32)


# ── ViT forward: direct (for AdamW backprop) ─────────────────────────────────
def vit_forward_direct(params, images, quantize, bf16):
    """Forward pass using params directly (for backprop). Uses STE if quantize."""
    p = params
    if bf16:
        p = jax.tree_util.tree_map(
            lambda v: v.astype(jnp.bfloat16) if isinstance(v, jax.Array) else v, params)

    B = images.shape[0]
    # STE for backprop, plain round for ZeroGrad
    q = quantize_4bit_ste if quantize else (lambda x: x)

    patches = extract_patches(images)
    if bf16:
        patches = patches.astype(jnp.bfloat16)
    x = q(patches @ p["patch_embed"]["weight"])

    cls = jnp.broadcast_to(p["cls_token"], (B, 1, EMBED_DIM))
    x = jnp.concatenate([cls, x], axis=1)
    x = x + q(p["pos_embed"][jnp.arange(NUM_TOKENS)])

    for i in range(NUM_LAYERS):
        layer = f"layer{i}"
        residual = x
        h = layer_norm(x, p[layer]["ln1"]["scale"], p[layer]["ln1"]["bias"])

        q_proj = q(h @ p[layer]["q"]["weight"])
        k_proj = q(h @ p[layer]["k"]["weight"])
        v_proj = q(h @ p[layer]["v"]["weight"])

        qh = q_proj.reshape(B, NUM_TOKENS, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        kh = k_proj.reshape(B, NUM_TOKENS, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        vh = v_proj.reshape(B, NUM_TOKENS, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        attn = jax.nn.softmax(
            jnp.einsum("bhnd,bhmd->bhnm", qh, kh) / jnp.sqrt(float(HEAD_DIM)), axis=-1)
        attn_out = q(jnp.einsum("bhnm,bhmd->bhnd", attn, vh)
                     .transpose(0, 2, 1, 3).reshape(B, NUM_TOKENS, EMBED_DIM)
                     @ p[layer]["o"]["weight"])

        x = residual + attn_out
        residual = x
        h = layer_norm(x, p[layer]["ln2"]["scale"], p[layer]["ln2"]["bias"])
        h = q(h @ p[layer]["mlp1"]["weight"])
        h = jax.nn.gelu(h)
        h = q(h @ p[layer]["mlp2"]["weight"])
        x = residual + h

    x = layer_norm(x, p["ln_f"]["scale"], p["ln_f"]["bias"])
    logits = x[:, 0, :] @ p["head"]["weight"] + p["head"]["bias"]
    return logits.astype(jnp.float32)


# ── Evaluation (unperturbed, float32, no quantization) ────────────────────────
def evaluate(params, x_test, y_test):
    """Evaluate on test set in float32 without quantization."""
    logits = vit_forward_direct(params, x_test, quantize=False, bf16=False)
    return jnp.mean(jnp.argmax(logits, axis=-1) == y_test)


# ── ZeroGrad training ─────────────────────────────────────────────────────────
def train_zerograd(name, params, manifest, x_train, y_train, x_test, y_test,
                   steps, batch_size, seed, quantize, bf16):
    optimizer = ZeroGrad(
        manifest,
        optax.adamw(learning_rate=5e-3, weight_decay=0.0),
        population_size=64,
        rank=8,
        sigma=0.005,
        seed=seed,
        run_id=f"vit-{name}",
    )
    state = optimizer.init(params)
    key = jax.random.key(seed + 100)
    num_train = x_train.shape[0]

    def loss_fn(params, candidate, batch, rng):
        x, y = batch
        logits = vit_forward_zg(params, candidate, x, quantize=quantize, bf16=bf16)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y)), None

    history = []
    for step in range(steps):
        idx = jax.random.randint(jax.random.fold_in(key, step), (batch_size,), 0, num_train)
        batch = (x_train[idx], y_train[idx])
        t0 = time.time()
        params, state, metrics = optimizer.step(state, params, batch, loss_fn)
        dt = time.time() - t0
        if step % 25 == 0 or step == steps - 1:
            acc = evaluate(params, x_test, y_test)
            history.append((metrics.generation, float(metrics.mean_loss),
                            float(metrics.min_loss), float(acc), dt))
            print(f"  [{name}] gen {metrics.generation:3d}  "
                  f"loss={metrics.mean_loss:.4f}  acc={float(acc):.1%}  ({dt:.1f}s)")
    return params, history


# ── AdamW backprop training ───────────────────────────────────────────────────
def train_adamw(name, params, x_train, y_train, x_test, y_test,
                steps, batch_size, seed, quantize, bf16):
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.0)
    opt_state = optimizer.init(params)
    key = jax.random.key(seed + 200)
    num_train = x_train.shape[0]

    def loss_fn(params, batch):
        x, y = batch
        logits = vit_forward_direct(params, x, quantize=quantize, bf16=bf16)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    history = []
    for step in range(steps):
        idx = jax.random.randint(jax.random.fold_in(key, step), (batch_size,), 0, num_train)
        batch = (x_train[idx], y_train[idx])
        t0 = time.time()
        loss, grads = grad_fn(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        dt = time.time() - t0
        if step % 25 == 0 or step == steps - 1:
            acc = evaluate(params, x_test, y_test)
            history.append((step, float(loss), float(loss), float(acc), dt))
            print(f"  [{name}] step {step:3d}  loss={float(loss):.4f}  acc={float(acc):.1%}  ({dt:.1f}s)")
    return params, history


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train ViT on CIFAR-10: ZeroGrad vs AdamW, bf16 vs 4-bit QAT")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Directory to save each variant's params + history.")
    args = parser.parse_args()

    print("Loading CIFAR-10 ...")
    x_train, y_train, x_test, y_test = load_cifar10()
    x_train = jnp.array(x_train)
    y_train = jnp.array(y_train)
    x_test = jnp.array(x_test)
    y_test = jnp.array(y_test)
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    sample_params = build_params(jax.random.key(args.seed))
    total_params = sum(v.size for v in jax.tree_util.tree_leaves(sample_params))
    print(f"\nViT: {NUM_LAYERS}L, {NUM_HEADS}H, dim={EMBED_DIM}, mlp={MLP_DIM}, "
          f"{NUM_PATCHES} patches ({PATCH_SIZE}×{PATCH_SIZE})")
    print(f"Parameters: {total_params:,}")
    print(f"Steps: {args.steps}, batch: {args.batch}\n")

    manifest = build_manifest()
    results = {}

    variants = [
        ("zerograd-bf16",   "zerograd", False, True),
        ("zerograd-4bit",   "zerograd", True,  False),
        ("adamw-bf16",      "adamw",    False, True),
        ("adamw-4bit-ste",  "adamw",    True,  False),
    ]

    for name, method, quantize, bf16 in variants:
        print("=" * 70)
        ste_note = " (with STE)" if (method == "adamw" and quantize) else (" (no STE)" if (method == "zerograd" and quantize) else "")
        print(f"{name.upper()}{ste_note}")
        print("=" * 70)

        params = build_params(jax.random.key(args.seed))
        t_start = time.time()

        if method == "zerograd":
            params, history = train_zerograd(
                name, params, manifest, x_train, y_train, x_test, y_test,
                steps=args.steps, batch_size=args.batch, seed=args.seed,
                quantize=quantize, bf16=bf16)
        else:
            params, history = train_adamw(
                name, params, x_train, y_train, x_test, y_test,
                steps=args.steps, batch_size=args.batch, seed=args.seed,
                quantize=quantize, bf16=bf16)

        total_time = time.time() - t_start
        results[name] = {
            "final_acc": history[-1][3],
            "final_loss": history[-1][1],
            "total_time": total_time,
            "avg_step": sum(h[4] for h in history) / len(history),
            "history": history,
        }
        if args.checkpoint:
            # Save each variant's final params + history so a long run is not
            # lost on interruption (see issue #34).
            import os
            save_checkpoint(
                os.path.join(args.checkpoint, f"{name}.ckpt"),
                step=args.steps, params=params, state=None,
                extra={"history": history, "method": method},
            )
            print(f"  checkpoint saved: {args.checkpoint}/{name}.ckpt")
        print()

    # ── Comparison table ──────────────────────────────────────────────────────
    print("=" * 70)
    print("FINDINGS")
    print("=" * 70)
    print(f"\n{'Variant':<25} {'Acc':>8} {'Loss':>8} {'Time':>8} {'s/step':>8}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<25} {r['final_acc']:>7.1%} {r['final_loss']:>8.4f} "
              f"{r['total_time']:>7.1f}s {r['avg_step']:>7.2f}s")

    # ── Key comparisons ───────────────────────────────────────────────────────
    print("\nKey comparisons:")
    zg_bf16 = results["zerograd-bf16"]["final_acc"]
    zg_4bit = results["zerograd-4bit"]["final_acc"]
    aw_bf16 = results["adamw-bf16"]["final_acc"]
    aw_4bit = results["adamw-4bit-ste"]["final_acc"]

    print(f"  ZeroGrad bf16 → 4bit:   {zg_bf16:.1%} → {zg_4bit:.1%}  (Δ={zg_4bit-zg_bf16:+.1%}, no STE)")
    print(f"  AdamW   bf16 → 4bit:    {aw_bf16:.1%} → {aw_4bit:.1%}  (Δ={aw_4bit-aw_bf16:+.1%}, with STE)")
    print(f"  bf16  ZG vs AdamW:      {zg_bf16:.1%} vs {aw_bf16:.1%}  (Δ={zg_bf16-aw_bf16:+.1%})")
    print(f"  4bit  ZG vs AdamW:      {zg_4bit:.1%} vs {aw_4bit:.1%}  (Δ={zg_4bit-aw_4bit:+.1%})")
    print()


if __name__ == "__main__":
    main()
