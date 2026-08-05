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
import os
import time

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from zerograd import ZeroGrad, mark_table
from zerograd._nnx import params_pure_dict

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


class VitBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.q = nnx.Linear(EMBED_DIM, EMBED_DIM, use_bias=False, rngs=rngs)
        self.k = nnx.Linear(EMBED_DIM, EMBED_DIM, use_bias=False, rngs=rngs)
        self.v = nnx.Linear(EMBED_DIM, EMBED_DIM, use_bias=False, rngs=rngs)
        self.o = nnx.Linear(EMBED_DIM, EMBED_DIM, use_bias=False, rngs=rngs)
        self.mlp1 = nnx.Linear(EMBED_DIM, MLP_DIM, use_bias=False, rngs=rngs)
        self.mlp2 = nnx.Linear(MLP_DIM, EMBED_DIM, use_bias=False, rngs=rngs)
        self.ln1_scale = nnx.Param(jnp.ones((EMBED_DIM,)))
        self.ln1_bias = nnx.Param(jnp.zeros((EMBED_DIM,)))
        self.ln2_scale = nnx.Param(jnp.ones((EMBED_DIM,)))
        self.ln2_bias = nnx.Param(jnp.zeros((EMBED_DIM,)))

    def __call__(self, x, q):
        B = x.shape[0]
        residual = x
        h = layer_norm(x, self.ln1_scale[...], self.ln1_bias[...])
        q_proj = q(self.q(h))
        k_proj = q(self.k(h))
        v_proj = q(self.v(h))
        qh = q_proj.reshape(B, NUM_TOKENS, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        kh = k_proj.reshape(B, NUM_TOKENS, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        vh = v_proj.reshape(B, NUM_TOKENS, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        attn = jax.nn.softmax(
            jnp.einsum("bhnd,bhmd->bhnm", qh, kh) / jnp.sqrt(float(HEAD_DIM)), axis=-1)
        attn_out = q(
            self.o(
                jnp.einsum("bhnm,bhmd->bhnd", attn, vh)
                .transpose(0, 2, 1, 3)
                .reshape(B, NUM_TOKENS, EMBED_DIM)
            )
        )
        x = residual + attn_out
        residual = x
        h = layer_norm(x, self.ln2_scale[...], self.ln2_bias[...])
        h = q(self.mlp1(h))
        h = jax.nn.gelu(h)
        h = q(self.mlp2(h))
        return residual + h


class TinyVit(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, *, quantize: bool = False, bf16: bool = False):
        self.patch_embed = nnx.Linear(PATCH_DIM, EMBED_DIM, use_bias=False, rngs=rngs)
        self.cls_token = nnx.Param(jax.random.normal(rngs.params(), (EMBED_DIM,)) * 0.02)
        self.pos_embed = mark_table(
            nnx.Param(jax.random.normal(rngs.params(), (NUM_TOKENS, EMBED_DIM)) * 0.02)
        )
        self.blocks = nnx.List([VitBlock(rngs=rngs) for _ in range(NUM_LAYERS)])
        self.ln_f_scale = nnx.Param(jnp.ones((EMBED_DIM,)))
        self.ln_f_bias = nnx.Param(jnp.zeros((EMBED_DIM,)))
        self.head = nnx.Linear(EMBED_DIM, NUM_CLASSES, rngs=rngs)
        self.quantize = quantize
        self.bf16 = bf16

    def __call__(self, images):
        q = quantize_4bit if self.quantize else (lambda x: x)
        B = images.shape[0]
        patches = extract_patches(images)
        if self.bf16:
            patches = patches.astype(jnp.bfloat16)
        x = q(self.patch_embed(patches))
        cls = jnp.broadcast_to(self.cls_token[...], (B, 1, EMBED_DIM))
        x = jnp.concatenate([cls, x], axis=1)
        x = x + q(self.pos_embed[jnp.arange(NUM_TOKENS)])
        for block in self.blocks:
            x = block(x, q)
        x = layer_norm(x, self.ln_f_scale[...], self.ln_f_bias[...])
        return self.head(x[:, 0, :]).astype(jnp.float32)


# ── AdamW keeps a functional params tree (no CandidateContext) ────────────────
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


def vit_forward_direct(params, images, quantize, bf16):
    """Forward pass using params directly (for backprop). Uses STE if quantize."""
    p = params
    if bf16:
        p = jax.tree_util.tree_map(
            lambda v: v.astype(jnp.bfloat16) if isinstance(v, jax.Array) else v, params)

    B = images.shape[0]
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


def evaluate_model(model, x_test, y_test):
    was_q, was_bf16 = model.quantize, model.bf16
    model.quantize = False
    model.bf16 = False
    logits = model(x_test)
    model.quantize = was_q
    model.bf16 = was_bf16
    return jnp.mean(jnp.argmax(logits, axis=-1) == y_test)


def evaluate_params(params, x_test, y_test):
    logits = vit_forward_direct(params, x_test, quantize=False, bf16=False)
    return jnp.mean(jnp.argmax(logits, axis=-1) == y_test)


def train_zerograd(name, x_train, y_train, x_test, y_test,
                   steps, batch_size, seed, quantize, bf16):
    model = TinyVit(nnx.Rngs(seed), quantize=quantize, bf16=bf16)
    optimizer = ZeroGrad(
        optax.adamw(learning_rate=5e-3, weight_decay=0.0),
        population_size=64,
        rank=8,
        sigma=0.005,
        seed=seed,
        run_id=f"vit-{name}",
    )
    state = optimizer.init(model)
    key = jax.random.key(seed + 100)
    num_train = x_train.shape[0]

    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y)), None

    history = []
    for step in range(steps):
        idx = jax.random.randint(jax.random.fold_in(key, step), (batch_size,), 0, num_train)
        batch = (x_train[idx], y_train[idx])
        t0 = time.time()
        model, state, metrics = optimizer.step(state, model, batch, loss_fn)
        dt = time.time() - t0
        if step % 25 == 0 or step == steps - 1:
            acc = evaluate_model(model, x_test, y_test)
            history.append((metrics.generation, float(metrics.mean_loss),
                            float(metrics.min_loss), float(acc), dt))
            print(f"  [{name}] gen {metrics.generation:3d}  "
                  f"loss={metrics.mean_loss:.4f}  acc={float(acc):.1%}  ({dt:.1f}s)")
    return model, history


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
            acc = evaluate_params(params, x_test, y_test)
            history.append((step, float(loss), float(loss), float(acc), dt))
            print(f"  [{name}] step {step:3d}  loss={float(loss):.4f}  acc={float(acc):.1%}  ({dt:.1f}s)")
    return params, history


def main():
    parser = argparse.ArgumentParser(
        description="Train ViT on CIFAR-10: ZeroGrad vs AdamW, bf16 vs 4-bit QAT")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Directory to save each variant's params + history "
                             "(save-only; no --resume, unlike train_mnist/train_cifar10).")
    args = parser.parse_args()

    print("Loading CIFAR-10 ...")
    x_train, y_train, x_test, y_test = load_cifar10()
    x_train = jnp.array(x_train)
    y_train = jnp.array(y_train)
    x_test = jnp.array(x_test)
    y_test = jnp.array(y_test)
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    sample = TinyVit(nnx.Rngs(args.seed))
    total_params = sum(v.size for v in jax.tree_util.tree_leaves(params_pure_dict(sample)))
    print(f"\nViT: {NUM_LAYERS}L, {NUM_HEADS}H, dim={EMBED_DIM}, mlp={MLP_DIM}, "
          f"{NUM_PATCHES} patches ({PATCH_SIZE}×{PATCH_SIZE})")
    print(f"Parameters: {total_params:,}")
    print(f"Steps: {args.steps}, batch: {args.batch}\n")

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

        t_start = time.time()

        if method == "zerograd":
            artifact, history = train_zerograd(
                name, x_train, y_train, x_test, y_test,
                steps=args.steps, batch_size=args.batch, seed=args.seed,
                quantize=quantize, bf16=bf16)
            ckpt_params = params_pure_dict(artifact)
        else:
            params = build_params(jax.random.key(args.seed))
            artifact, history = train_adamw(
                name, params, x_train, y_train, x_test, y_test,
                steps=args.steps, batch_size=args.batch, seed=args.seed,
                quantize=quantize, bf16=bf16)
            ckpt_params = artifact

        total_time = time.time() - t_start
        results[name] = {
            "final_acc": history[-1][3],
            "final_loss": history[-1][1],
            "total_time": total_time,
            "avg_step": sum(h[4] for h in history) / len(history),
            "history": history,
        }
        if args.checkpoint:
            save_checkpoint(
                os.path.join(args.checkpoint, f"{name}.ckpt"),
                step=args.steps, params=ckpt_params, state=None,
                extra={"history": history, "method": method},
            )
            print(f"  checkpoint saved: {args.checkpoint}/{name}.ckpt")
        print()

    print("=" * 70)
    print("FINDINGS")
    print("=" * 70)
    print(f"\n{'Variant':<25} {'Acc':>8} {'Loss':>8} {'Time':>8} {'s/step':>8}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<25} {r['final_acc']:>7.1%} {r['final_loss']:>8.4f} "
              f"{r['total_time']:>7.1f}s {r['avg_step']:>7.2f}s")

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
