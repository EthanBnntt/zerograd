"""Tests for Flax NNX surgery and the NNX-first ZeroGrad API."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from zerograd import (
    ClusterZeroGrad,
    IntEmbedding,
    IntLinear,
    ParameterLayout,
    ZeroGrad,
    ZgEmbed,
    ZgIntEmbedding,
    ZgIntLinear,
    ZgLinear,
    ZgTable,
    ZgVector,
    apply_surgery,
    mark_table,
)
from zerograd._integer import float_to_int, int_relu
from zerograd._nnx import params_pure_dict


class TinyMLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(2, 8, rngs=rngs)
        self.l2 = nnx.Linear(8, 1, rngs=rngs)

    def __call__(self, x):
        return self.l2(nnx.relu(self.l1(x)))


class EmbedModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.emb = nnx.Embed(16, 4, rngs=rngs)
        self.head = nnx.Linear(4, 3, rngs=rngs)

    def __call__(self, idx):
        return self.head(self.emb(idx))


class BareParamModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.vec = nnx.Param(jnp.ones((4,)))
        self.table = mark_table(nnx.Param(jnp.ones((8, 4))))
        self.proj = nnx.Linear(4, 2, rngs=rngs)

    def __call__(self, idx):
        # After surgery: vec is ZgVector (array-coercible), table is ZgTable.
        gathered = self.table[idx]
        return self.proj(gathered + self.vec)


class TestSurgery:
    def test_replaces_linear_with_zg_linear(self):
        model = TinyMLP(nnx.Rngs(0))
        model, manifest = apply_surgery(model, rank=2, sigma=0.05)
        assert isinstance(model.l1, ZgLinear)
        assert isinstance(model.l2, ZgLinear)
        assert hasattr(model, "zg_slot")
        paths = {e.path for e in manifest.entries}
        assert ("l1", "kernel") in paths
        assert ("l1", "bias") in paths
        assert ("l2", "kernel") in paths

    def test_replaces_embed(self):
        model = EmbedModel(nnx.Rngs(0))
        model, manifest = apply_surgery(model, rank=2, sigma=0.05)
        assert isinstance(model.emb, ZgEmbed)
        assert any(e.layout is ParameterLayout.TABLE for e in manifest.entries)

    def test_replaces_int_embedding(self):
        class IntEmbedModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.emb = IntEmbedding(16, 4, rngs=rngs)
                self.head = IntLinear(4, 3, act_dtype=jnp.int32, rngs=rngs)

            def __call__(self, idx):
                return self.head(self.emb(idx))

        model = IntEmbedModel(nnx.Rngs(0))
        idx = jnp.arange(4, dtype=jnp.int32)
        before = model(idx)
        model, manifest = apply_surgery(model, rank=2, sigma=0.05, integer_es=True)
        assert isinstance(model.emb, ZgIntEmbedding)
        assert isinstance(model.head, ZgIntLinear)
        assert ("emb", "embedding") in {e.path for e in manifest.entries}
        assert any(e.layout is ParameterLayout.TABLE for e in manifest.entries)
        model.zg_slot.enabled = False
        after = model(idx)
        assert jnp.array_equal(before, after)

    def test_wraps_bare_vector_and_marked_table(self):
        model = BareParamModel(nnx.Rngs(0))
        model, manifest = apply_surgery(model, rank=2, sigma=0.05)
        assert isinstance(model.vec, ZgVector)
        assert isinstance(model.table, ZgTable)
        layouts = {e.path: e.layout for e in manifest.entries}
        assert layouts[("vec", "param")] is ParameterLayout.VECTOR
        assert layouts[("table", "param")] is ParameterLayout.TABLE

    def test_bare_matrix_param_rejected(self):
        class Bad(nnx.Module):
            def __init__(self):
                self.w = nnx.Param(jnp.ones((3, 3)))

        model = Bad()
        with pytest.raises(ValueError, match="mark_table"):
            apply_surgery(model, rank=2, sigma=0.05)

    def test_eval_path_unperturbed(self):
        model = TinyMLP(nnx.Rngs(0))
        x = jnp.ones((4, 2))
        before = model(x)
        model, _ = apply_surgery(model, rank=2, sigma=0.5)
        model.zg_slot.enabled = False
        after = model(x)
        assert jnp.allclose(before, after)

    def test_tied_embed_attend(self):
        model = EmbedModel(nnx.Rngs(0))
        model, _ = apply_surgery(model, rank=2, sigma=0.05)
        model.zg_slot.enabled = False
        q = jnp.ones((2, 4))
        logits = model.emb.attend(q)
        assert logits.shape == (2, 16)

    def test_replaces_int_linear(self):
        class TinyInt(nnx.Module):
            def __init__(self, rngs):
                self.l1 = IntLinear(2, 8, rngs=rngs)
                self.l2 = IntLinear(8, 1, rngs=rngs)

            def __call__(self, x):
                return self.l2(int_relu(self.l1(x)))

        model = TinyInt(nnx.Rngs(0))
        assert model.l1.kernel[...].dtype == jnp.int8
        model, manifest = apply_surgery(model, rank=2, sigma=1.0)
        assert isinstance(model.l1, ZgIntLinear)
        assert isinstance(model.l2, ZgIntLinear)
        assert any(e.layout is ParameterLayout.MATRIX for e in manifest.entries)

        x = float_to_int(jnp.ones((4, 2)))
        model.zg_slot.enabled = False
        y = model(x)
        assert y.dtype == jnp.int8


class TestNnxOptimizer:
    def test_xor_learns(self):
        x = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = jnp.array([0, 1, 1, 0])

        model = TinyMLP(nnx.Rngs(0))
        opt = ZeroGrad(
            optax.adamw(1e-2, weight_decay=0.0),
            population_size=32,
            rank=4,
            sigma=0.1,
            seed=0,
            run_id="nnx-xor",
        )
        state = opt.init(model)

        def loss_fn(model, batch):
            bx, by = batch
            logits = jnp.squeeze(model(bx), -1)
            return jnp.mean(
                optax.sigmoid_binary_cross_entropy(logits, by.astype(jnp.float32))
            ), None

        for _ in range(300):
            model, state, _ = opt.step(state, model, (x, y), loss_fn)

        logits = jnp.squeeze(model(x), -1)
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)
        acc = float(jnp.mean(preds == y))
        assert acc >= 0.75

    def test_step_updates_params(self):
        model = TinyMLP(nnx.Rngs(0))
        opt = ZeroGrad(
            optax.sgd(0.1),
            population_size=8,
            rank=2,
            sigma=0.1,
            seed=1,
            run_id="nnx-step",
        )
        state = opt.init(model)
        before = params_pure_dict(model)

        def loss_fn(model, batch):
            return jnp.mean(model(batch) ** 2), None

        model, state, metrics = opt.step(state, model, jnp.ones((4, 2)), loss_fn)
        after = params_pure_dict(model)
        changed = jax.tree.map(lambda a, b: bool(jnp.any(a != b)), before, after)
        assert any(jax.tree.leaves(changed))
        assert state.generation == 1
        assert metrics.population_size == 8

    def test_int_linear_step_keeps_int8_weights(self):
        class TinyInt(nnx.Module):
            def __init__(self, rngs):
                self.l1 = IntLinear(2, 8, rngs=rngs)
                self.l2 = IntLinear(8, 1, rngs=rngs)

            def __call__(self, x):
                return self.l2(int_relu(self.l1(x)))

        model = TinyInt(nnx.Rngs(0))
        opt = ZeroGrad(
            optax.sgd(1.0),
            population_size=8,
            rank=2,
            sigma=1.0,
            seed=2,
            run_id="nnx-int",
        )
        state = opt.init(model)

        def loss_fn(model, batch):
            return jnp.mean(model(batch).astype(jnp.float32) ** 2), None

        x = float_to_int(jnp.ones((4, 2)))
        model, state, _ = opt.step(state, model, x, loss_fn)
        params = params_pure_dict(model)
        leaves = jax.tree.leaves(params)
        assert all(jnp.issubdtype(v.dtype, jnp.integer) for v in leaves)
        assert state.generation == 1

    def test_manifest_available_after_init(self):
        model = TinyMLP(nnx.Rngs(0))
        opt = ZeroGrad(optax.sgd(0.1), population_size=4, rank=2, sigma=0.1, seed=0, run_id="m")
        with pytest.raises(RuntimeError):
            _ = opt.manifest
        opt.init(model)
        assert len(opt.manifest.entries) >= 4


class TestNnxCluster:
    def test_cluster_stays_synced(self):
        def build_model(key):
            return TinyMLP(nnx.Rngs(key))

        def loss_fn(model, batch):
            return jnp.mean(model(batch) ** 2), None

        opt = ZeroGrad(
            optax.sgd(0.1),
            population_size=8,
            rank=2,
            sigma=0.1,
            seed=0,
            run_id="nnx-cluster",
        )
        cluster = ClusterZeroGrad(opt, build_model, loss_fn, seed=7, num_nodes=2)
        batch = jnp.ones((4, 2))
        for _ in range(3):
            cluster.step(batch)
        assert cluster.verify_sync()
