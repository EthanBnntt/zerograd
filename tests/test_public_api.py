"""Smoke tests and branch coverage for :data:`zerograd.__all__` exports."""

from __future__ import annotations

import typing
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

import zerograd as zg
from zerograd import (
    IntEmbedding,
    ParameterPath,
    ParameterTree,
    StepMetrics,
    ZeroGrad,
    ZeroGradSlot,
    ZeroGradState,
    ZgLayerNorm,
    apply_bin_updates,
    bin_update_threshold,
    float_to_egg_i8,
    fused_linear_lut,
    int_conv2d,
    int_conv2d_flat,
    int_matmul,
    shape_antithetical_loss,
)
from zerograd._fused_lut import linear_lut_reference
from zerograd._nnx import model_zg_slot


def test_all_exports_importable():
    for name in zg.__all__:
        assert hasattr(zg, name), f"missing export {name!r}"


def test_type_aliases_are_usable():
    path: ParameterPath = ("w",)
    tree: ParameterTree = {"w": jnp.ones((2, 2))}
    assert path == ("w",)
    assert tree["w"].shape == (2, 2)

    def loss_fn(model: nnx.Module, batch) -> tuple[jax.Array, None]:
        return jnp.sum(batch), None

    _: zg.ModelLossFn = loss_fn


class TestPublicApiValidation:
    def test_mark_table_rejects_non_param(self):
        bad_param: Any = jnp.ones((2, 2))
        with pytest.raises(TypeError):
            zg.mark_table(bad_param)

    def test_shape_antithetical_loss_validation(self):
        with pytest.raises(ValueError):
            shape_antithetical_loss(jnp.ones((2, 2)))
        with pytest.raises(ValueError):
            shape_antithetical_loss(jnp.ones((3,)))

    def test_bin_update_threshold_validation(self):
        with pytest.raises(ValueError):
            bin_update_threshold(0.0, 4)
        with pytest.raises(ValueError):
            bin_update_threshold(0.5, 0)
        with pytest.raises(ValueError):
            bin_update_threshold(0.5, 4, rank=0)

    def test_apply_bin_updates_tree_branches(self):
        params = {"w": jnp.zeros((2,), dtype=jnp.int8), "frozen": jnp.ones((2,), dtype=jnp.int8)}
        evidence = {"w": jnp.array([100, -100], dtype=jnp.int32)}
        out = apply_bin_updates(params, evidence, 10)
        frozen = cast(jax.Array, out["frozen"])
        assert int(frozen[0]) == 1  # missing evidence key → unchanged

        with pytest.raises(TypeError):
            bad_threshold: Any = {"w": 1}
            apply_bin_updates(
                jnp.zeros((2,), dtype=jnp.int8),
                jnp.zeros((2,), dtype=jnp.int32),
                bad_threshold,
            )

        passthrough = apply_bin_updates(
            jnp.ones((2,), dtype=jnp.float32),
            jnp.ones((2,), dtype=jnp.float32),
            1,
        )
        np.testing.assert_array_equal(np.asarray(passthrough), np.ones((2,), dtype=np.float32))
        bad_params: Any = 1
        bad_evidence: Any = 2
        bad_scalar_threshold: Any = 3
        assert apply_bin_updates(bad_params, bad_evidence, bad_scalar_threshold) == 1

    def test_threshold_tree_includes_non_manifest_leaves(self):
        params = {"w": jnp.zeros((2, 2), dtype=jnp.int8), "extra": jnp.zeros((2,), dtype=jnp.int8)}
        manifest = zg.Manifest(
            version=1,
            entries=(zg.ManifestEntry(("w",), zg.ParameterLayout.MATRIX, "w"),),
        )
        tree = zg.threshold_tree_for_manifest(params, manifest, alpha=0.5, num_directions=4)
        assert tree["extra"] == 2**30

    def test_int_avg_pool2d_rejects_wrong_rank(self):
        with pytest.raises(ValueError):
            zg.int_avg_pool2d(jnp.ones((4, 4), dtype=jnp.int8), (2, 2), (2, 2))

    def test_fused_linear_lut_operand_validation(self):
        x = jnp.zeros((2, 16), dtype=jnp.int8)
        w = jnp.zeros((16, 8), dtype=jnp.int8)
        table = jnp.zeros((256,), dtype=jnp.int8)
        with pytest.raises(ValueError):
            fused_linear_lut(x, w, table, block_k=5)

    @pytest.mark.skipif(jax.default_backend() == "gpu", reason="CPU-only fused fallback error")
    def test_fused_linear_lut_requires_gpu_when_no_fallback(self):
        x = jnp.zeros((2, 16), dtype=jnp.int8)
        w = jnp.zeros((16, 8), dtype=jnp.int8)
        table = jnp.zeros((256,), dtype=jnp.int8)
        with pytest.raises(RuntimeError):
            fused_linear_lut(x, w, table, mode="fused", allow_fallback=False)


class TestIntegerPublicApi:
    def test_float_to_egg_i8_scales_and_clips(self):
        x = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)
        y = float_to_egg_i8(x, scale=16.0)
        assert y.dtype == jnp.int8
        np.testing.assert_array_equal(y, np.array([0, 8, 16], dtype=np.int8))

    def test_int_matmul_int8_and_narrow_paths(self):
        a = jnp.ones((4, 8), dtype=jnp.int8)
        b = jnp.ones((8, 2), dtype=jnp.int8)
        out = int_matmul(a, b)
        assert out.dtype == jnp.int32
        assert out.shape == (4, 2)

        # int4-in-int8 operands use the int8 WMMA path.
        a4 = jnp.array([-8, -1, 0, 7], dtype=jnp.int8)
        b4 = jnp.array([1, 2, 3, 4], dtype=jnp.int8)
        row = int_matmul(a4[None, :], b4[:, None])
        assert row.shape == (1, 1)

        # Wider-than-int8 integers still use preferred_element_type=int32.
        a16 = jnp.ones((2, 2), dtype=jnp.int16)
        b16 = jnp.ones((2, 2), dtype=jnp.int16)
        out16 = int_matmul(a16, b16)
        assert out16.dtype == jnp.int32

    def test_int_matmul_rejects_float(self):
        with pytest.raises(TypeError):
            int_matmul(jnp.ones((2, 2)), jnp.ones((2, 2), dtype=jnp.float32))

    def test_int_conv2d_and_flat(self):
        x = jnp.ones((1, 4, 4, 2), dtype=jnp.int8)
        k = jnp.ones((3, 3, 2, 4), dtype=jnp.int8)
        y = int_conv2d(x, k)
        assert y.dtype == jnp.int32
        assert y.shape[0] == 1 and y.shape[-1] == 4

        flat = k.reshape(3 * 3 * 2, 4)
        y2 = int_conv2d_flat(x, flat, kernel_hw=(3, 3))
        np.testing.assert_array_equal(np.asarray(y), np.asarray(y2))

    def test_int_conv2d_validation(self):
        with pytest.raises(ValueError):
            int_conv2d(
                jnp.ones((1, 4, 4, 2), dtype=jnp.int8),
                jnp.ones((18, 4), dtype=jnp.int8),
            )
        with pytest.raises(ValueError):
            int_conv2d(
                jnp.ones((1, 4, 4, 3), dtype=jnp.int8),
                jnp.ones((3, 3, 2, 4), dtype=jnp.int8),
            )
        with pytest.raises(ValueError):
            int_conv2d(
                jnp.ones((4, 4, 2), dtype=jnp.int8),
                jnp.ones((3, 3, 2, 4), dtype=jnp.int8),
            )

    def test_int_conv2d_flat_validation(self):
        with pytest.raises(ValueError):
            int_conv2d_flat(
                jnp.ones((4, 4, 2), dtype=jnp.int8),
                jnp.ones((18, 4), dtype=jnp.int8),
                kernel_hw=(3, 3),
            )
        with pytest.raises(TypeError):
            int_conv2d_flat(
                jnp.ones((1, 4, 4, 2), dtype=jnp.float32),
                jnp.ones((18, 4), dtype=jnp.int8),
                kernel_hw=(3, 3),
            )
        with pytest.raises(ValueError):
            int_conv2d_flat(
                jnp.ones((1, 4, 4, 2), dtype=jnp.int8),
                jnp.ones((17, 4), dtype=jnp.int8),
                kernel_hw=(3, 3),
            )

    def test_int_conv2d_rejects_flat_kernel(self):
        with pytest.raises(ValueError):
            int_conv2d(
                jnp.ones((1, 4, 4, 2), dtype=jnp.int8),
                jnp.ones((18, 4), dtype=jnp.int8),
            )


class TestFusedLutPublicApi:
    def test_reference_mode_matches_linear_lut_reference(self):
        k = 16
        x = jax.random.randint(jax.random.key(0), (2, 3, k), -40, 40, dtype=jnp.int8)
        w = jax.random.randint(jax.random.key(1), (k, 8), -40, 40, dtype=jnp.int8)
        table = jax.random.randint(jax.random.key(2), (256,), -127, 127, dtype=jnp.int8)
        bias = jnp.zeros((8,), dtype=jnp.int32)
        ref = linear_lut_reference(x, w, table, bias=bias)
        got = fused_linear_lut(x, w, table, bias=bias, mode="reference")
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

    def test_invalid_mode_raises(self):
        x = jnp.zeros((2, 16), dtype=jnp.int8)
        w = jnp.zeros((16, 8), dtype=jnp.int8)
        table = jnp.zeros((256,), dtype=jnp.int8)
        with pytest.raises(ValueError):
            fused_linear_lut(x, w, table, mode="nope")


class TestNnxPublicApi:
    def test_layernorm_surgery_replaces_with_zg_layernorm(self):
        class M(nnx.Module):
            ln: nnx.LayerNorm | ZgLayerNorm

            def __init__(self, rngs: nnx.Rngs):
                self.ln = nnx.LayerNorm(4, rngs=rngs)

            def __call__(self, x):
                return self.ln(x)

        model = M(nnx.Rngs(0))
        model, manifest = zg.apply_surgery(model, rank=2, sigma=0.05)
        assert isinstance(model.ln, ZgLayerNorm)
        y = model(jnp.ones((2, 4)))
        assert y.shape == (2, 4)
        assert any("ln" in ".".join(e.path) for e in manifest.entries)

    def test_int_embedding_attend_uses_public_int_matmul(self):
        emb = IntEmbedding(8, 4, rngs=nnx.Rngs(0))
        q = jnp.ones((3, 4), dtype=jnp.int8)
        logits = emb.attend(q)
        assert logits.shape == (3, 8)
        assert logits.dtype == jnp.int32


class TestOptimizerPublicTypes:
    def test_step_metrics_and_state_fields(self):
        class Tiny(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.l = nnx.Linear(4, 2, rngs=rngs)

            def __call__(self, x):
                return self.l(x)

        model = Tiny(nnx.Rngs(0))
        opt = ZeroGrad(optax.sgd(0.1), population_size=4, rank=1, sigma=0.05, seed=0, run_id="api")
        state: ZeroGradState = opt.init(model)
        assert state.generation == 0
        assert isinstance(model_zg_slot(model), ZeroGradSlot)

        def loss_fn(m, batch):
            return jnp.mean(m(batch) ** 2), None

        _, new_state, metrics = opt.step(state, model, jnp.ones((3, 4)), loss_fn)
        assert isinstance(metrics, StepMetrics)
        assert metrics.population_size == 4
        assert new_state.generation == 1
        assert typing.get_type_hints(StepMetrics)
