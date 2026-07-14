"""Regression tests for library issues #14-#26.

Each test pins the behaviour fixed in the corresponding GitHub issue so the
fix cannot silently regress.
"""

from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from zerograd import (
    ClusterZeroGrad,
    FaultTolerantCluster,
    Manifest,
    ManifestEntry,
    NodeStatus,
    ParameterLayout,
    ZeroGrad,
    candidate_key,
    matrix_factors,
    shape_centered_loss,
    validate_losses,
)
from zerograd._fault_tolerant import DEFAULT_MAX_LOSS_HISTORY


# ── Shared helpers ───────────────────────────────────────────────────────────

def _manifest():
    return Manifest(version=1, entries=(
        ManifestEntry(("w",), ParameterLayout.MATRIX, "w"),
        ManifestEntry(("v",), ParameterLayout.VECTOR, "v"),
    ))


def _params():
    return {"w": jnp.ones((4, 2)), "v": jnp.zeros((2,))}


def _make_opt(**kw):
    defaults = dict(
        manifest=_manifest(), transform=optax.adamw(0.01),
        population_size=8, rank=2, sigma=0.1, seed=42, run_id="regress",
    )
    defaults.update(kw)
    return ZeroGrad(**defaults)


def _trivial_loss_fn(p, candidate, batch, rng):
    return jnp.sum(p["w"] ** 2) + jnp.sum(p["v"] ** 2), None


def _build_params(key):
    k1, k2 = jax.random.split(key)
    return {"w": jax.random.normal(k1, (4, 2)) * 0.1, "v": jnp.zeros((2,))}


def _loss_fn(p, candidate, batch, rng):
    y = candidate.linear(p, ("w",), batch)
    return jnp.sum(y ** 2), None


def _batch():
    return jnp.ones((3, 4))


# ── Issue #14: integer losses silently truncated ─────────────────────────────

class TestIssue14IntegerLosses:
    def test_shape_centered_loss_rejects_integer(self):
        with pytest.raises(TypeError):
            shape_centered_loss(jnp.array([0, 1, 2, 3], dtype=jnp.int32), 0.1)

    def test_validate_losses_rejects_integer(self):
        with pytest.raises(TypeError):
            validate_losses(jnp.array([0, 1, 2, 3], dtype=jnp.int32))

    def test_float_losses_still_accepted(self):
        out = shape_centered_loss(jnp.array([0.0, 1.0, 2.0, 3.0]), 0.1)
        assert out.dtype == jnp.float32


# ── Issue #15: non-dict Mapping parameter trees ──────────────────────────────

class TestIssue15NonDictMapping:
    def test_mapping_proxy_type_tree_steps(self):
        params = {"block": MappingProxyType({"weight": jnp.ones((4, 4))})}
        manifest = Manifest(
            version=1,
            entries=(ManifestEntry(("block", "weight"), ParameterLayout.MATRIX, "w"),),
        )
        opt = ZeroGrad(manifest, optax.sgd(0.1), population_size=8, rank=2, sigma=0.1, seed=42, run_id="t")
        state = opt.init(params)
        x = jnp.ones((1, 4))

        def loss_fn(p, candidate, batch, rng):
            return jnp.sum(candidate.linear(p, ("block", "weight"), x) ** 2), None

        new_params, new_state, _ = opt.step(state, params, None, loss_fn)
        assert new_state.generation == 1
        assert bool(jnp.all(jnp.isfinite(new_params["block"]["weight"])))


# ── Issue #16: negative scalar array candidate IDs ──────────────────────────

class TestIssue16NegativeArrayCandidate:
    def test_negative_scalar_array_rejected(self):
        base = jax.random.key(0)
        with pytest.raises(ValueError):
            candidate_key(base, jnp.asarray(-1))

    def test_negative_consistency_with_int_branch(self):
        base = jax.random.key(0)
        with pytest.raises(ValueError):
            candidate_key(base, -1)
        with pytest.raises(ValueError):
            candidate_key(base, jnp.asarray(-5))

    def test_positive_scalar_array_still_works(self):
        base = jax.random.key(0)
        k_int = candidate_key(base, 7)
        k_arr = candidate_key(base, jnp.asarray(7))
        assert jnp.array_equal(k_int, k_arr)

    def test_vmap_path_still_works(self):
        # candidate_key is called inside evaluate_shard's vmap over a
        # non-negative candidate-id array; the negativity check must not
        # break tracing there.
        opt = _make_opt()
        losses = opt.evaluate_shard(
            _params(), 0, _trivial_loss_fn, None, jnp.arange(8, dtype=jnp.int32)
        )
        assert losses.shape == (8,)


# ── Issue #17: factor dtype parity across forward/replay ────────────────────

class TestIssue17FactorDtypeParity:
    def test_bf16_draw_matches_float32_draw_within_precision(self):
        key = jax.random.key(7)
        a16 = matrix_factors(key, (4, 2), rank=2, dtype=jnp.bfloat16)[0]
        a32 = matrix_factors(key, (4, 2), rank=2, dtype=jnp.float32)[0]
        # After the fix the bf16 draw is the float32 draw cast to bf16, so the
        # two agree to bf16 precision (~2 decimal digits).
        np.testing.assert_allclose(
            np.asarray(a16, dtype=np.float32), np.asarray(a32), rtol=0.05
        )

    def test_bf16_draw_differs_from_direct_bf16_draw(self):
        # The old (broken) behaviour drew directly in bf16, which produces an
        # entirely different sequence from the float32 draw.
        key = jax.random.key(7)
        a32 = matrix_factors(key, (4, 2), rank=2, dtype=jnp.float32)[0]
        direct_bf16 = jax.random.normal(key, (4, 2), dtype=jnp.bfloat16)
        assert not np.allclose(
            np.asarray(direct_bf16, dtype=np.float32), np.asarray(a32), rtol=0.05
        )


# ── Issue #21: bounded default loss history ─────────────────────────────────

class TestIssue21BoundedLossHistory:
    def test_default_is_finite(self):
        fc = FaultTolerantCluster(_make_opt(), _build_params, _loss_fn, seed=42, initial_nodes=1)
        assert fc.max_loss_history == DEFAULT_MAX_LOSS_HISTORY
        assert fc.max_loss_history > 0

    def test_explicit_zero_means_unlimited(self):
        fc = FaultTolerantCluster(
            _make_opt(), _build_params, _loss_fn, seed=42, initial_nodes=1, max_loss_history=0
        )
        for _ in range(5):
            fc.step(_batch())
        assert fc.loss_history_size == 5


# ── Issue #22: step no longer triple-validates, but still validates once ────

class TestIssue22ValidationAtBoundary:
    def test_step_rejects_invalid_params(self):
        opt = _make_opt()
        state = opt.init(_params())
        with pytest.raises(ValueError):
            opt.step(state, {"w": jnp.ones((4,))}, None, _trivial_loss_fn)

    def test_step_rejects_non_finite_losses(self):
        opt = _make_opt()
        state = opt.init(_params())

        def inf_loss(p, c, b, rng):
            return jnp.inf, None

        with pytest.raises(ValueError):
            opt.step(state, _params(), None, inf_loss)

    def test_step_rejects_integer_losses(self):
        opt = _make_opt()
        state = opt.init(_params())

        def int_loss(p, c, b, rng):
            return jnp.asarray(1, dtype=jnp.int32), None

        with pytest.raises(TypeError):
            opt.step(state, _params(), None, int_loss)


# ── Issue #23: NodeStatus slots + declared _shard_ids ───────────────────────

class TestIssue23NodeStatusSlots:
    def test_slots_reject_undeclared_attribute(self):
        ns = NodeStatus(name="x", node=None)
        with pytest.raises(AttributeError):
            ns.undeclared = 1

    def test_shard_ids_field_present_and_default_none(self):
        ns = NodeStatus(name="x", node=None)
        assert ns._shard_ids is None

    def test_shard_ids_assignable_after_construction(self):
        ns = NodeStatus(name="x", node=None)
        ns._shard_ids = jnp.arange(4, dtype=jnp.int32)
        assert ns._shard_ids.shape == (4,)


# ── Issue #24: partition/cluster input validation ──────────────────────────

class TestIssue24PartitionValidation:
    def test_compute_partition_sizes_rejects_zero_population(self):
        from zerograd import compute_partition_sizes

        with pytest.raises(ValueError):
            compute_partition_sizes(0, [1.0, 1.0])

    def test_compute_partition_sizes_rejects_negative_population(self):
        from zerograd import compute_partition_sizes

        with pytest.raises(ValueError):
            compute_partition_sizes(-4, [1.0, 1.0])

    def test_compute_partition_sizes_rejects_bool_population(self):
        from zerograd import compute_partition_sizes

        with pytest.raises(ValueError):
            compute_partition_sizes(True, [1.0])

    def test_cluster_rejects_bool_num_nodes(self):
        with pytest.raises(ValueError):
            ClusterZeroGrad(_make_opt(), _build_params, _loss_fn, seed=42, num_nodes=True)

    def test_fault_tolerant_rejects_bool_initial_nodes(self):
        with pytest.raises(ValueError):
            FaultTolerantCluster(_make_opt(), _build_params, _loss_fn, seed=42, initial_nodes=True)


# ── Issue #25: O(1) manifest entry/group lookups ─────────────────────────────

class TestIssue25ManifestCache:
    def test_entry_returns_correct_entry(self):
        m = _manifest()
        assert m.entry(("w",)).layout is ParameterLayout.MATRIX
        assert m.entry(("v",)).layout is ParameterLayout.VECTOR

    def test_group_index_returns_correct_index(self):
        m = _manifest()
        assert m.group_index("w") == 0
        assert m.group_index("v") == 1

    def test_unknown_path_raises_keyerror(self):
        with pytest.raises(KeyError):
            _manifest().entry(("missing",))

    def test_unknown_group_raises_keyerror(self):
        with pytest.raises(KeyError):
            _manifest().group_index("nope")

    def test_equality_ignores_internal_cache(self):
        # The lookup caches must not participate in equality/hash.
        m1 = _manifest()
        m2 = _manifest()
        assert m1 == m2
        assert hash(m1) == hash(m2)


# ── Issue #26: per-candidate RNG ────────────────────────────────────────────

class TestIssue26PerCandidateRng:
    def test_candidates_receive_distinct_rng(self):
        opt = _make_opt()

        def rng_loss(p, candidate, batch, rng):
            return jax.random.normal(rng, ()), None

        losses = opt.evaluate_shard(
            _params(), 0, rng_loss, None, jnp.arange(8, dtype=jnp.int32)
        )
        # With a shared default key all candidates would be identical; the fix
        # derives a per-candidate subkey so they differ.
        assert not bool(jnp.all(losses == losses[0]))

    def test_step_still_matches_evaluate_then_step_from_losses(self):
        opt = _make_opt()
        params = _params()
        state = opt.init(params)
        ids = jnp.arange(8, dtype=jnp.int32)
        losses = opt.evaluate_shard(params, 0, _trivial_loss_fn, None, ids)
        p1, _, _ = opt.step_from_losses(state, params, losses)
        p2, _, _ = opt.step(state, params, None, _trivial_loss_fn)
        assert jnp.array_equal(p1["w"], p2["w"])
