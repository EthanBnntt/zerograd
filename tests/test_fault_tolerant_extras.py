"""Supplementary fault-tolerant cluster tests: properties, resume replay, drift."""

import jax
import jax.numpy as jnp
import optax
import pytest

from zerograd import (
    FaultTolerantCluster,
    Manifest,
    ManifestEntry,
    ParameterLayout,
    ZeroGrad,
)


def _manifest():
    return Manifest(version=1, entries=(
        ManifestEntry(("w",), ParameterLayout.MATRIX, "w"),
        ManifestEntry(("v",), ParameterLayout.VECTOR, "v"),
    ))


def _build_params(key):
    k1, k2 = jax.random.split(key)
    return {
        "w": jax.random.normal(k1, (4, 2)) * 0.1,
        "v": jnp.zeros((2,)),
    }


def _loss_fn(p, candidate, batch, rng):
    y = candidate.linear(p, ("w",), batch)
    y = y + candidate.vector(p, ("v",))
    return jnp.sum(y ** 2), None


def _make_opt(pop=16):
    return ZeroGrad(
        _manifest(), optax.adamw(1e-2),
        population_size=pop, rank=2, sigma=0.1, seed=42, run_id="ft-test",
    )


def _batch():
    return jnp.ones((3, 4))


class TestFaultTolerantProperties:
    def test_generation_and_history_track_steps(self):
        fc = FaultTolerantCluster(_make_opt(), _build_params, _loss_fn, seed=42, initial_nodes=2)
        assert fc.generation == 0
        assert fc.loss_history_size == 0
        fc.step(_batch())
        assert fc.generation == 1
        assert fc.loss_history_size == 1

    def test_init_returns_node_zero_state(self):
        fc = FaultTolerantCluster(_make_opt(), _build_params, _loss_fn, seed=42, initial_nodes=2)
        state = fc.init()
        assert state.generation == 0
        assert state is fc.nodes[0].state

    def test_active_nodes_excludes_paused(self):
        fc = FaultTolerantCluster(_make_opt(), _build_params, _loss_fn, seed=42, initial_nodes=3)
        assert len(fc.active_nodes) == 3
        fc.pause_node(0)
        assert len(fc.active_nodes) == 2
        assert fc.nodes[0] not in fc.active_nodes

    def test_communication_accounting(self):
        pop = 16
        fc = FaultTolerantCluster(
            _make_opt(pop=pop), _build_params, _loss_fn, seed=42, initial_nodes=2,
        )
        assert fc.losses_bytes_per_step == pop * 4
        param_count = sum(v.size for v in jax.tree_util.tree_leaves(fc.nodes[0].params))
        assert fc.params_bytes == param_count * 4

    def test_params_bytes_zero_when_no_nodes(self):
        fc = FaultTolerantCluster(_make_opt(), _build_params, _loss_fn, seed=42, initial_nodes=1)
        fc.remove_node(0)
        assert fc.params_bytes == 0


class TestResumeKeepsNodeSynced:
    def test_resume_after_pause_stays_synced(self):
        # Paused nodes continue stepping (they receive gathered losses), so a
        # resumed node is already in sync — verify_sync must hold.
        fc = FaultTolerantCluster(_make_opt(), _build_params, _loss_fn, seed=42, initial_nodes=3)
        for _ in range(4):
            fc.step(_batch())
        fc.pause_node(1)
        for _ in range(3):
            fc.step(_batch())
        fc.resume_node(1)
        assert fc.get_status(1).last_generation == fc.generation
        assert fc.verify_sync()

    def test_resume_marks_node_active(self):
        fc = FaultTolerantCluster(_make_opt(), _build_params, _loss_fn, seed=42, initial_nodes=2)
        fc.pause_node(0)
        assert not fc.get_status(0).active
        fc.resume_node(0)
        assert fc.get_status(0).active
        assert not fc.get_status(0).paused


class TestVerificationDrift:
    def test_verify_sync_detects_drift(self):
        fc = FaultTolerantCluster(_make_opt(), _build_params, _loss_fn, seed=42, initial_nodes=2)
        fc.step(_batch())
        # Corrupt node 1's params so sync breaks.
        fc.nodes[1]._params = {"w": jnp.ones((4, 2)), "v": jnp.zeros((2,))}
        assert not fc.verify_sync()

    def test_verify_against_single_detects_mismatch(self):
        fc = FaultTolerantCluster(_make_opt(), _build_params, _loss_fn, seed=42, initial_nodes=1)
        fc.step(_batch())
        # A deliberately different baseline.
        baseline = _build_params(jax.random.key(999))
        assert not fc.verify_against_single(baseline)

    def test_verify_sync_trivially_true_with_one_node(self):
        fc = FaultTolerantCluster(_make_opt(), _build_params, _loss_fn, seed=42, initial_nodes=1)
        fc.step(_batch())
        assert fc.verify_sync()
