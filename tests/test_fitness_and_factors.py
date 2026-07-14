"""Validation and behavior tests for fitness shaping and factor generation."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from zerograd._factors import matrix_factors, scaled_factor, table_factors, vector_noise
from zerograd._fitness import shape_centered_loss, validate_losses


class TestShapeCenteredLoss:
    def test_rejects_multidimensional_losses(self):
        with pytest.raises(ValueError):
            shape_centered_loss(jnp.ones((3, 3)), 0.1)

    def test_rejects_too_few_losses(self):
        with pytest.raises(ValueError):
            shape_centered_loss(jnp.ones((1,)), 0.1)

    def test_rejects_nonpositive_sigma(self):
        with pytest.raises(ValueError):
            shape_centered_loss(jnp.ones((4,)), 0.0)

    def test_negative_sigma_rejected(self):
        with pytest.raises(ValueError):
            shape_centered_loss(jnp.ones((4,)), -0.1)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_sigma(self, bad):
        # Regression for issue #7: shape_centered_loss must reject non-finite
        # sigma (NaN <= 0 is False, so it previously slipped past the check).
        with pytest.raises(ValueError):
            shape_centered_loss(jnp.ones((4,)), bad)

    def test_shaped_weights_sum_to_zero(self):
        # Centered-loss weights are mean-subtracted, so they sum to zero.
        losses = jnp.array([0.5, 1.0, 2.0, 4.0])
        shaped = shape_centered_loss(losses, 0.2)
        assert abs(float(jnp.sum(shaped))) < 1e-6

    def test_below_mean_losses_get_positive_weight(self):
        losses = jnp.array([0.0, 1.0, 1.0, 1.0])
        shaped = shape_centered_loss(losses, 0.1)
        assert float(shaped[0]) > 0  # below-mean candidate → positive (descent)


class TestValidateLosses:
    def test_rejects_multidimensional(self):
        with pytest.raises(ValueError):
            validate_losses(jnp.ones((2, 2)))

    def test_rejects_too_few(self):
        with pytest.raises(ValueError):
            validate_losses(jnp.ones((1,)))

    def test_accepts_valid_losses(self):
        validate_losses(jnp.array([1.0, 2.0, 3.0]))

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            validate_losses(jnp.array([1.0, jnp.nan, 3.0, 4.0]))

    def test_rejects_negative_infinity(self):
        with pytest.raises(ValueError):
            validate_losses(jnp.array([1.0, -jnp.inf, 3.0, 4.0]))


class TestScaledFactor:
    def test_value_is_sigma_over_sqrt_rank(self):
        result = float(scaled_factor(4, 0.2, jnp.float32))
        assert abs(result - 0.2 / math.sqrt(4)) < 1e-6

    def test_rejects_nonpositive_rank(self):
        with pytest.raises(ValueError):
            scaled_factor(0, 0.1, jnp.float32)

    def test_rejects_bool_rank(self):
        with pytest.raises(ValueError):
            scaled_factor(True, 0.1, jnp.float32)

    def test_rejects_nonpositive_sigma(self):
        with pytest.raises(ValueError):
            scaled_factor(2, 0.0, jnp.float32)

    def test_rejects_non_finite_sigma(self):
        with pytest.raises(ValueError):
            scaled_factor(2, float("inf"), jnp.float32)

    def test_rejects_bool_sigma(self):
        with pytest.raises(ValueError):
            scaled_factor(2, True, jnp.float32)


class TestMatrixFactors:
    def test_shapes_match_matrix_layout(self):
        a, b = matrix_factors(jax_key(0), (8, 4), rank=2, dtype=jnp.float32)
        assert a.shape == (8, 2)
        assert b.shape == (2, 4)

    def test_rejects_wrong_ndim_shape(self):
        with pytest.raises(ValueError):
            matrix_factors(jax_key(0), (8,), rank=2, dtype=jnp.float32)

    def test_rejects_nonpositive_dimension(self):
        with pytest.raises(ValueError):
            matrix_factors(jax_key(0), (8, 0), rank=2, dtype=jnp.float32)

    def test_rejects_nonpositive_rank(self):
        with pytest.raises(ValueError):
            matrix_factors(jax_key(0), (8, 4), rank=0, dtype=jnp.float32)

    def test_deterministic_for_same_key(self):
        a1, b1 = matrix_factors(jax_key(7), (4, 3), rank=2, dtype=jnp.float32)
        a2, b2 = matrix_factors(jax_key(7), (4, 3), rank=2, dtype=jnp.float32)
        np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))
        np.testing.assert_array_equal(np.asarray(b1), np.asarray(b2))


class TestTableFactors:
    def test_shapes_match_table_layout(self):
        a, b = table_factors(jax_key(0), (16, 4), rank=3, dtype=jnp.float32)
        assert a.shape == (16, 3)
        assert b.shape == (4, 3)

    def test_rejects_wrong_ndim_shape(self):
        with pytest.raises(ValueError):
            table_factors(jax_key(0), (16,), rank=2, dtype=jnp.float32)


class TestVectorNoise:
    def test_shape_matches_vector_layout(self):
        noise = vector_noise(jax_key(0), (8,), dtype=jnp.float32)
        assert noise.shape == (8,)

    def test_rejects_wrong_ndim_shape(self):
        with pytest.raises(ValueError):
            vector_noise(jax_key(0), (8, 4), dtype=jnp.float32)

    def test_rejects_nonpositive_dimension(self):
        with pytest.raises(ValueError):
            vector_noise(jax_key(0), (0,), dtype=jnp.float32)

    def test_deterministic_for_same_key(self):
        n1 = vector_noise(jax_key(5), (8,), dtype=jnp.float32)
        n2 = vector_noise(jax_key(5), (8,), dtype=jnp.float32)
        np.testing.assert_array_equal(np.asarray(n1), np.asarray(n2))


def jax_key(seed):
    import jax
    return jax.random.key(seed)
