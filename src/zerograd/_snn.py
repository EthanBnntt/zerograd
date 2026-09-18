"""Native spiking-neural-network modules for ZeroGrad.

ZeroGrad's factor-only ES turns out to be an excellent fit for spiking
neural networks — hard thresholds and non-differentiable spike/reset
dynamics are irrelevant to a gradient-free optimizer, and the low-rank
perturbations explore membrane dynamics (decay, threshold) as easily as
weights.  This module hosts the core SNN primitives natively:

* :class:`Leaky` — the leaky linear module: per-neuron temporal decay
  ``y_t = beta ⊙ y_{t-1} + x_t @ W + b`` (the membrane state; no
  threshold).  The decay ``beta`` is ES-trainable (sigmoid-logit param).
* :class:`Spiking` — the LIF neuron on top of it: leaky integration, hard
  threshold, subtractive reset, binary output ``s_t = 1[v_t ≥ thr]``.
  Threshold is ES-trainable (softplus-logit param); optional
  weight-norm pinning (default on) freezes each neuron's effective input
  norm at init so ES cannot profit from uniform weight growth — without
  it, SNN+ES reliably collapses to saturation or silence.
* :func:`encode_rate` / :func:`encode_poisson` — continuous inputs →
  binary spike trains.
* :func:`spike_count_logits` — readout spike trains → classification
  logits.

All parameters are ES-trainable through ordinary :func:`apply_surgery`
(no special-casing): the inner ``nnx.Linear`` is wrapped as ``ZgLinear``
(factor-only MATRIX perturbation inside ``__call__``) and the bare 1-D
``beta`` / threshold params are wrapped as ``ZgVector`` — read them with
:func:`_param_value`, never by indexing, so candidate perturbations stay
visible.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

__all__ = [
    "Leaky",
    "Spiking",
    "encode_rate",
    "encode_poisson",
    "spike_count_logits",
]


def _param_value(p) -> jax.Array:
    """Read a param that is a raw ``nnx.Param`` (pre-surgery) or a ZeroGrad
    wrapper (post-surgery, perturbation-aware)."""
    if isinstance(p, nnx.Module):  # ZgVector
        return p()
    return p[...]  # nnx.Param


def _leaky_scan(current: jax.Array, beta: jax.Array) -> jax.Array:
    """Serial temporal integration ``y_t = beta ⊙ y_{t-1} + current_t``.

    ``current: [B, T, D]`` (batched timesteps), ``beta: [D]`` → ``[B, T, D]``.
    """
    def step(v, c_t):
        v = beta * v + c_t
        return v, v  # the membrane IS the output of a leaky linear module

    _, y = jax.lax.scan(step, jnp.zeros(current.shape[0:1] + current.shape[2:],
                                       dtype=current.dtype),
                        current.transpose(1, 0, 2))
    return y.transpose(1, 0, 2)


def _lif_scan(current: jax.Array, beta: jax.Array, thr: jax.Array) -> jax.Array:
    """Serial LIF: integrate → hard threshold spike → subtractive reset.

    ``current: [B, T, D]``, ``beta``/``thr: [D]`` → binary ``[B, T, D]``.
    """
    def step(v, c_t):
        v = beta * v + c_t
        s = (v >= thr).astype(current.dtype)
        return v - s * thr, s

    _, spikes = jax.lax.scan(step, jnp.zeros(current.shape[0:1] + current.shape[2:],
                                             dtype=current.dtype),
                            current.transpose(1, 0, 2))
    return spikes.transpose(1, 0, 2)


# ── spike coding / readout ───────────────────────────────────────────────────


def encode_rate(x: jax.Array, t_len: int) -> jax.Array:
    """Deterministic rate coding: ``x [B, D] ∈ [0, 1]`` → ``[B, T, D]`` binary.

    Pixel intensity ``p`` produces ``floor(p·T)`` spikes at the *leading*
    timesteps of the window — noise-free fitness, the best pairing for ES.
    """
    if x.ndim != 2:
        raise ValueError(f"encode_rate expects [B, D], got {x.shape}")
    if t_len <= 0:
        raise ValueError(f"t_len must be positive, got {t_len}")
    thresholds = (jnp.arange(t_len, dtype=jnp.float32) + 0.5) / t_len
    return (x[:, None, :] > thresholds[None, :, None]).astype(jnp.float32)


def encode_poisson(x: jax.Array, t_len: int, rng: jax.Array) -> jax.Array:
    """Poisson rate coding: intensity = per-timestep firing probability.

    ``x [B, D] ∈ [0, 1]`` → ``[B, T, D]`` binary; stochastic, pass a per-call rng.
    """
    if x.ndim != 2:
        raise ValueError(f"encode_poisson expects [B, D], got {x.shape}")
    if t_len <= 0:
        raise ValueError(f"t_len must be positive, got {t_len}")
    u = jax.random.uniform(rng, (x.shape[0], t_len, x.shape[1]))
    return (u < x[:, None, :]).astype(jnp.float32)


def spike_count_logits(out_spikes: jax.Array) -> jax.Array:
    """Reduce readout spike trains ``[B, T, C]`` to spike-count logits ``[B, C]``."""
    return out_spikes.sum(axis=1)


# ── modules ─────────────────────────────────────────────────────────────────


class Leaky(nnx.Module):
    """Leaky linear module: ``y_t = beta ⊙ y_{t-1} + x_t @ W + b``.

    The temporal analogue of ``nnx.Linear`` — every output neuron is a
    leaky integrator (SNN membrane state) with per-neuron decay
    ``beta ∈ (0, 1)``.  ``beta`` is stored as a sigmoid logit so any ES move
    maps to a valid decay, and is ES-trainable (surgery wraps it as
    ``ZgVector``).  The inner ``nnx.Linear`` is wrapped as ``ZgLinear``,
    so weights and bias carry factor-only MATRIX/VECTOR perturbations —
    call ``self.lin(x)``, never index ``lin.kernel``, for the forward pass.

    Input/output: ``[B, T, Din] → [B, T, Dout]`` float currents (analog —
    this module does not spike; add :class:`Spiking` for that).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        leak: float = 0.9,
        use_bias: bool = False,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if not 0.0 < leak < 1.0:
            raise ValueError(f"leak must be in (0, 1), got {leak}")
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"in/out features must be positive, got {in_features}/{out_features}"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.dtype = dtype
        # Sigmoid logit of the initial leak, so sigmoid(beta_init) == leak.
        beta_init = math.log(leak / (1.0 - leak))
        self.beta = nnx.Param(
            jnp.full((self.out_features,), beta_init, dtype=dtype)
        )
        self.lin = nnx.Linear(
            in_features, out_features, use_bias=use_bias,
            param_dtype=dtype, rngs=rngs,
        )

    def beta_value(self) -> jax.Array:
        """Effective per-neuron decay in ``(0, 1)`` — ES-perturbation aware."""
        return jax.nn.sigmoid(_param_value(self.beta).astype(jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        """``[B, T, Din] → [B, T, Dout]``: currents in, decayed currents out."""
        if x.ndim != 3:
            raise ValueError(f"Leaky expects [B, T, Din], got {x.shape}")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Leaky expects Din={self.in_features}, got {x.shape[-1]}"
            )
        current = self.lin(x).astype(self.dtype)  # perturbing linear
        return _leaky_scan(current, self.beta_value())


class Spiking(nnx.Module):
    """LIF layer on top of the leaky linear module: ``[B, T, Din] → [B, T, Dout]``
    binary spike trains.

    Per timestep, per output neuron::

        v ← beta_j · v + x_t @ W[:, j] (+ b_j)      leaky integrate
        s_t = 1[v ≥ threshold_j]                     hard threshold
        v ← v − s_t · threshold_j                    subtractive reset

    All parameters are ES-trainable: weights/bias through the inner
    ``ZgLinear``, per-neuron decay ``beta`` (sigmoid logit) and threshold
    (softplus logit, init ``threshold``) through ``ZgVector`` — hard
    non-differentiable dynamics are fine for gradient-free ES, no
    surrogate gradients needed.

    With ``pin_norm=True`` (default) each neuron's effective input norm is
    pinned to its init value (the forward rescales the perturbing linear's
    output by ``s_j / ‖W_j‖`` with ``s_j`` frozen at init).  Without it,
    SNN+ES reliably collapses: uniform weight growth saturates or silences
    all rates and is fitness-neutral.  Classification: pass readout spike
    trains through :func:`spike_count_logits` (typically ``log1p`` of the
    counts as logits).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        leak: float = 0.9,
        threshold: float = 1.0,
        use_bias: bool = False,
        pin_norm: bool = True,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if not 0.0 < leak < 1.0:
            raise ValueError(f"leak must be in (0, 1), got {leak}")
        if threshold <= 0.0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"in/out features must be positive, got {in_features}/{out_features}"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.dtype = dtype
        self.pin_norm = bool(pin_norm)
        # Sigmoid logit of the initial leak.
        beta_init = math.log(leak / (1.0 - leak))
        self.beta = nnx.Param(
            jnp.full((self.out_features,), beta_init, dtype=dtype)
        )
        # Softplus logit of the initial threshold.
        thresh_init = math.log(math.expm1(threshold))
        self.thresh_raw = nnx.Param(
            jnp.full((self.out_features,), thresh_init, dtype=dtype)
        )
        self.lin = nnx.Linear(
            in_features, out_features, use_bias=use_bias,
            param_dtype=dtype, rngs=rngs,
        )
        if self.pin_norm:
            # Frozen init-time norms — plain constants, invisible to ES.
            norms = jnp.linalg.norm(self.lin.kernel[...], axis=0)
            self.w_scale: tuple[float, ...] = tuple(float(v) for v in norms)

    def beta_value(self) -> jax.Array:
        """Effective per-neuron decay in ``(0, 1)`` — ES-perturbation aware."""
        return jax.nn.sigmoid(_param_value(self.beta).astype(jnp.float32))

    def threshold_value(self) -> jax.Array:
        """Effective per-neuron threshold ``> 0`` — ES-perturbation aware."""
        return jax.nn.softplus(_param_value(self.thresh_raw).astype(jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        """``[B, T, Din]`` spikes/currents → ``[B, T, Dout]`` binary 0/1."""
        if x.ndim != 3:
            raise ValueError(f"Spiking expects [B, T, Din], got {x.shape}")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Spiking expects Din={self.in_features}, got {x.shape[-1]}"
            )
        current = self.lin(x).astype(self.dtype)  # perturbing linear
        if self.pin_norm:
            # Read the raw kernel only for norms (never for the forward):
            # rescale the candidate-perturbed output to the frozen norm so
            # ES sees orientation changes only, keeping fitness signal.
            norms = jnp.linalg.norm(self.lin.kernel[...], axis=0) + 1e-8
            scale = (jnp.asarray(self.w_scale, dtype=jnp.float32) / norms)
            current = current * scale.astype(current.dtype)
        return _lif_scan(current, self.beta_value(), self.threshold_value())
