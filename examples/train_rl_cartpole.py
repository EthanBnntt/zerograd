"""Train a recurrent RL policy with ZeroGrad — no gradients, no BPTT.

A GRU policy with a linear action head learns to balance a *partially
observable* CartPole: the agent observes only cart position and pole angle
(never the velocities), so it must integrate observations through its
recurrent state to act well.  Training is pure evolution strategies — each
generation perturbs the population with ZeroGrad factor noise, scores every
candidate by mean episode return, and shapes a descent direction from the
fitness ranking.  ``jax.grad`` never appears; there is no backprop through
time because there is no backprop at all.

Everything is JAX-traceable: the environment step, the GRU, and the episode
rollout (``lax.scan``) compile into the single vmapped population program
that ``ZeroGrad.step`` dispatches, so one generation is one GPU program.

Examples
--------
Smoke test (CPU, no wandb)::

    uv run python examples/train_rl_cartpole.py --generations 5 --population 16 --no-wandb

Full run with Weights & Biases logging::

    uv run python examples/train_rl_cartpole.py --generations 400 --wandb-project zerograd-rl
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from zerograd import ZeroGrad
from zerograd._nnx import disable_candidates

# ── Environment: partially-observable CartPole ───────────────────────────────
# Classic CartPole dynamics (gym parameters), but the observation is only
# (x, theta) — the velocities x_dot and theta_dot are hidden, so a feedforward
# policy cannot solve it and the recurrent state has to do real work.
GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
LENGTH = 0.5  # half-pole length
POLEMASS_LENGTH = MASS_POLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02
X_THRESHOLD = 2.4
THETA_THRESHOLD = 12.0 * jnp.pi / 180.0  # 12 degrees
OBS_DIM = 2
N_ACTIONS = 2


def _cartpole_step(state: jax.Array, action: jax.Array) -> jax.Array:
    """One Euler integration step of CartPole. ``state`` = (x, x_dot, th, th_dot)."""
    x, x_dot, theta, theta_dot = state
    force = jnp.where(action == 1, FORCE_MAG, -FORCE_MAG)
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sintheta) / TOTAL_MASS
    theta_acc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASS_POLE * costheta**2 / TOTAL_MASS)
    )
    x_acc = temp - POLEMASS_LENGTH * theta_acc * costheta / TOTAL_MASS
    return jnp.array(
        [
            x + TAU * x_dot,
            x_dot + TAU * x_acc,
            theta + TAU * theta_dot,
            theta_dot + TAU * theta_acc,
        ]
    )


def _observe(state: jax.Array) -> jax.Array:
    """Partial observation: normalized (x, theta) — velocities stay hidden."""
    return jnp.array([state[0] / X_THRESHOLD, state[2] / THETA_THRESHOLD])


# ── Policy: GRU + linear action head ─────────────────────────────────────────
class GRUCell(nnx.Module):
    """Minimal GRU built from plain ``nnx.Linear`` layers.

    Building the cell from ``nnx.Linear`` (rather than a fused kernel) means
    ``ZeroGrad.init`` graph surgery wraps every gate in a factor-aware
    ``ZgLinear``, so the whole recurrent policy is ES-perturbable.
    """

    def __init__(self, in_dim: int, hidden: int, rngs: nnx.Rngs):
        self.xz = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.hz = nnx.Linear(hidden, hidden, rngs=rngs)
        self.xr = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.hr = nnx.Linear(hidden, hidden, rngs=rngs)
        self.xn = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.hn = nnx.Linear(hidden, hidden, rngs=rngs)

    def __call__(self, x: jax.Array, h: jax.Array) -> jax.Array:
        z = jax.nn.sigmoid(self.xz(x) + self.hz(h))
        r = jax.nn.sigmoid(self.xr(x) + self.hr(h))
        n = jnp.tanh(self.xn(x) + self.hn(r * h))
        return (1.0 - z) * n + z * h


class RecurrentPolicy(nnx.Module):
    """GRU policy: observation -> hidden state -> action logits."""

    def __init__(self, obs_dim: int, hidden: int, n_actions: int, rngs: nnx.Rngs):
        self.hidden_dim = hidden
        self.gru = GRUCell(obs_dim, hidden, rngs)
        self.head = nnx.Linear(hidden, n_actions, rngs=rngs)

    def __call__(self, obs: jax.Array, h: jax.Array) -> tuple[jax.Array, jax.Array]:
        h_next = self.gru(obs, h)
        return self.head(h_next), h_next


# ── Episode rollout (fully traceable: no Python control flow on values) ──────
def _episode_return(
    model: RecurrentPolicy, key: jax.Array, max_steps: int, hidden: int
) -> jax.Array:
    """Undiscounted return of one episode under a deterministic argmax policy."""
    init_state = jax.random.uniform(key, (4,), minval=-0.05, maxval=0.05)
    h0 = jnp.zeros((hidden,), dtype=jnp.float32)

    def step_fn(carry, _):
        state, h, done, ret = carry
        logits, h_new = model(_observe(state), h)
        action = jnp.argmax(logits)
        state_new = _cartpole_step(state, action)
        terminated = (jnp.abs(state_new[0]) > X_THRESHOLD) | (
            jnp.abs(state_new[2]) > THETA_THRESHOLD
        )
        reward = jnp.where(done, 0.0, 1.0)  # freeze everything after termination
        state_out = jnp.where(done, state, state_new)
        h_out = jnp.where(done, h, h_new)
        return (state_out, h_out, done | terminated, ret + reward), None

    carry, _ = jax.lax.scan(
        step_fn,
        (init_state, h0, jnp.bool_(False), jnp.float32(0.0)),
        None,
        length=max_steps,
    )
    return carry[3]


def make_loss(n_episodes: int, max_steps: int, hidden: int):
    """ZeroGrad loss: negative mean episode return over keyed episodes.

    ZeroGrad folds a per-candidate RNG into ``rng``, so every candidate is
    scored on its own set of random initial states — the population ranking
    is the only signal used for learning.
    """

    def rl_loss(model: RecurrentPolicy, batch, rng: jax.Array):
        keys = jax.random.split(rng, n_episodes)
        returns = jax.vmap(
            lambda k: _episode_return(model, k, max_steps, hidden)
        )(keys)
        mean_return = jnp.mean(returns)
        return -mean_return, mean_return

    return rl_loss


# ── Held-out evaluation of the current mean policy ───────────────────────────
def make_eval(max_steps: int, hidden: int):
    """Build a jitted eval fn; rollout config is closure-static, not traced."""

    @nnx.jit
    def eval_policy(model: RecurrentPolicy, keys: jax.Array):
        returns = jax.vmap(lambda k: _episode_return(model, k, max_steps, hidden))(keys)
        return jnp.mean(returns), jnp.max(returns)

    return eval_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--generations", type=int, default=400)
    p.add_argument("--population", type=int, default=128)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--episodes", type=int, default=4, help="episodes per candidate")
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="zerograd-rl")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--out", default="results/rl_cartpole_policy.pkl")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                tags=["zerograd", "es", "no-bptt", "po-cartpole"],
            )
        except Exception as exc:  # wandb missing or offline — keep training
            print(f"[wandb disabled: {exc}]")
            wandb_run = None

    model = RecurrentPolicy(OBS_DIM, args.hidden, N_ACTIONS, nnx.Rngs(args.seed))
    optimizer = ZeroGrad(
        optax.adamw(learning_rate=args.lr, weight_decay=0.0),
        population_size=args.population,
        rank=args.rank,
        sigma=args.sigma,
        seed=args.seed,
        run_id="rl-po-cartpole",
    )
    state = optimizer.init(model)
    loss_fn = make_loss(args.episodes, args.max_steps, args.hidden)
    eval_policy = make_eval(args.max_steps, args.hidden)
    eval_keys = jax.random.split(jax.random.key(args.seed + 777_777), args.eval_episodes)

    n_params = sum(x.size for x in jax.tree.leaves(nnx.state(model)))
    print(
        f"policy params={n_params}  population={args.population}  "
        f"episodes/candidate={args.episodes}  max_steps={args.max_steps}"
    )

    t0 = time.time()
    for gen in range(args.generations):
        # Fresh episode RNG per generation so fitness never overfits one set
        # of initial states.
        rng = jax.random.key(args.seed * 1_000_003 + gen)
        model, state, metrics = optimizer.step(state, model, None, loss_fn, rng=rng)

        mean_return = -float(metrics.mean_loss)
        best_return = -float(metrics.min_loss)
        log = {
            "gen": metrics.generation,
            "pop/mean_return": mean_return,
            "pop/best_return": best_return,
            "pop/worst_return": -float(metrics.max_loss),
            "pop/std_loss": float(metrics.std_loss),
        }

        if gen % args.eval_every == 0 or gen == args.generations - 1:
            disable_candidates(model)  # score the mean policy, not a candidate
            eval_mean, eval_max = eval_policy(model, eval_keys)
            log["eval/mean_return"] = float(eval_mean)
            log["eval/max_return"] = float(eval_max)
            print(
                f"gen {gen:4d}  pop_mean={mean_return:7.1f}  pop_best={best_return:7.1f}  "
                f"eval_mean={float(eval_mean):7.1f}  eval_max={float(eval_max):7.1f}  "
                f"({time.time() - t0:.0f}s)"
            )
        if wandb_run is not None:
            wandb_run.log(log, step=gen)

    # ── Save final policy ────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _, model_state = nnx.split(model)
    payload = {
        "config": vars(args),
        "flat_state": jax.device_get(
            [(tuple(map(str, path)), var) for path, var in nnx.to_flat_state(model_state)]
        ),
    }
    with out.open("wb") as f:
        pickle.dump(payload, f)
    print(f"saved final policy -> {out}")

    if wandb_run is not None:
        wandb_run.summary["final_eval_mean_return"] = log.get("eval/mean_return")
        wandb_run.summary["final_eval_max_return"] = log.get("eval/max_return")
        wandb_run.finish()


if __name__ == "__main__":
    main()
