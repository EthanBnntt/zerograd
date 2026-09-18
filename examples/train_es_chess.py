"""Gradient-free MuZero-style chess: ZeroGrad ES + MCTS, no backprop anywhere.

A conv policy-value network learns to play chess (or Gardner minichess) purely
from evolution strategies.  There is no replay buffer, no policy/value loss,
and no gradient: each generation ZeroGrad perturbs the population with
low-rank factor noise, every candidate plays MCTS-guided matches against a
frozen champion, and the game outcomes alone shape the parameter update.
MCTS uses the *true* environment as its dynamics model (AlphaZero-style
planning with a perfect simulator — the MuZero machinery minus the learned
model, which is the part that needs gradients).

Structure per generation (one jit+vmap GPU program per candidate shard):

    candidate params (zerograd factor perturbation)
      -> G parallel games vs frozen champion (vmap = parallel workers)
        -> each ply: two Gumbel-MCTS searches (candidate & champion),
           negamax backup (discount = -1), legal-action masking
      -> fitness = mean score (win 1 / draw .5 / loss 0) + material shaping
    champion replaced when the mean policy beats it above a threshold

Examples
--------
Smoke test (tiny, no wandb)::

    uv run python examples/train_es_chess.py --env gardner_chess \
        --generations 2 --population 4 --games 2 --sims 2 --max-moves 32 --no-wandb

Full run::

    uv run python examples/train_es_chess.py --env chess --generations 300 \
        --population 32 --games 8 --sims 16 --wandb-project zerograd-rl
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
from flax import nnx

from zerograd import ZeroGrad
from zerograd._nnx import disable_candidates, optional_model_zg_slot

# Standard piece values (P, N, B, R, Q, K) for material shaping.  pgx board
# observations are current-player-relative: planes 0-5 = my pieces, 6-11 =
# opponent pieces (order P, N, B, R, Q, K).
PIECE_VALUES = jnp.array([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])


# ── Policy-value network ─────────────────────────────────────────────────────
class PolicyValueNet(nnx.Module):
    """Small conv tower -> shared trunk -> policy logits + tanh value."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        num_actions: int,
        channels: int,
        width: int,
        rngs: nnx.Rngs,
    ):
        h, w, c = obs_shape
        self.c1 = nnx.Conv(c, channels, (3, 3), padding="SAME", rngs=rngs)
        self.c2 = nnx.Conv(channels, channels, (3, 3), padding="SAME", rngs=rngs)
        self.fc = nnx.Linear(h * w * channels, width, rngs=rngs)
        self.policy_head = nnx.Linear(width, num_actions, rngs=rngs)
        self.value_head = nnx.Linear(width, 1, rngs=rngs)

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = nnx.relu(self.c1(obs))
        x = nnx.relu(self.c2(x))
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.fc(x))
        logits = self.policy_head(x)
        value = jnp.tanh(jnp.squeeze(self.value_head(x), -1))
        return logits, value


# ── MCTS with the true env as dynamics (MuZero minus the learned model) ──────
def make_search(env, num_simulations: int, max_considered: int, gumbel_scale: float):
    """Build a batched Gumbel-MCTS action search around a forward callable."""

    def search(forward, state, key):
        logits, value = forward(state.observation)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        def recurrent_fn(params, rng_key, action, embedding):
            del params, rng_key
            moving_player = embedding.current_player
            nxt = jax.vmap(env.step)(embedding, action)
            lg, v = forward(nxt.observation)
            lg = lg - jnp.max(lg, axis=-1, keepdims=True)
            lg = jnp.where(
                nxt.legal_action_mask, lg, jnp.finfo(lg.dtype).min
            )
            batch = nxt.rewards.shape[0]
            reward = nxt.rewards[jnp.arange(batch), moving_player]
            # Negamax two-player backup: value is always from the player-to-
            # move's perspective, so flip sign across the turn boundary and
            # cut the backup at terminal states.
            v = jnp.where(nxt.terminated, 0.0, v)
            discount = jnp.where(nxt.terminated, 0.0, -1.0)
            return mctx.RecurrentFnOutput(
                reward=reward, discount=discount, prior_logits=lg, value=v
            ), nxt

        return mctx.gumbel_muzero_policy(
            params=None,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~state.legal_action_mask,
            max_num_considered_actions=max_considered,
            gumbel_scale=gumbel_scale,
        )

    return search


def _freeze_done(done: jax.Array, old, new):
    """Tree-select old vs new per game: terminated games stop evolving."""

    def select(o, n):
        mask = done.reshape((-1,) + (1,) * (n.ndim - 1))
        return jnp.where(mask, o, n)

    return jax.tree.map(select, old, new)


def _material_score(obs: jax.Array) -> jax.Array:
    """Material balance (mine - theirs) from the current player's planes."""
    my = obs[..., 0:6].sum(axis=(1, 2))  # [B, 6]
    their = obs[..., 6:12].sum(axis=(1, 2))
    return (my - their) @ PIECE_VALUES


# ── Matches: candidate forward vs opponent forward, both colors ──────────────
def make_match(env, search, max_moves: int, material_coef: float):
    """Play a batch of games; return per-game fitness in [-1, 1] for `cand`."""

    def play(cand_forward, opp_forward, state0, cand_pid, key):
        games = state0.current_player.shape[0]

        def ply(carry, k):
            state, done, fitness, plies, mat = carry
            k1, k2 = jax.random.split(k)
            out_c = search(cand_forward, state, k1)
            out_o = search(opp_forward, state, k2)
            is_cand_turn = state.current_player == cand_pid
            action = jnp.where(is_cand_turn, out_c.action, out_o.action)
            nxt = jax.vmap(env.step)(state, action)

            live = ~done
            reward = nxt.rewards[jnp.arange(games), cand_pid] * live.astype(jnp.float32)
            # Track material balance (candidate perspective) at the last live
            # ply; applied once at game end as a small tiebreak bonus so
            # decisive results always dominate.
            mat_now = _material_score(nxt.observation) / 39.0
            mat_sign = jnp.where(
                nxt.current_player == cand_pid, 1.0, -1.0
            )  # obs is next-player relative
            mat = jnp.where(live, mat_now * mat_sign, mat)
            done2 = done | nxt.terminated
            nxt = _freeze_done(done, state, nxt)
            return (
                nxt,
                done2,
                fitness + reward,
                plies + live.astype(jnp.float32),
                mat,
            ), None

        keys = jax.random.split(key, max_moves)
        (_, _, fitness, plies, mat), _ = jax.lax.scan(
            ply,
            (
                state0,
                jnp.zeros(games, bool),
                jnp.zeros(games),
                jnp.zeros(games),
                jnp.zeros(games),
            ),
            keys,
        )
        return fitness + material_coef * mat, plies

    return play


def init_match_states(env, games: int, key):
    keys = jax.random.split(key, games)
    state0 = jax.vmap(env.init)(keys)
    cand_pid = jnp.arange(games, dtype=jnp.int32) % 2  # alternate player ids
    return state0, cand_pid


# ── zerograd loss: candidate (bound model) vs champion (state in batch) ──────
def make_loss(env, play, graphdef, games: int):
    def loss_fn(model: PolicyValueNet, batch, rng: jax.Array):
        # batch = (champion_state, gen_key).  All candidates in a generation
        # share gen_key (common random numbers): same initial player orders,
        # same search-noise stream — so fitness differences reflect the
        # candidate's params, not luck of the draw.  The per-candidate rng is
        # deliberately ignored.
        champion_state, gen_key = batch
        del rng
        champion = nnx.merge(graphdef, champion_state)
        slot = optional_model_zg_slot(champion)
        if slot is not None:
            slot.enabled = False  # champion plays clean (no perturbation)
        k1, k2 = jax.random.split(gen_key)
        state0, cand_pid = init_match_states(env, games, k1)
        fitness, _plies = play(
            lambda obs: model(obs),
            lambda obs: champion(obs),
            state0,
            cand_pid,
            k2,
        )
        mean_fitness = jnp.mean(fitness)
        return -mean_fitness, mean_fitness

    return loss_fn


# ── Held-out evaluation: challenger state vs champion state / baseline ───────
def make_eval(env, play, graphdef, games: int):
    def _merge_clean(state):
        model = nnx.merge(graphdef, state)
        slot = optional_model_zg_slot(model)
        if slot is not None:
            slot.enabled = False
        return model

    @jax.jit
    def vs_champion(challenger_state, champion_state, key):
        challenger = _merge_clean(challenger_state)
        champion = _merge_clean(champion_state)
        k1, k2 = jax.random.split(key)
        state0, cand_pid = init_match_states(env, games, k1)
        return play(
            lambda obs: challenger(obs),
            lambda obs: champion(obs),
            state0,
            cand_pid,
            k2,
        )  # (fitness, plies)

    @jax.jit
    def vs_baseline(challenger_state, key):
        challenger = _merge_clean(challenger_state)
        k1, k2 = jax.random.split(key)
        state0, cand_pid = init_match_states(env, games, k1)

        def baseline_forward(obs):
            logits, _ = challenger(obs)
            return jnp.zeros_like(logits), jnp.zeros(obs.shape[0])

        return play(lambda obs: challenger(obs), baseline_forward, state0, cand_pid, k2)

    return vs_champion, vs_baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env", default="chess", choices=["chess", "gardner_chess"])
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--games", type=int, default=8, help="games per candidate")
    p.add_argument("--sims", type=int, default=16, help="MCTS simulations per move")
    p.add_argument("--max-considered", type=int, default=16)
    p.add_argument("--max-moves", type=int, default=256, help="ply cap per game")
    p.add_argument("--material-coef", type=float, default=0.05)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=32)
    p.add_argument("--champion-threshold", type=float, default=0.58)
    p.add_argument("--wandb-project", default="zerograd-rl")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--init-checkpoint", default=None,
                   help="warm-start params from a stage-1 imitation checkpoint")
    p.add_argument("--out", default="results/es_chess_champion.pkl")
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
                tags=["zerograd", "es", "mcts", "no-gradients", args.env],
            )
        except Exception as exc:
            print(f"[wandb disabled: {exc}]")
            wandb_run = None

    env = pgx.make(args.env)
    obs_shape = env.init(jax.random.key(0)).observation.shape
    num_actions = env.num_actions
    print(f"env={args.env}  obs={obs_shape}  actions={num_actions}")

    model = PolicyValueNet(
        obs_shape, num_actions, args.channels, args.width, nnx.Rngs(args.seed)
    )
    optimizer = ZeroGrad(
        optax.adamw(learning_rate=args.lr, weight_decay=0.0),
        population_size=args.population,
        rank=args.rank,
        sigma=args.sigma,
        seed=args.seed,
        run_id=f"es-mcts-{args.env}",
    )
    opt_state = optimizer.init(model)
    if args.init_checkpoint:
        with open(args.init_checkpoint, "rb") as f:
            payload = pickle.load(f)
        saved = {p: v for p, v in payload["flat_state"]}
        _, fresh = nnx.split(model)
        merged = [
            (path, saved.get(tuple(map(str, path)), var))
            for path, var in nnx.to_flat_state(fresh)
        ]
        model = nnx.merge(nnx.split(model)[0], nnx.from_flat_state(merged))
        print(f"warm-started from {args.init_checkpoint}")
    graphdef, model_state = nnx.split(model)
    champion_state = model_state  # initial champion = initial (random) net

    search = make_search(env, args.sims, args.max_considered, gumbel_scale=1.0)
    play = make_match(env, search, args.max_moves, args.material_coef)
    loss_fn = make_loss(env, play, graphdef, args.games)
    vs_champion, vs_baseline = make_eval(env, play, graphdef, args.eval_games)

    n_params = sum(
        x.size for x in jax.tree.leaves(model_state) if hasattr(x, "size")
    )
    print(
        f"params={n_params}  pop={args.population}  games/candidate={args.games}  "
        f"sims={args.sims}  max_moves={args.max_moves}"
    )

    champion_updates = 0
    t0 = time.time()
    for gen in range(args.generations):
        # One key per generation, shared by all candidates (paired games).
        gen_key = jax.random.key(args.seed * 1_000_003 + gen)
        model, opt_state, metrics = optimizer.step(
            opt_state, model, (champion_state, gen_key), loss_fn, rng=gen_key
        )
        fitness = -float(metrics.mean_loss)
        log = {
            "gen": metrics.generation,
            "pop/mean_fitness": fitness,
            "pop/best_fitness": -float(metrics.min_loss),
            "pop/worst_fitness": -float(metrics.max_loss),
            "champion/updates": champion_updates,
        }

        if gen % args.eval_every == 0 or gen == args.generations - 1:
            disable_candidates(model)
            _, challenger_state = nnx.split(model)
            k1, k2 = jax.random.split(jax.random.key(args.seed + 555_555 + gen))
            f_champ, plies_champ = vs_champion(challenger_state, champion_state, k1)
            f_base, _ = vs_baseline(challenger_state, k2)
            score_champ = float(jnp.mean((f_champ + 1.0) / 2.0))
            score_base = float(jnp.mean((f_base + 1.0) / 2.0))
            log["eval/score_vs_champion"] = score_champ
            log["eval/score_vs_baseline"] = score_base
            log["eval/mean_game_plies"] = float(jnp.mean(plies_champ))
            if score_champ > args.champion_threshold:
                champion_state = challenger_state
                champion_updates += 1
                log["champion/updates"] = champion_updates
                log["champion/just_updated"] = 1
                ckpt = Path(args.out).parent / "champions" / f"champion_gen{gen:04d}.pkl"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                with ckpt.open("wb") as f:
                    pickle.dump(
                        {
                            "config": vars(args),
                            "gen": gen,
                            "flat_state": jax.device_get(
                                [
                                    (tuple(map(str, p)), v)
                                    for p, v in nnx.to_flat_state(champion_state)
                                ]
                            ),
                        },
                        f,
                    )
            print(
                f"gen {gen:4d}  fit={fitness:+.3f}  best={-float(metrics.min_loss):+.3f}  "
                f"vs_champ={score_champ:.3f}  vs_base={score_base:.3f}  "
                f"plies={float(jnp.mean(plies_champ)):.0f}  "
                f"champs={champion_updates}  ({time.time() - t0:.0f}s)"
            )
        if wandb_run is not None:
            wandb_run.log(log, step=gen)

    # ── Save final champion ──────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "champion_updates": champion_updates,
        "flat_state": jax.device_get(
            [(tuple(map(str, p)), v) for p, v in nnx.to_flat_state(champion_state)]
        ),
    }
    with out.open("wb") as f:
        pickle.dump(payload, f)
    print(f"saved final champion -> {out}")

    if wandb_run is not None:
        wandb_run.summary["champion_updates"] = champion_updates
        wandb_run.summary["final_score_vs_baseline"] = log.get("eval/score_vs_baseline")
        wandb_run.finish()


if __name__ == "__main__":
    main()
