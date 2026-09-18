"""Measure Elo of a ZeroGrad ES+MCTS chess champion against Stockfish.

Plays the saved champion (from ``train_es_chess.py``) against Stockfish at a
ladder of skill levels using the UCI protocol (via python-chess), then fits a
maximum-likelihood logistic Elo over all games.

ANCHORING ASSUMPTION (approximate, stated honestly): Stockfish "Skill Level"
ratings are not official; we use the commonly cited community approximation
{0: 1350, 1: 1550, 2: 1700, 3: 1800, 5: 2000, 8: 2150, 12: 2400}.  Treat the
absolute Elo as ±100; the per-level win rates are the ground truth.

The pgx state is kept in lockstep with the python-chess board.  pgx chess
squares are column-major (square = file*8 + rank) from the CURRENT player's
perspective, with ranks mirrored for Black; conversions below handle both
directions by decoding against the legal action mask.

    python examples/eval_elo_chess.py --checkpoint results/es_chess_champion.pkl \
        --stockfish stockfish --sims 32 --games-per-level 20
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import chess
import chess.engine
import jax
import jax.numpy as jnp
import numpy as np
import pgx
import pgx._src.games.chess as gc
from flax import nnx

sys.path.insert(0, str(Path(__file__).parent))
from train_es_chess import PolicyValueNet, make_search  # noqa: E402

from zerograd._nnx import optional_model_zg_slot  # noqa: E402

# Approximate community anchor ratings for Stockfish skill levels (see docstring)
ANCHORS = {0: 1350, 1: 1550, 2: 1700, 3: 1800, 5: 2000, 8: 2150, 12: 2400}

FROM_PLANE = np.array(gc.FROM_PLANE)  # [64, 73] -> to-square
UNDERPROMO_PIECE = {0: chess.ROOK, 1: chess.BISHOP, 2: chess.KNIGHT}


def decode_label(label: int) -> tuple[int, int, int]:
    """pgx label -> (from_sq, to_sq, underpromotion) in current-player coords."""
    from_, plane = label // 73, label % 73
    under = plane // 3 if plane < 9 else -1
    return from_, int(FROM_PLANE[from_, plane]), under


def pgx_sq_to_abs(sq: int, white_to_move: bool) -> int:
    """pgx square (file*8+rank, current-player view) -> python-chess square."""
    f, r = sq // 8, sq % 8
    if not white_to_move:
        r = 7 - r
    return chess.square(f, r)


def abs_sq_to_pgx(sq: int, white_to_move: bool) -> int:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    if not white_to_move:
        r = 7 - r
    return f * 8 + r


def label_to_move(label: int, white_to_move: bool, board: chess.Board) -> chess.Move:
    from_, to, under = decode_label(label)
    frm = pgx_sq_to_abs(from_, white_to_move)
    dst = pgx_sq_to_abs(to, white_to_move)
    promotion = UNDERPROMO_PIECE.get(under) if under >= 0 else None
    piece = board.piece_at(frm)
    if (
        promotion is None
        and piece is not None
        and piece.piece_type == chess.PAWN
        and chess.square_rank(dst) in (0, 7)
    ):
        promotion = chess.QUEEN  # normal-plane pawn move to last rank
    return chess.Move(frm, dst, promotion)


def move_to_label(move: chess.Move, white_to_move: bool, legal_mask: np.ndarray) -> int:
    """python-chess move -> pgx label, resolved against the legal mask."""
    frm = abs_sq_to_pgx(move.from_square, white_to_move)
    dst = abs_sq_to_pgx(move.to_square, white_to_move)
    under = -1
    if move.promotion in UNDERPROMO_PIECE.values():
        under = {v: k for k, v in UNDERPROMO_PIECE.items()}[move.promotion]
    for label in np.nonzero(legal_mask)[0]:
        f, t, u = decode_label(int(label))
        if f == frm and t == dst and u == under:
            return int(label)
    raise ValueError(f"no legal pgx label for {move.uci()}")


def load_champion(path: str):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    cfg = payload["config"]
    env = pgx.make(cfg["env"])
    obs_shape = env.init(jax.random.key(0)).observation.shape
    model = PolicyValueNet(
        obs_shape, env.num_actions, cfg["channels"], cfg["width"], nnx.Rngs(0)
    )
    # Rebuild the ZeroGrad-surgered structure, then substitute saved values.
    from zerograd import ZeroGrad
    import optax

    opt = ZeroGrad(
        optax.adamw(1e-3), population_size=2, rank=1, sigma=0.1, seed=0, run_id="eval"
    )
    opt.init(model)
    graphdef, state = nnx.split(model)
    saved = {p: v for p, v in payload["flat_state"]}
    flat = [
        (path, saved.get(tuple(map(str, path)), var))
        for path, var in nnx.to_flat_state(state)
    ]
    model = nnx.merge(graphdef, nnx.from_flat_state(flat))
    slot = optional_model_zg_slot(model)
    if slot is not None:
        slot.enabled = False
    return env, model, cfg


def play_game(env, run_search, engine, key, our_pid=0) -> float:
    """One game vs Stockfish. Returns score for us: 1 / 0.5 / 0."""
    state = env.init(key)
    order = np.array(state._player_order)
    our_color = chess.WHITE if int(order[our_pid]) == 0 else chess.BLACK
    board = chess.Board()
    while not board.is_game_over(claim_draw=False) and not bool(state.terminated):
        white_to_move = board.turn == chess.WHITE
        if board.turn == our_color:
            batched = jax.tree.map(lambda x: x[None], state)
            out = run_search(batched, key)
            action = int(out.action[0])
            move = label_to_move(action, white_to_move, board)
            assert move in board.legal_moves, (
                f"pgx label {action} -> {move.uci()} not legal; converter bug"
            )
            board.push(move)
            state = env.step(state, jnp.int32(action))
        else:
            result = engine.play(board, chess.engine.Limit(time=0.02))
            board.push(result.move)
            label = move_to_label(result.move, white_to_move, np.array(state.legal_action_mask))
            state = env.step(state, jnp.int32(label))
        key = jax.random.fold_in(key, int(state._step_count))
    # Ground truth from pgx rewards for our pid
    r = float(state.rewards[our_pid])
    if r == 0.0 and board.is_game_over():
        res = board.result()
        r = {"1-0": 1.0, "0-1": -1.0}.get(res, 0.0) if our_color == chess.WHITE else {
            "1-0": -1.0, "0-1": 1.0
        }.get(res, 0.0)
        if "*" in res:
            r = 0.0
    return (r + 1.0) / 2.0


def fit_elo(results: dict[int, list[float]]) -> float:
    """MLE logistic Elo over anchored skill levels (grid search)."""
    best_e, best_ll = 1000.0, -np.inf
    for e in np.arange(800, 2600, 1.0):
        ll = 0.0
        for level, scores in results.items():
            if not scores:
                continue
            p = 1.0 / (1.0 + 10 ** (-(e - ANCHORS[level]) / 400.0))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            ll += sum(np.log(p) * s + np.log(1 - p) * (1 - s) for s in scores)
        if ll > best_ll:
            best_ll, best_e = ll, e
    return best_e


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stockfish", default="stockfish")
    p.add_argument("--sims", type=int, default=32)
    p.add_argument("--games-per-level", type=int, default=20)
    p.add_argument("--levels", default="0,1,2,5,8")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    env, model, cfg = load_champion(args.checkpoint)
    graphdef, mstate = nnx.split(model)
    search = make_search(env, args.sims, 16, gumbel_scale=0.0)

    @jax.jit
    def run_search(mstate, state, key):
        m = nnx.merge(graphdef, mstate)
        return search(lambda obs: m(obs), state, key)

    from functools import partial

    run_search = partial(run_search, mstate)
    levels = [int(x) for x in args.levels.split(",")]

    results: dict[int, list[float]] = {}
    for level in levels:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        engine.configure({"Skill Level": level})
        scores = []
        for g in range(args.games_per_level):
            key = jax.random.key(args.seed + level * 1000 + g)
            s = play_game(env, run_search, engine, key)
            scores.append(s)
            print(f"level {level} game {g}: score {s}", flush=True)
        engine.quit()
        results[level] = scores
        wr = np.mean(scores)
        print(f"== skill {level} (anchor {ANCHORS[level]}): score {wr:.3f} over {len(scores)}")

    elo = fit_elo(results)
    print(f"\nEstimated Elo: {elo:.0f}  (anchors approximate, see docstring)")
    out = Path(args.checkpoint).with_suffix(".elo.txt")
    out.write_text(
        f"elo={elo:.0f}\n"
        + "\n".join(
            f"level {l} (anchor {ANCHORS[l]}): {np.mean(s):.3f} ({len(s)} games)"
            for l, s in results.items()
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
