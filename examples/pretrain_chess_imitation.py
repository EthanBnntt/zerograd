"""Stage 1 — gradient-free imitation pretraining on expert chess (Lichess Elite).

"Pretrain on expert level chess" without gradients: behavior cloning driven
purely by evolution strategies.  Each generation, every ZeroGrad candidate
scores the SAME batch of expert positions (common random numbers); fitness is
the mean log-probability the policy assigns to the expert's move (plus a
value-head fit to the game outcome).  Dense signal, no game noise, no
backprop — the policy head learns expert move preferences and the value head
learns position evaluation, giving stage-2 self-play a strong starting point.

Data: Lichess Elite Database (high-rated standard games, PGN.zst).  Positions
are converted FEN -> pgx state -> exact 8x8x119 observation planes; the expert
move is mapped to its pgx action label via the legal action mask.

    # on the training host
    python examples/pretrain_chess_imitation.py --prepare   # download + convert
    python examples/pretrain_chess_imitation.py --wandb-project zerograd-rl
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import sys

sys.path.insert(0, str(Path(__file__).parent))
from train_es_chess import PolicyValueNet  # noqa: E402

from zerograd import ZeroGrad  # noqa: E402

DATA_PATH = Path("data/lichess_elite_positions.npz")
PGN_URL = "https://database.nikonoel.fr/lichess_elite_2025-07.zip"
PGN_PATH = Path("data/lichess_elite_2025-07.zip")


# ── Fast vectorized position conversion (numpy + vmapped pgx fns) ────────────
# pgx chess squares are column-major (sq = file*8 + rank), white pieces
# +1..+6 (P,N,B,R,Q,K), board stored from the side-to-move's perspective
# (rank-flip + color negation for Black).  We build GameState fields in numpy
# and run pgx's OWN observe / legal_action_mask under vmap, so the training
# planes are bit-identical to what self-play produces.
import pgx._src.games.chess as gc  # noqa: E402

ZB = np.array(gc.ZOBRIST_BOARD)  # (64, 13, 2) uint32
ZS = np.array(gc.ZOBRIST_SIDE)  # (2,)
ZC = np.array(gc.ZOBRIST_CASTLING)  # (4, 2)
ZE = np.array(gc.ZOBRIST_EN_PASSANT)  # (65, 2); index -1 wraps to "none" row
SQ64 = np.arange(64)


def _persp(abs_board: np.ndarray, color: int) -> np.ndarray:
    """Absolute column-major board -> side-to-move perspective."""
    if color == 0:
        return abs_board
    return -abs_board.reshape(8, 8)[:, ::-1].reshape(64)


def game_ply_arrays(game_moves) -> dict:
    """Per-ply absolute (white-perspective) arrays for a game."""
    import chess

    board = chess.Board()
    L = len(game_moves)
    boards = np.zeros((L + 1, 64), dtype=np.int8)
    colors = np.zeros(L + 1, dtype=np.int8)
    casts = np.zeros((L + 1, 4), dtype=bool)  # absolute [WQ, WK, BQ, BK]
    eps = np.full(L + 1, -1, dtype=np.int8)
    half = np.zeros(L + 1, dtype=np.int16)

    def record(idx):
        for sq, piece in board.piece_map().items():
            code = piece.piece_type if piece.color == chess.WHITE else -piece.piece_type
            f, r = chess.square_file(sq), chess.square_rank(sq)
            boards[idx, f * 8 + r] = code
        colors[idx] = 0 if board.turn == chess.WHITE else 1
        casts[idx] = [
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.BLACK),
            board.has_kingside_castling_rights(chess.BLACK),
        ]
        if board.ep_square is not None:
            f, r = chess.square_file(board.ep_square), chess.square_rank(board.ep_square)
            eps[idx] = f * 8 + r
        half[idx] = board.halfmove_clock

    record(0)  # initial position: index p == after p moves
    for ply, mv in enumerate(game_moves):
        board.push(mv)
        record(ply + 1)
    return {"boards": boards, "colors": colors, "casts": casts, "eps": eps, "half": half}


def ply_hashes(arr: dict) -> np.ndarray:
    """Zobrist hashes for every ply, each in its own mover's perspective."""
    boards, colors, casts, eps = arr["boards"], arr["colors"], arr["casts"], arr["eps"]
    L = len(colors)
    flipped = -boards.reshape(L, 8, 8)[:, :, ::-1].reshape(L, 64)
    bp = np.where(colors[:, None] == 0, boards, flipped).astype(np.int64)
    h = np.broadcast_to(ZS, (L, 2)).copy()
    h[colors == 1] = 0
    h ^= np.bitwise_xor.reduce(ZB[SQ64[None, :], bp + 6], axis=1)
    # castling, mover-relative order [my_q, my_k, opp_q, opp_k]
    rel = np.where(colors[:, None] == 0, casts, casts[:, [2, 3, 0, 1]])
    h ^= np.bitwise_xor.reduce(np.where(rel[:, :, None], ZC[None], 0), axis=1)
    ep_persp = np.where(colors == 0, eps, (eps // 8) * 8 + (7 - eps % 8))
    h ^= ZE[ep_persp]  # eps==-1 -> last row (wraps like jax indexing)
    return h.astype(np.uint32)


def build_gamestates(arr: dict, hashes: np.ndarray, plies: list[int]) -> dict:
    """Stack GameState field arrays for the given plies (numpy, batched)."""
    boards, colors, casts, eps, half = (
        arr["boards"], arr["colors"], arr["casts"], arr["eps"], arr["half"],
    )
    out = {k: [] for k in (
        "color", "board", "castling_rights", "en_passant", "halfmove_count",
        "fullmove_count", "hash_history", "board_history", "step_count")}
    for p in plies:
        c = int(colors[p])
        out["color"].append(c)
        out["board"].append(_persp(boards[p], c))
        rel = casts[p] if c == 0 else casts[p][[2, 3, 0, 1]]
        out["castling_rights"].append(rel.reshape(2, 2))
        ep = int(eps[p])
        if c == 1 and ep >= 0:
            ep = (ep // 8) * 8 + (7 - ep % 8)
        out["en_passant"].append(ep)
        out["halfmove_count"].append(int(half[p]))
        out["fullmove_count"].append(p // 2 + 1)
        out["step_count"].append(p)
        bh = np.zeros((8, 64), dtype=np.int32)
        for i in range(0, min(8, p + 1)):
            bh[i] = _persp(boards[p - i], c)  # all frames in current perspective
        out["board_history"].append(bh)
        hh = np.zeros((gc.MAX_TERMINATION_STEPS + 1, 2), dtype=np.uint32)
        lo = max(0, p - gc.MAX_TERMINATION_STEPS)
        hh[: p - lo + 1] = hashes[lo : p + 1][::-1]
        out["hash_history"].append(hh)
    return {k: np.array(v) for k, v in out.items()}


_GAME = gc.Game()


@jax.jit
def _observe_and_mask(states: gc.GameState):
    # NOTE: the Game *method* legal_action_mask silently returns an empty
    # mask under vmap; the module-level function vmaps correctly.
    return jax.vmap(_GAME.observe)(states), jax.vmap(gc._legal_action_mask)(states)


def states_from_fields(fields: dict) -> gc.GameState:
    return gc.GameState(
        color=jnp.array(fields["color"], dtype=jnp.int32),
        board=jnp.array(fields["board"], dtype=jnp.int32),
        castling_rights=jnp.array(fields["castling_rights"], dtype=jnp.bool_),
        en_passant=jnp.array(fields["en_passant"], dtype=jnp.int32),
        halfmove_count=jnp.array(fields["halfmove_count"], dtype=jnp.int32),
        fullmove_count=jnp.array(fields["fullmove_count"], dtype=jnp.int32),
        hash_history=jnp.array(fields["hash_history"], dtype=jnp.uint32),
        board_history=jnp.array(fields["board_history"], dtype=jnp.int32),
        legal_action_mask=jnp.zeros((len(fields["color"]), 4672), dtype=jnp.bool_),
        step_count=jnp.array(fields["step_count"], dtype=jnp.int32),
    )


# ── Data preparation ─────────────────────────────────────────────────────────
def prepare(args) -> None:
    import io
    import urllib.request
    import zipfile

    import chess.pgn

    from eval_elo_chess import move_to_label

    PGN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PGN_PATH.exists():
        print(f"downloading {PGN_URL} ...")
        urllib.request.urlretrieve(PGN_URL, PGN_PATH)
    print(f"parsing {PGN_PATH} ({PGN_PATH.stat().st_size/1e6:.0f} MB)")

    pend_fields, pend_moves, pend_white, pend_z = [], [], [], []
    obs_list, label_list, z_list = [], [], []
    rng = np.random.default_rng(args.seed)
    n_games = 0

    def flush():
        if not pend_fields:
            return
        fields = {k: np.concatenate([f[k] for f in pend_fields]) for k in pend_fields[0]}
        states = states_from_fields(fields)
        obs, masks = _observe_and_mask(states)
        obs_np, masks_np = np.array(obs), np.array(masks)
        for i, (mv, white, z) in enumerate(zip(pend_moves, pend_white, pend_z)):
            try:
                label = move_to_label(mv, white, masks_np[i])
            except ValueError:
                continue
            obs_list.append(obs_np[i].astype(np.float16))
            label_list.append(label)
            z_list.append(z)
        pend_fields.clear(); pend_moves.clear(); pend_white.clear(); pend_z.clear()

    zf = zipfile.ZipFile(PGN_PATH)
    member = next(n for n in zf.namelist() if n.endswith(".pgn"))
    with zf.open(member) as f:
        reader = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
        while n_games < args.max_games and len(obs_list) < args.max_positions:
            game = chess.pgn.read_game(reader)
            if game is None:
                break
            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                continue
            moves = list(game.mainline_moves())
            if len(moves) < 20 or len(moves) > gc.MAX_TERMINATION_STEPS:
                continue
            n_games += 1
            # boards[p] is the position after p moves; imitate moves[p].
            plies = rng.choice(
                np.arange(4, len(moves)),
                size=min(args.positions_per_game, len(moves) - 4),
                replace=False,
            )
            arr = game_ply_arrays(moves)
            hashes = ply_hashes(arr)
            fields = build_gamestates(arr, hashes, [int(p) for p in plies])
            pend_fields.append(fields)
            for p in plies:
                p = int(p)
                pend_moves.append(moves[p])
                pend_white.append(bool(arr["colors"][p] == 0))
                pend_z.append(
                    0.0 if result == "1/2-1/2"
                    else (1.0 if (result == "1-0") == (arr["colors"][p] == 0) else -1.0)
                )
            if sum(len(f["color"]) for f in pend_fields) >= 4096:
                flush()
            if n_games % 1000 == 0:
                print(f"  {n_games} games -> {len(obs_list)} positions", flush=True)
    flush()

    n = len(label_list)
    obs = np.stack(obs_list)
    labels = np.array(label_list, dtype=np.int32)
    zs = np.array(z_list, dtype=np.float32)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_PATH, obs=obs, labels=labels, zs=zs)
    print(f"saved {n} positions -> {DATA_PATH} ({obs.nbytes/1e9:.2f} GB f16)")


# ── ES imitation training ────────────────────────────────────────────────────
def make_loss():
    def imitation_loss(model: PolicyValueNet, batch, rng: jax.Array):
        del rng  # all candidates score the same batch (common random numbers)
        obs, labels, zs = batch
        logits, value = model(obs)
        logp = jnp.take_along_axis(
            jax.nn.log_softmax(logits), labels[:, None], axis=1
        ).squeeze(-1)
        value_mse = jnp.mean((value - zs) ** 2)
        return -(jnp.mean(logp)) + 0.5 * value_mse, None

    return imitation_loss


def make_eval(num_positions: int):
    @nnx.jit
    def evaluate(model: PolicyValueNet, obs, labels, zs):
        logits, value = model(obs)
        top1 = jnp.mean(jnp.argmax(logits, -1) == labels)
        top5 = jnp.mean(
            jnp.any(
                jnp.argsort(logits, -1)[:, -5:] == labels[:, None], axis=-1
            )
        )
        sign_acc = jnp.mean((value > 0).astype(jnp.float32) == (zs > 0))
        return top1, top5, sign_acc

    return evaluate


def train(args) -> None:
    data = np.load(DATA_PATH)
    obs_all, labels_all, zs_all = data["obs"], data["labels"], data["zs"]
    n = len(labels_all)
    n_val = min(4096, n // 10)
    val_ix = np.arange(n - n_val, n)
    print(f"positions={n}  val={n_val}")

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                tags=["zerograd", "es", "imitation", "stage1", "chess"],
            )
        except Exception as exc:
            print(f"[wandb disabled: {exc}]")

    model = PolicyValueNet((8, 8, 119), 4672, args.channels, args.width, nnx.Rngs(args.seed))
    optimizer = ZeroGrad(
        optax.adamw(learning_rate=args.lr, weight_decay=0.0),
        population_size=args.population,
        rank=args.rank,
        sigma=args.sigma,
        seed=args.seed,
        run_id="es-imitation-chess",
        candidate_chunk_size=args.candidate_chunk,
    )
    opt_state = optimizer.init(model)
    loss_fn = make_loss()
    evaluate = make_eval(n_val)
    val_batch = (
        jnp.array(obs_all[val_ix].astype(np.float32)),
        jnp.array(labels_all[val_ix]),
        jnp.array(zs_all[val_ix]),
    )

    rng = np.random.default_rng(args.seed + 1)
    best_top1 = 0.0
    t0 = time.time()
    for gen in range(args.generations):
        ix = rng.integers(0, n - n_val, size=args.batch)
        batch = (
            jnp.array(obs_all[ix].astype(np.float32)),
            jnp.array(labels_all[ix]),
            jnp.array(zs_all[ix]),
        )
        model, opt_state, metrics = optimizer.step(opt_state, model, batch, loss_fn)
        log = {
            "gen": metrics.generation,
            "train/neg_logp": float(metrics.mean_loss),
        }
        if gen % args.eval_every == 0 or gen == args.generations - 1:
            top1, top5, sign_acc = evaluate(model, *val_batch)
            log["val/move_top1"] = float(top1)
            log["val/move_top5"] = float(top5)
            log["val/value_sign_acc"] = float(sign_acc)
            print(
                f"gen {gen:4d}  loss={float(metrics.mean_loss):.4f}  "
                f"top1={float(top1):.3f}  top5={float(top5):.3f}  "
                f"vsign={float(sign_acc):.3f}  ({time.time()-t0:.0f}s)",
                flush=True,
            )
            if float(top1) > best_top1:
                best_top1 = float(top1)
                out = Path(args.out)
                out.parent.mkdir(parents=True, exist_ok=True)
                _, mstate = nnx.split(model)
                with out.open("wb") as f:
                    pickle.dump(
                        {
                            "config": {**vars(args), "env": "chess"},
                            "best_top1": best_top1,
                            "flat_state": jax.device_get(
                                [
                                    (tuple(map(str, p)), v)
                                    for p, v in nnx.to_flat_state(mstate)
                                ]
                            ),
                        },
                        f,
                    )
        if wandb_run is not None:
            wandb_run.log(log, step=gen)

    print(f"best val top1 = {best_top1:.3f}; checkpoint -> {args.out}")
    if wandb_run is not None:
        wandb_run.summary["best_val_top1"] = best_top1
        wandb_run.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prepare", action="store_true", help="download + convert data")
    p.add_argument("--max-games", type=int, default=20000)
    p.add_argument("--positions-per-game", type=int, default=6)
    p.add_argument("--max-positions", type=int, default=120000)
    p.add_argument("--generations", type=int, default=1500)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--candidate-chunk", type=int, default=None,
                   help="chunk candidates through lax.map to cap peak VRAM")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--wandb-project", default="zerograd-rl")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--out", default="results/es_chess_pretrained.pkl")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.prepare:
        prepare(args)
    else:
        train(args)
