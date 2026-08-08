"""Cluster ZeroGrad on XOR — seed / multiprocess / unreliable modes.

    uv run python examples/train_cluster_xor.py --mode seed
    uv run python examples/train_cluster_xor.py --mode multiprocess --nodes 4
    uv run python examples/train_cluster_xor.py --mode unreliable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``from _xor_model import ...`` and ``from _cluster...`` when run as a script.
_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Multiprocess workers are spawned with ``--worker`` and must not require --mode.
    if "--worker" in argv:
        from _cluster import multiprocess as mode

        sys.argv = [sys.argv[0], *argv]
        mode.main()
        return

    parser = argparse.ArgumentParser(description="Cluster ZeroGrad on XOR")
    parser.add_argument(
        "--mode",
        choices=("seed", "multiprocess", "unreliable"),
        default="seed",
        help="Cluster demo mode (default: seed).",
    )
    args, remaining = parser.parse_known_args(argv)

    if args.mode == "seed":
        from _cluster import seed as mode
    elif args.mode == "multiprocess":
        from _cluster import multiprocess as mode
    else:
        from _cluster import unreliable as mode

    sys.argv = [sys.argv[0], *remaining]
    mode.main()


if __name__ == "__main__":
    main()
