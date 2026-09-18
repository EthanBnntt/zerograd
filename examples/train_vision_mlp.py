"""Train a 2-layer MLP on MNIST or CIFAR-10 with ZeroGrad.

Thin CLI over ``_vision_mlp_train.py``:

    uv run python examples/train_vision_mlp.py --dataset mnist
    uv run python examples/train_vision_mlp.py --dataset cifar10 --steps 300
"""

from __future__ import annotations

import argparse

from _data import load_cifar10, load_mnist
from _vision_mlp_train import VisionMlpSpec, build_arg_parser, run

SPECS = {
    "mnist": VisionMlpSpec(
        name="MNIST",
        load_data=load_mnist,
        input_dim=784,
        hidden=64,
        default_steps=200,
        default_batch=128,
        lr=1e-2,
        sigma=0.1,
        run_id="mnist-demo",
    ),
    "cifar10": VisionMlpSpec(
        name="CIFAR-10",
        load_data=load_cifar10,
        input_dim=3072,
        hidden=128,
        default_steps=300,
        default_batch=128,
        lr=5e-3,
        sigma=0.02,
        run_id="cifar-demo",
    ),
}


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--dataset",
        choices=sorted(SPECS),
        default="mnist",
        help="Vision dataset (default: mnist).",
    )
    pre_args, remaining = pre.parse_known_args()
    spec = SPECS[pre_args.dataset]
    parser = build_arg_parser(spec)
    parser.add_argument(
        "--dataset",
        choices=sorted(SPECS),
        default=pre_args.dataset,
        help="Vision dataset (default: mnist).",
    )
    args = parser.parse_args(remaining)
    args.dataset = pre_args.dataset
    run(spec, args)


if __name__ == "__main__":
    main()
