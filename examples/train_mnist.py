"""Train an MLP classifier on MNIST using ZeroGrad evolutionary optimization.

Model: 784 → 64 → 10, ReLU hidden, softmax cross-entropy loss.

ZeroGrad evaluates 32 perturbed candidates per step (no backprop), so each
step costs ~32× a forward pass.  On CPU, expect ~2-4 seconds per step.
Expected accuracy after 200 steps: ~85-92%.  This won't match gradient-based
training (98%+) — ES trades sample efficiency for the ability to optimize
through non-differentiable operations and arbitrary objectives.

    uv run python examples/train_mnist.py [--steps N] [--batch N]

Data is downloaded on first run to ~/.cache/zerograd/ (~12 MB).

This is a thin wrapper around the shared 2-layer MLP training loop in
``_vision_mlp_train.py`` (see also ``train_cifar10.py``).
"""

from __future__ import annotations

from _data import load_mnist
from _vision_mlp_train import VisionMlpSpec, main

SPEC = VisionMlpSpec(
    name="MNIST",
    load_data=load_mnist,
    input_dim=784,
    hidden=64,
    default_steps=200,
    default_batch=128,
    lr=1e-2,  # ES pseudo-gradients need a higher LR than typical gradient training
    sigma=0.1,
    run_id="mnist-demo",
)


if __name__ == "__main__":
    main(SPEC)
