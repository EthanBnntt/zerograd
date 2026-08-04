"""Train an MLP classifier on CIFAR-10 using ZeroGrad evolutionary optimization.

Model: 3072 → 128 → 10, ReLU hidden, softmax cross-entropy loss.

CIFAR-10 is a much harder dataset than MNIST for a small MLP — even gradient-
based training tops out around 50-55% with this architecture.  ZeroGrad's ES
approach will reach a more modest accuracy (~35-45%) because:

  1. Evolutionary strategies are less sample-efficient than backprop.
  2. The low-rank perturbation (rank=8) limits the search space dimensionality.
  3. Each step evaluates 32 candidates, not one gradient.
  4. 3072-dim inputs amplify perturbation noise, requiring a smaller sigma.

The point is not to beat backprop — it's to demonstrate that ZeroGrad can
optimize a real image classifier end-to-end using only fitness evaluation,
which enables training through non-differentiable operations (see train_qat_xor.py).

    uv run python examples/train_cifar10.py [--steps N] [--batch N]

Data is downloaded on first run to ~/.cache/zerograd/ (~170 MB).

This is a thin wrapper around the shared 2-layer MLP training loop in
``_vision_mlp_train.py`` (see also ``train_mnist.py``).
"""

from __future__ import annotations

from _data import load_cifar10
from _vision_mlp_train import VisionMlpSpec, main

SPEC = VisionMlpSpec(
    name="CIFAR-10",
    load_data=load_cifar10,
    input_dim=3072,  # 32×32×3 flattened
    hidden=128,
    default_steps=300,
    default_batch=128,
    lr=5e-3,  # ES pseudo-gradients need a higher LR than typical gradient training
    sigma=0.02,  # lower than MNIST — 3072-dim inputs amplify perturbation noise
    run_id="cifar-demo",
    manifest_prefix="cifar",
)


if __name__ == "__main__":
    main(SPEC)
