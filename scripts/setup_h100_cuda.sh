#!/usr/bin/env bash
# Bootstrap CUDA JAX env on a Lambda H100 (Ubuntu). Repo root = parent of scripts/.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

# Fresh venv — do not use the ROCm jax extra from pyproject on NVIDIA.
uv venv --python 3.12 .venv
# Editable install without resolving the rocm jax extra.
uv pip install -e . --no-deps
uv pip install \
  "jax[cuda12]>=0.10.2" \
  "flax>=0.12.8" \
  "optax>=0.2.8" \
  "numpy" \
  "datasets" \
  "transformers" \
  "huggingface_hub" \
  "wandb"

.venv/bin/python - <<'PY'
import jax
print("jax", jax.__version__)
print("devices", jax.devices())
print("backend", jax.default_backend())
assert jax.default_backend() == "gpu", jax.devices()
x = jax.numpy.ones((4096, 4096))
y = (x @ x).block_until_ready()
print("matmul ok", float(y[0, 0]))
PY
echo "H100 CUDA env ready."
