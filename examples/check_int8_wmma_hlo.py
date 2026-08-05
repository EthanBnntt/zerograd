"""Verify int8×int8→int32 matmul stays narrow in HLO (RDNA4 WMMA path).

Run on GPU::

    uv run python examples/check_int8_wmma_hlo.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from zerograd import int_matmul


def _hlo_text(fn, *args) -> str:
    lowered = jax.jit(fn).lower(*args)
    # Prefer compiled text when the backend supports it; fall back to HLO module.
    try:
        return lowered.compile().as_text()
    except Exception:
        return lowered.as_text()


def main() -> None:
    print(f"devices={jax.devices()} backend={jax.default_backend()}")
    x = jnp.ones((64, 128), dtype=jnp.int8)
    w = jnp.ones((128, 96), dtype=jnp.int8)

    bad = lambda a, b: a.astype(jnp.int32) @ b.astype(jnp.int32)
    good = lambda a, b: int_matmul(a, b)

    bad_hlo = _hlo_text(bad, x, w)
    good_hlo = _hlo_text(good, x, w)

    def score(label: str, text: str) -> None:
        # StableHLO / HLO markers for element types around dots.
        has_s8 = ("s8" in text) or ("i8" in text) or ("S8" in text)
        upcast_before = ("convert(" in text.lower() and "s32" in text.lower()) or (
            "convert" in text and "i32" in text
        )
        print(f"\n=== {label} ===")
        print(f"  mentions s8/i8: {has_s8}")
        # Show dot-related lines
        lines = [ln for ln in text.splitlines() if "dot" in ln.lower() or "wmma" in ln.lower()]
        for ln in lines[:20]:
            print(" ", ln.strip()[:160])
        if not lines:
            print("  (no 'dot'/'wmma' lines; dumping excerpt)")
            print(text[:1200])

    score("BAD  (astype int32 @)", bad_hlo)
    score("GOOD (int_matmul / preferred_element_type=int32)", good_hlo)

    y = good(x, w)
    assert y.dtype == jnp.int32
    print("\nnumeric ok: int_matmul →", y.dtype, "shape", y.shape)


if __name__ == "__main__":
    main()
