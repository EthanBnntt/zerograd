# v1.0.0 readiness checklist

Current state: solid core library (~90% test coverage, 331 tests), good README and examples story, but several gaps between "works locally on a feature branch" and "publishable 1.0."

---

## Release blockers

### 1. Uncommitted / untracked work on the branch

Several things referenced by docs or examples are still untracked:

- `examples/_cluster/` (required by `train_cluster_xor.py`)
- `examples/train_cluster_xor.py`, `train_vision_mlp.py`
- `src/zerograd/_bin_updates.py`, `_fused_lut.py`
- New test files (`test_public_api.py`, `test_integer_es.py`, etc.)

A fresh clone of the current branch would not run cluster examples or pass the full test suite. **Land and stabilize the branch before tagging.**

### 2. No `LICENSE` file

`pyproject.toml` declares MIT, but there is no `LICENSE` in the repo. PyPI and downstream users expect the file. Add a standard MIT `LICENSE` with copyright holder.

### 3. No CI

There is no `.github/workflows/`. For 1.0 you want at minimum:

- `pytest` on CPU (and ideally one CUDA job if feasible)
- `ruff check`
- `basedpyright src/zerograd`
- Coverage gate (`--cov-fail-under=85` is used manually but not in config)

Without CI, regressions will slip in immediately after release.

### 4. Version / maturity mismatch

| Item | Current | For 1.0 |
|------|---------|---------|
| `pyproject.toml` version | `0.2.0` | `1.0.0` |
| Trove classifier | `Development Status :: 3 - Alpha` | `4 - Beta` or `5 - Production/Stable` |
| Changelog | none | `CHANGELOG.md` with 0.x → 1.0 notes |

---

## API surface (stabilize before locking 1.0)

### Public API leaks private modules

Examples import underscored internals:

```python
from zerograd._nnx import params_pure_dict, update_params, disable_candidates
```

Anyone copying examples will depend on private APIs. For 1.0, either:

- **Promote** `params_pure_dict`, `update_params`, `disable_candidates` to `zerograd.__all__`, or
- **Refactor examples** to use only public `ZeroGrad` / `apply_surgery` flows

Same applies to `linear_lut_reference` (used in tests but not exported).

### Naming debt worth fixing *now* (breaking changes get harder after 1.0)

| Current | Issue | Suggestion |
|---------|-------|------------|
| `egg_clip_cast`, `egg_init_matrix`, `egg_requantize`, `float_to_egg_i8` | Paper jargon, unclear to users | `int8_clip_cast`, `int8_init_matrix`, `int8_requantize`, `float_to_int8` (keep old names as aliases for one release if needed) |
| `int_linear_lut_fused` vs `fused_linear_lut` | Asymmetric naming | Pick one pattern |
| `ThresholdTree` | Useful type, not exported | Export from `zerograd` or document as internal |

### `integer_es` mode is powerful but under-documented

The README covers float ES well. Integer ES (antithetical pairs, bin updates, `sigma_shift`, `update_alpha`, `int_bits`, `int_bin_updates`) is only documented in example scripts and scattered Appendix references in code comments. **Add a dedicated README section** before 1.0 — this is a major differentiator.

### Documented limitations

`ZgConv` raises `NotImplementedError` for `CIRCULAR`/`REFLECT`/`CAUSAL` padding. `_replay.py` is the lowest-coverage module (74%). Call these out in docs so users don't discover them in production.

---

## Documentation gaps

**Strong today:** install matrix (CPU/CUDA/ROCm), NNX quick start, distributed/cluster/fault-tolerant overview, examples README.

**Missing for 1.0:**

| Gap | Why it matters |
|-----|----------------|
| `CHANGELOG.md` | Users need to know what 1.0 guarantees |
| Related work / citations | Eggroll algorithms are used; cite in README "Related work" (not in code names) |
| `pyproject.toml` `[project.urls]` | repository, homepage, issues links for PyPI |
| API reference | Even a single `docs/api.md` generated from docstrings helps |
| Migration guide | If renaming `egg_*` → `int8_*`, document it |
| Integer ES guide | See above |
| Contributing / dev setup | Extend README "Development" with lint/typecheck commands |

`examples/vit_findings.md` is great research notes but probably shouldn't be the only place explaining ViT limitations.

---

## Engineering hygiene

### Tooling not wired into project config

- **Ruff** works via `uv tool run ruff` but isn't in `[project.optional-dependencies].dev` or `[tool.ruff]`
- **Pyright** only checks `src/`; tests have ~115 pyright errors (mostly NNX dynamic attrs)
- **Coverage** not enforced in `pyproject.toml` — add to `[tool.pytest.ini_options]`:

```toml
addopts = "--cov=zerograd --cov-fail-under=85"
```

### Duplicate dev dependency definitions

`pyproject.toml` has both `[project.optional-dependencies].dev` and `[dependency-groups].dev` (with only `pillow`). Consolidate — confusing for contributors.

### No example smoke tests in CI

Examples are runnable but never exercised automatically. A lightweight `pytest examples/` or a `scripts/smoke_examples.sh` that runs `train_xor.py` for 2 steps would catch import/path regressions (especially `examples/_cluster/`).

### JAX version pinning

`jax>=0.10.2` is loose. JAX breaks downstream often. For 1.0 consider:

- Upper bound (`jax>=0.10.2,<0.11`) or
- Document tested JAX version in README / lockfile

---

## Test coverage opportunities

Overall **90%** is good. Weak spots:

| Module | Coverage | Risk |
|--------|----------|------|
| `_replay.py` | 74% | Core pseudo-gradient path |
| `_candidate.py` | 79% | Conv/table/integer perturbation branches |
| `_integer.py` | 82% | Quantization edge cases |
| `_nnx.py` | 87% | Large surface, many layer types |

`test_public_api.py` smoke-tests many exports but not all — cluster/distributed types are only covered in dedicated test files (fine), but several `__all__` symbols (`egg_*`, `TernaryLinear`, `IntAffine`, key helpers) have thin or indirect coverage.

---

## Packaging / PyPI checklist

- [ ] Add `LICENSE`
- [ ] Add `[project.urls]` (repository, issues)
- [ ] Bump version to `1.0.0`
- [ ] Update classifier to Stable/Beta
- [ ] Add `CHANGELOG.md`
- [ ] Verify `uv build` produces a wheel that installs cleanly with `pip install zerograd[cpu]`
- [ ] Confirm `README.md` renders on PyPI (no broken relative links)
- [ ] Tag `v1.0.0` only after CI is green on that commit

---

## Suggested priority order

### Before tagging 1.0.0 (do these)

1. Commit/stabilize all untracked code (`_cluster`, `_bin_updates`, tests, etc.)
2. Add `LICENSE`, `CHANGELOG.md`, CI workflow
3. Promote or remove private API usage in examples
4. Document integer ES in README
5. Enforce coverage + lint in pytest/ruff config
6. Bump version and classifier

### Strongly recommended (can be 1.0.0-rc)

7. Rename `egg_*` public API → `int8_*` (with deprecation aliases)
8. Export `params_pure_dict` / `update_params` / `disable_candidates`
9. Example smoke tests in CI
10. Pin or document JAX compatibility

### Post-1.0 / 1.1

11. Sphinx/mkdocs API reference
12. Raise `_replay` / `_candidate` coverage
13. Broader `ZgConv` padding support

---

## Bottom line

The **algorithm and architecture are in good shape** for a first stable release — transactional optimizer, manifest system, distributed/cluster story, and integer ES are all real and tested. What's missing is mostly **release engineering**: legal file, CI, changelog, API boundary cleanup, and committing the in-flight refactor work on this branch.
