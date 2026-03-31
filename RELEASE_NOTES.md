# Znum 4.3.0

## Configurable comparison weighting + Znum.config()

- **`Znum.config()` context manager** — Unified configuration for Z-number computation. Replaces `Znum.fast()` as the primary configuration entry point. Accepts `fast_triangle`, `min_b`, and the new `sort_a_weight` parameter.
- **`sort_a_weight` parameter** — Controls how much the A (restriction/value) component weighs against B (reliability) in `Sort.solver_main` comparisons. Default `0.5` preserves the original equal weighting. Higher values (e.g., `0.7`) make the actual value dominate, preventing B mismatches from masking real value differences in constraint checks.
- **`Znum.fast()` preserved** — Now a convenience alias for `Znum.config(fast_triangle=True, ...)`. Fully backward compatible.

### Motivation

When comparing Z-numbers with mismatched B values (e.g., a summed total with degraded B vs a crisp limit with B=1.0), the original equal weighting allowed B differences to override clear A differences. A value 33 minutes over a time budget could be ranked as "smaller" simply because its B was lower. `sort_a_weight=0.7` fixes this by weighting A at 70%, ensuring that actual value dominance is preserved while B still contributes to the comparison.

### Example

```python
with Znum.config(fast_triangle=True, min_b=True, sort_a_weight=0.7):
    result = z1 + z2        # fast analytical + min_b
    is_over = total > limit  # A weighted 70%, B weighted 30%
```

---

# Znum 4.2.0

## min_b option for Znum.fast()

- **`min_b` parameter on `Znum.fast()`** — When `min_b=True`, arithmetic operations use `min(B1, B2)` element-wise instead of convolution/LP-based B computation. Prevents B reliability degradation through repeated operations (Aliev et al. 2017).

---

# Znum 4.0.0

## Li et al. analytical engine

- **`Znum.fast()` now uses Li et al. 2023 analytical method** — For triangular Z-numbers, B computation uses extended triangular distribution convolutions instead of LP. ~5x faster, deterministic, and produces narrower B spread than LP for high-B inputs. Non-triangular Z-numbers fall back to LP automatically.
- **New module `tri_math.py`** — Implements the full analytical pipeline: extended triangular PDF, convolution, and closed-form B computation.
- **New properties** — `is_triangular`, `A_tri`, `B_tri` for detecting and decomposing triangular Z-numbers.
- **Removed old fast_b mode** — The v3 `B = min(B1, B2)` heuristic is replaced by the analytically grounded Li et al. method. `test_fast_b.py` deleted, replaced by `test_tri_analytical.py` (68 tests).

### Reference

Li, Y. et al. (2023). The arithmetic of triangular Z-numbers with reduced calculation complexity using an extension of triangular distribution. *Information Sciences*, 647, 119477. [doi:10.1016/j.ins.2023.119477](https://doi.org/10.1016/j.ins.2023.119477)

---

# Znum 3.0.0

## Refactored arithmetic pipeline

- **Separated A and B computation** — `math_ops.py` rewritten with clear phases: `_compute_a_pairs`, `_merge_rows`, `_extract_trapezoid` for A; `_compute_b_columns`, `_compute_prob_pos`, `_compute_result_B_lp` for B. Old internal methods (`get_matrix_main`, `get_minimized_matrix`, `get_Q_from_matrix`, `get_prob_pos`) removed.
- **Added `Znum.fast()` context manager** — Thread-safe opt-in for fast B computation via `threading.local()`. Supports nesting and exception safety.

---

# Znum 2.1.0

## Performance

- **Replaced scipy with highspy** — Calls the HiGHS LP solver directly instead of going through scipy's Python wrapper. Same solver, same results, ~3x faster arithmetic operations. Dependency shrinks from 46MB (scipy) to 2.5MB (highspy).
- **LP model reuse** — Builds the HiGHS model once per `get_matrix` call and re-solves by changing only the RHS, avoiding redundant model construction.

### Dependency change
```diff
- scipy>=1.10.0
+ highspy>=1.7.0
```

---

# Znum 1.0.0

## Breaking Changes

- **`Beast` renamed to `MCDMUtils`** — All references to `Beast` must be updated.
- **`shouldNormalizeWeight` renamed to `normalize_weights`** — Affects `Topsis` and `Promethee` constructors.
- **`distanceType` renamed to `distance_type`** — Affects `Topsis` constructor.
- **Removed `Znum.Topsis`, `Znum.Sort`, `Znum.Promethee`, `Znum.Beast`, `Znum.Math`, `Znum.Dist`** — Import these directly from `znum` instead of accessing them as class attributes.
- **Removed `Vikor` module** — It had incorrect return values and no test coverage.
- **Removed `IncompatibleABPartsException`** — It was never raised anywhere.
- **Removed `Type` class (`znum/ztype.py`)** — Shape properties (`is_trapezoid`, `is_triangle`, `is_even`) are now directly on `Znum`.

## Bug Fixes

- **Fixed `Topsis.ordered_indices` sorting in wrong direction** — Was sorting worst-first instead of best-first. The core `solver_main()` was unaffected.
- **Fixed division-by-zero risks** in `sort.py`, `topsis.py`, and `utils.py` with proper guards.
- **Fixed silent B-value mutation** — Now emits a `UserWarning` when near-zero B values are adjusted to avoid degenerate LP solutions.
- **Fixed decorator wrappers** — Added `@functools.wraps` and switched to `isinstance()` checks in validators.

## Performance

- **Constant Z-number LP shortcut** — Ideal Z-numbers (used in Hellinger distance) now solve 1 LP instead of 10, reducing TOPSIS Hellinger runtime by ~27%.
- **Promethee symmetry exploitation** — Preference table computation now uses upper-triangle iteration, cutting Sort comparisons in half (~22% faster).

## Improvements

- Full type hints on all public APIs.
- Docstrings on all public classes and methods.
- Consistent snake_case naming throughout.
- Magic numbers replaced with named constants.
- `Promethee` API aligned with `Topsis` (added `result` property and guards on all accessor properties).
