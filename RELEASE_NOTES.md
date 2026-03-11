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
