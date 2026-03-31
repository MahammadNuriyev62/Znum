# API Reference

Complete reference for all public classes and methods in the `znum` package.

## Znum

```python
from znum import Znum
```

### Constructor

```python
Znum(
    A: list | None = None,       # Fuzzy restriction [a1, a2, a3, a4]
    B: list | None = None,       # Reliability [b1, b2, b3, b4]
    left: int = 4,               # Left slope intermediate points
    right: int = 4,              # Right slope intermediate points
    A_int: dict | None = None,   # Pre-computed A intermediate
    B_int: dict | None = None,   # Pre-computed B intermediate
)
```

### Factory methods

| Method | Returns | Description |
|--------|---------|-------------|
| `Znum.crisp(value)` | `Znum` | Crisp Z-number: `A=(v,v,v,v)`, `B=(1,1,1,1)` |
| `Znum.config(...)` | context manager | Configure Z-number computation: `fast_triangle`, `min_b`, `sort_a_weight`. |
| `Znum.fast(min_b=False)` | context manager | Convenience alias for `Znum.config(fast_triangle=True, min_b=...)`. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `A` | `ndarray` | Fuzzy restriction values (read/write) |
| `B` | `ndarray` | Reliability values (read/write) |
| `mu_A` | `ndarray` | Membership function of A (read-only) |
| `mu_B` | `ndarray` | Membership function of B (read-only) |
| `dimension` | `int` | Number of corner points |
| `is_trapezoid` | `bool` | True if 4 corner points |
| `is_triangle` | `bool` | True if `A[1] == A[-2]` |
| `is_triangular` | `bool` | True if both A and B are triangular (`A[1] == A[2]` and `B[1] == B[2]`) |
| `is_even` | `bool` | True if even number of points |
| `A_tri` | `ndarray \| None` | A as `[a, m, b]` triangle, or None if A is not triangular |
| `B_tri` | `ndarray \| None` | B as `[a, m, b]` triangle, or None if B is not triangular |

### Operators

| Operator | Signature | Description |
|----------|-----------|-------------|
| `+` | `Znum + Znum` | Fuzzy addition (LP) |
| `+` | `Znum + 0` | Returns self (enables `sum()`) |
| `-` | `Znum - Znum` | Fuzzy subtraction (LP) |
| `*` | `Znum * Znum` | Fuzzy multiplication (LP) |
| `*` | `Znum * scalar` | Scale A by scalar, copy B |
| `/` | `Znum / Znum` | Fuzzy division (LP) |
| `**` | `Znum ** n` | Element-wise power on A, copy B |
| `>` | `Znum > Znum` | Fuzzy dominance greater-than |
| `<` | `Znum < Znum` | Fuzzy dominance less-than |
| `>=` | `Znum >= Znum` | Fuzzy dominance greater-or-equal |
| `<=` | `Znum <= Znum` | Fuzzy dominance less-or-equal |
| `==` | `Znum == Znum` | Fuzzy dominance equality |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `copy()` | `Znum` | Deep copy |
| `to_json()` | `dict` | `{"A": [...], "B": [...]}` |
| `to_array()` | `ndarray` | Concatenated `[A..., B...]` |

### Static methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_default_A()` | `ndarray` | `[1, 2, 3, 4]` |
| `get_default_B()` | `ndarray` | `[0.1, 0.2, 0.3, 0.4]` |

### Znum.config()

```python
with Znum.config(fast_triangle=False, min_b=False, sort_a_weight=0.5):
    ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fast_triangle` | `bool` | `False` | Use Li et al. 2023 analytical engine for triangular Z-numbers |
| `min_b` | `bool` | `False` | Use `min(B1, B2)` element-wise instead of convolution for B |
| `sort_a_weight` | `float` | `0.5` | Weight for A in `Sort.solver_main`. B gets `1 - sort_a_weight`. Higher values make value (A) dominate over reliability (B) in comparisons |

Context manager that configures Z-number computation for all operations
within its scope. Thread-safe (uses `threading.local`). Supports nesting;
inner contexts restore the outer context's settings on exit.

`Znum.fast(min_b=False)` is a convenience alias for
`Znum.config(fast_triangle=True, min_b=...)`.

---

## Topsis

```python
from znum import Topsis
```

### Constructor

```python
Topsis(
    table: list[list],             # Decision matrix
    normalize_weights: bool = False,
    distance_type: int | None = None,  # Default: Hellinger
)
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `solve()` | `list[float]` | Run TOPSIS, return closeness coefficients |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `result` | `list[float]` | Closeness coefficients |
| `ordered_indices` | `list[int]` | Indices sorted best-first |
| `index_of_best_alternative` | `int` | Best alternative index |
| `index_of_worst_alternative` | `int` | Worst alternative index |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `Topsis.DistanceMethod.HELLINGER` | `2` | Hellinger distance (default) |
| `Topsis.DistanceMethod.SIMPLE` | `1` | Simple (Manhattan) distance |

---

## Promethee

```python
from znum import Promethee
```

### Constructor

```python
Promethee(
    table: list[list],             # Decision matrix
    normalize_weights: bool = False,
)
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `solve()` | `tuple` | Sorted `(index, net_flow)` tuples |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `result` | `tuple` | Sorted (index, Znum) tuples |
| `ordered_indices` | `list[int]` | Indices sorted best-first |
| `index_of_best_alternative` | `int` | Best alternative index |
| `index_of_worst_alternative` | `int` | Worst alternative index |

---

## MCDMUtils

```python
from znum import MCDMUtils
```

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MCDMUtils.CriteriaType.BENEFIT` | `"B"` | Benefit criterion (higher is better) |
| `MCDMUtils.CriteriaType.COST` | `"C"` | Cost criterion (lower is better) |

### Static methods

| Method | Description |
|--------|-------------|
| `normalize(znums, criteria_type)` | Normalize column in-place |
| `normalize_benefit(znums)` | Divide A by max(A) |
| `normalize_cost(znums)` | Divide min(A) by each A (reversed) |
| `normalize_weight(weights)` | Normalize weights to sum to 1 |
| `subtract_matrix(o1, o2)` | Element-wise subtraction |
| `accurate_sum(znums)` | Sequential sum for precision |
| `parse_table(table)` | Extract `[weights, alternatives, criteria_types]` |
| `numerate(column)` | Pair elements with 1-based indices |
| `sort_numerated_single_column_table(table)` | Sort numerated list descending |
| `transpose_matrix(matrix)` | Transpose a 2D matrix |

---

## Exceptions

```python
from znum import (
    InvalidAPartOfZnumException,
    InvalidBPartOfZnumException,
    InvalidZnumDimensionException,
    InvalidZnumCPartDimensionException,
    ZnumMustBeEvenException,
    ZnumsMustBeInSameDimensionException,
)
```

| Exception | When raised |
|-----------|-------------|
| `InvalidAPartOfZnumException` | A values are not non-decreasing |
| `InvalidBPartOfZnumException` | B values are not non-decreasing or outside `[0, 1]` |
| `InvalidZnumDimensionException` | A and B have different lengths |
| `InvalidZnumCPartDimensionException` | Non-trapezoid Z-number without custom C |
| `ZnumMustBeEvenException` | Odd number of elements in A/B |
| `ZnumsMustBeInSameDimensionException` | Operating on Z-numbers with different dimensions |
