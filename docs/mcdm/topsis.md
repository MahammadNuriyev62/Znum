# TOPSIS

**Technique for Order of Preference by Similarity to Ideal Solution**

TOPSIS ranks alternatives by measuring their distance to an ideal best and ideal worst solution. Alternatives closer to the ideal best and farther from the ideal worst are ranked higher.

## Quick example

```python
from znum import Znum, Topsis, MCDMUtils

# Weights for each criterion
weights = [
    Znum(A=[0.2, 0.3, 0.4, 0.5], B=[0.1, 0.2, 0.3, 0.4]),
    Znum(A=[0.4, 0.5, 0.6, 0.7], B=[0.3, 0.4, 0.5, 0.6]),
    Znum(A=[0.1, 0.2, 0.3, 0.4], B=[0.2, 0.3, 0.4, 0.5]),
]

# 3 alternatives, each with 3 criteria values
alt1 = [
    Znum(A=[7, 8, 9, 10], B=[0.6, 0.7, 0.8, 0.9]),
    Znum(A=[5, 6, 7, 8], B=[0.5, 0.6, 0.7, 0.8]),
    Znum(A=[3, 4, 5, 6], B=[0.4, 0.5, 0.6, 0.7]),
]
alt2 = [
    Znum(A=[4, 5, 6, 7], B=[0.5, 0.6, 0.7, 0.8]),
    Znum(A=[8, 9, 10, 11], B=[0.6, 0.7, 0.8, 0.9]),
    Znum(A=[6, 7, 8, 9], B=[0.3, 0.4, 0.5, 0.6]),
]
alt3 = [
    Znum(A=[6, 7, 8, 9], B=[0.4, 0.5, 0.6, 0.7]),
    Znum(A=[3, 4, 5, 6], B=[0.3, 0.4, 0.5, 0.6]),
    Znum(A=[8, 9, 10, 11], B=[0.5, 0.6, 0.7, 0.8]),
]

# Criteria types
criteria = [
    MCDMUtils.CriteriaType.BENEFIT,
    MCDMUtils.CriteriaType.BENEFIT,
    MCDMUtils.CriteriaType.COST,
]

# Build table: [weights, alt1, alt2, ..., altN, criteria_types]
table = [weights, alt1, alt2, alt3, criteria]

# Solve
topsis = Topsis(table)
result = topsis.solve()
```

## Decision table format

The table is a list of lists with this structure:

```
table = [
    [w1, w2, ..., wM],          # Row 0: weights (M criteria)
    [a11, a12, ..., a1M],       # Row 1: alternative 1
    [a21, a22, ..., a2M],       # Row 2: alternative 2
    ...
    [aN1, aN2, ..., aNM],       # Row N: alternative N
    [type1, type2, ..., typeM], # Last row: criteria types
]
```

- **Weights**: Z-numbers representing the importance of each criterion
- **Alternatives**: Z-numbers representing each alternative's score per criterion
- **Criteria types**: `MCDMUtils.CriteriaType.BENEFIT` (`"B"`) or `MCDMUtils.CriteriaType.COST` (`"C"`)

## Constructor parameters

```python
topsis = Topsis(
    table,
    normalize_weights=False,  # Normalize weight vector to sum to 1
    distance_type=None,       # Distance method (default: Hellinger)
)
```

## Distance methods

| Method | Constant | Description |
|--------|----------|-------------|
| Hellinger | `Topsis.DistanceMethod.HELLINGER` | Default. Uses Hellinger distance with A, B, and H components |
| Simple | `Topsis.DistanceMethod.SIMPLE` | Manhattan-style distance to crisp ideal |

```python
# Hellinger distance (default)
topsis = Topsis(table)

# Simple distance
topsis = Topsis(table, distance_type=Topsis.DistanceMethod.SIMPLE)
```

## Accessing results

```python
topsis = Topsis(table)
result = topsis.solve()

# Closeness coefficients (list of floats, one per alternative)
print(topsis.result)
# [0.523, 0.612, 0.489]

# Indices sorted by rank (best first)
print(topsis.ordered_indices)
# [1, 0, 2]

# Best and worst
print(topsis.index_of_best_alternative)   # 1
print(topsis.index_of_worst_alternative)  # 2
```

!!! warning "Table is modified in-place"
    TOPSIS normalizes the table during solving. If you need the original values, make a deep copy before calling `solve()`:

    ```python
    import copy
    table_copy = copy.deepcopy(table)
    topsis = Topsis(table_copy)
    topsis.solve()
    ```

## Weight normalization

When alternatives use different scales, normalize weights:

```python
topsis = Topsis(table, normalize_weights=True)
result = topsis.solve()
```

This divides each weight by the sum of all weights before applying them.

## Normalization

Before computing distances, each criterion column is normalized:

- **Benefit criteria**: Each A value is divided by the maximum A value in the column
- **Cost criteria**: The minimum A value in the column is divided by each A value (and reversed)
