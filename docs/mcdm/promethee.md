# PROMETHEE

**Preference Ranking Organization METHod for Enrichment Evaluations**

PROMETHEE ranks alternatives using pairwise preference comparisons. Each pair of alternatives is compared on every criterion using fuzzy dominance, producing net flow scores where higher = better.

## Quick example

```python
from znum import Znum, Promethee, MCDMUtils

# Weights
weights = [
    Znum(A=[0.2, 0.3, 0.4, 0.5], B=[0.1, 0.2, 0.3, 0.4]),
    Znum(A=[0.4, 0.5, 0.6, 0.7], B=[0.3, 0.4, 0.5, 0.6]),
]

# 3 alternatives, 2 criteria each
alt1 = [
    Znum(A=[7, 8, 9, 10], B=[0.6, 0.7, 0.8, 0.9]),
    Znum(A=[5, 6, 7, 8], B=[0.5, 0.6, 0.7, 0.8]),
]
alt2 = [
    Znum(A=[4, 5, 6, 7], B=[0.5, 0.6, 0.7, 0.8]),
    Znum(A=[8, 9, 10, 11], B=[0.6, 0.7, 0.8, 0.9]),
]
alt3 = [
    Znum(A=[6, 7, 8, 9], B=[0.4, 0.5, 0.6, 0.7]),
    Znum(A=[3, 4, 5, 6], B=[0.3, 0.4, 0.5, 0.6]),
]

criteria = [MCDMUtils.CriteriaType.BENEFIT, MCDMUtils.CriteriaType.BENEFIT]

table = [weights, alt1, alt2, alt3, criteria]

promethee = Promethee(table)
result = promethee.solve()
```

## Decision table format

Same format as [TOPSIS](topsis.md#decision-table-format):

```
table = [weights, alt1, alt2, ..., altN, criteria_types]
```

## Constructor parameters

```python
promethee = Promethee(
    table,
    normalize_weights=False,  # Normalize weights to sum to 1
)
```

## Accessing results

```python
promethee = Promethee(table)
result = promethee.solve()

# Sorted tuple of (index, net_flow_znum) — best first
print(promethee.result)
# ((1, Znum(...)), (0, Znum(...)), (2, Znum(...)))

# Indices sorted by rank (best first)
print(promethee.ordered_indices)
# [1, 0, 2]

# Best and worst
print(promethee.index_of_best_alternative)   # 1
print(promethee.index_of_worst_alternative)  # 2
```

## How it works

1. **Normalize** each criterion column (benefit or cost normalization)
2. **Pairwise comparison**: For each pair of alternatives (i, j), compute the fuzzy dominance score on each criterion using the NxF framework
3. **Apply weights**: Multiply each preference value by its criterion weight
4. **Sum preferences**: Collapse per-criterion preferences into a single value per pair
5. **Net flow**: For each alternative, compute `leaving_flow - entering_flow`
6. **Rank**: Sort by net flow (descending)

### Symmetry optimization

The pairwise comparison only computes the upper triangle of the preference matrix. Each `Sort.solver_main(z_i, z_j)` call gives the scores for both directions, so `preference[i][j]` and `preference[j][i]` are computed together. This cuts comparison time roughly in half.

!!! warning "Table is modified in-place"
    Like TOPSIS, PROMETHEE normalizes the table during solving. Deep copy the table if you need original values.

## PROMETHEE vs TOPSIS

| Feature | PROMETHEE | TOPSIS |
|---------|-----------|--------|
| Ranking method | Pairwise net flows | Distance to ideal |
| Result type | `(index, Znum)` tuples | Float closeness coefficients |
| Distance options | N/A | Hellinger or Simple |
| Best for | Outranking relationships | Distance-based ranking |

Both methods typically agree on the best alternative, but may differ in intermediate rankings.
