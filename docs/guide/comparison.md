# Comparison & Sorting

## Comparison operators

Znum supports all Python comparison operators:

```python
from znum import Znum

z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
z2 = Znum(A=[5, 6, 7, 8], B=[0.2, 0.3, 0.4, 0.5])

z1 < z2    # True
z1 > z2    # False
z1 <= z2   # True
z1 >= z2   # False
z1 == z1   # True
z1 == z2   # False
```

### Chained comparisons

Python's chained comparison syntax works:

```python
z1 = Znum(A=[1, 2, 3, 4], B=[0.2, 0.3, 0.4, 0.5])
z2 = Znum(A=[5, 6, 7, 8], B=[0.2, 0.3, 0.4, 0.5])
z3 = Znum(A=[9, 10, 11, 12], B=[0.2, 0.3, 0.4, 0.5])

z1 < z2 < z3    # True
z3 > z2 > z1    # True
z1 <= z1 <= z2  # True
```

## How comparison works

Comparisons use the **NxF (Necessity for Fuzzy) framework** — a fuzzy dominance-based approach. For two Z-numbers, the algorithm:

1. **Normalizes** both A parts together into `[0, 1]`
2. Computes **intermediate differences** between the normalized values
3. Evaluates **possibility measures** against three linguistic categories:
    - `nbF` (better): the first Z-number dominates
    - `neF` (equal): the two are roughly equivalent
    - `nwF` (worse): the second Z-number dominates
4. Combines A and B results into a final **dominance score** `(d, do)` where `do = 1 - d`

The comparison then checks:

- `z1 > z2` : `do(z1, z2) > do(z2, z1)`
- `z1 < z2` : `do(z1, z2) < do(z2, z1)`
- `z1 == z2` : `do(z1, z2) == 1 and do(z2, z1) == 1`

!!! note "Reliability matters"
    Comparisons consider **both** the fuzzy values (A) and reliability (B). A Z-number with lower A values but much higher reliability can dominate one with higher A values but low reliability. This is a fundamental property of Z-numbers — reliability is part of the information.

## Sorting

Z-numbers work with Python's built-in sorting:

```python
z1 = Znum(A=[5, 6, 7, 8], B=[0.2, 0.3, 0.4, 0.5])
z2 = Znum(A=[1, 2, 3, 4], B=[0.2, 0.3, 0.4, 0.5])
z3 = Znum(A=[3, 4, 5, 6], B=[0.2, 0.3, 0.4, 0.5])

sorted_znums = sorted([z1, z2, z3])
# [z2, z3, z1] — sorted ascending

best = max([z1, z2, z3])    # z1
worst = min([z1, z2, z3])   # z2
```

## Crisp comparisons

Crisp Z-numbers compare as expected for their numeric values:

```python
a = Znum.crisp(3)
b = Znum.crisp(5)

a < b   # True
a > b   # False
a == a  # True

# Sorting crisp values
values = [Znum.crisp(v) for v in [5, 1, 3, 2, 4]]
sorted_values = sorted(values)
# [crisp(1), crisp(2), crisp(3), crisp(4), crisp(5)]
```

## Equality with non-Znum types

Comparing a Znum with a non-Znum type always returns `False`:

```python
z = Znum.crisp(5)
z == 5          # False
z == "hello"    # False
```
