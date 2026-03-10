# Znum

**Z-number arithmetic and multi-criteria decision making for Python.**

A Z-number $Z = (A, B)$ extends ordinary fuzzy numbers with a reliability component:

- **A** — a trapezoidal fuzzy number restricting the possible values of a variable
- **B** — a trapezoidal fuzzy number measuring the reliability (confidence) of A

Znum provides full arithmetic, comparison operators, and two MCDM solvers (TOPSIS and PROMETHEE) — all operating natively on Z-numbers.

## Features

- **Full arithmetic**: `+`, `-`, `*`, `/`, `**` between Z-numbers and scalars
- **Comparison operators**: `<`, `>`, `<=`, `>=`, `==` using fuzzy dominance
- **Sorting**: Works with Python's `sorted()`, `min()`, `max()`
- **TOPSIS**: Hellinger and Simple distance methods
- **PROMETHEE**: Pairwise preference with net flow ranking
- **Crisp values**: `Znum.crisp(x)` for exact numbers with full reliability

## Quick example

```python
from znum import Znum

# Fuzzy Z-numbers
z1 = Znum(A=[1, 2, 3, 4], B=[0.6, 0.7, 0.8, 0.9])
z2 = Znum(A=[2, 3, 5, 7], B=[0.5, 0.6, 0.7, 0.8])

# Arithmetic
z3 = z1 + z2
z4 = z1 * z2

# Comparison
print(z1 < z2)  # True

# Crisp (exact) values
five = Znum.crisp(5)
ten = Znum.crisp(10)
print(five + ten)  # Znum(A=[15.0, 15.0, 15.0, 15.0], B=[1.0, 1.0, 1.0, 1.0])
```

## Requirements

- Python >= 3.10
- NumPy >= 1.24
- SciPy >= 1.10
