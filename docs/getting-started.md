# Getting Started

## Installation

```bash
pip install znum
```

For development:

```bash
git clone https://github.com/MahammadNuriyev62/Znum.git
cd Znum
pip install -e ".[dev]"
```

## Your first Z-number

A Z-number has two parts, each defined as a trapezoidal fuzzy number with 4 values `[a1, a2, a3, a4]` where `a1 <= a2 <= a3 <= a4`:

```python
from znum import Znum

# A = "approximately 2 to 3" with reliability "around 0.7"
z = Znum(A=[1, 2, 3, 4], B=[0.6, 0.7, 0.8, 0.9])
print(z)
# Znum(A=[1.0, 2.0, 3.0, 4.0], B=[0.6, 0.7, 0.8, 0.9])
```

**A** defines the fuzzy restriction — the trapezoidal shape over possible values:

```
membership
    1 |    ____
      |   /    \
      |  /      \
    0 |_/________\___
        a1 a2 a3 a4
```

**B** defines reliability using the same trapezoidal shape, with values in `[0, 1]`.

## Crisp values

When you need an exact number (no fuzziness, full reliability), use the `crisp()` factory:

```python
five = Znum.crisp(5)
print(five)
# Znum(A=[5.0, 5.0, 5.0, 5.0], B=[1.0, 1.0, 1.0, 1.0])
```

This is equivalent to `Znum(A=[5, 5, 5, 5], B=[1, 1, 1, 1])` but more readable.

## Basic operations

```python
z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])

# Arithmetic
z3 = z1 + z2    # Addition
z4 = z1 - z2    # Subtraction
z5 = z1 * z2    # Multiplication
z6 = z1 / z2    # Division
z7 = z1 ** 2    # Power

# Scalar operations
z8 = z1 * 3     # Scalar multiplication

# Comparison
print(z1 < z2)   # True
print(z1 == z1)  # True

# Sorting
znums = [z2, z1]
sorted_znums = sorted(znums)  # [z1, z2]
```

## Serialization

```python
z = Znum(A=[1, 2, 3, 4], B=[0.6, 0.7, 0.8, 0.9])

# To JSON-compatible dict
z.to_json()
# {'A': [1.0, 2.0, 3.0, 4.0], 'B': [0.6, 0.7, 0.8, 0.9]}

# To flat numpy array
z.to_array()
# array([1. , 2. , 3. , 4. , 0.6, 0.7, 0.8, 0.9])

# Deep copy
z_copy = z.copy()
```

## What's next?

- [Creating Z-numbers](guide/creating.md) — all the ways to construct Z-numbers
- [Arithmetic](guide/arithmetic.md) — how fuzzy arithmetic works under the hood
- [Comparison & Sorting](guide/comparison.md) — fuzzy dominance-based ordering
- [TOPSIS](mcdm/topsis.md) — multi-criteria decision making with TOPSIS
- [PROMETHEE](mcdm/promethee.md) — multi-criteria decision making with PROMETHEE
