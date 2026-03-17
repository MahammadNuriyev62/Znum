# Arithmetic Operations

Znum supports all standard arithmetic operators between Z-numbers and with scalars.

## Operators

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Addition | `z1 + z2` | Fuzzy addition via LP |
| Subtraction | `z1 - z2` | Fuzzy subtraction via LP |
| Multiplication | `z1 * z2` | Fuzzy multiplication via LP |
| Division | `z1 / z2` | Fuzzy division via LP |
| Power | `z1 ** n` | Element-wise power on A |
| Scalar multiply | `z1 * 3` | Element-wise multiply A by scalar |

## Z-number arithmetic

```python
from znum import Znum

z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])

z3 = z1 + z2    # Addition
z4 = z1 - z2    # Subtraction
z5 = z1 * z2    # Multiplication
z6 = z1 / z2    # Division
```

### How it works

The arithmetic pipeline has two independent phases:

1. **A computation** (no LP): Cross-product of A intermediates, apply the operation, merge duplicates, extract trapezoidal corners. Pure arithmetic.
2. **B computation** (LP): Solve linear programs (via [HiGHS](https://ergo-code.github.io/HiGHS/)) to build probability distributions, then derive the result B via probability-possibility transform.

This produces correct fuzzy results that respect the membership function shape, but the LP step makes it slower than simple interval arithmetic.

## Fast mode (Li et al. 2023)

For performance-critical code, use `Znum.fast()` to switch B computation from LP to an analytical method based on extended triangular distributions ([Li et al. 2023](https://doi.org/10.1016/j.ins.2023.119477)). **~5x faster**, deterministic, and produces narrower B spread than LP.

Requires **triangular** Z-numbers (where `A[1] == A[2]` and `B[1] == B[2]`). Non-triangular Z-numbers automatically fall back to LP inside the `fast()` block.

```python
# Triangular Z-numbers: use [a, m, m, b] form
z1 = Znum(A=[1, 2, 2, 3], B=[0.7, 0.8, 0.8, 0.9])
z2 = Znum(A=[7, 8, 8, 9], B=[0.4, 0.5, 0.5, 0.6])

with Znum.fast():
    result = z1 + z2    # analytical, no LP
    result = z1 * z2    # works for all operators
    result = z1 + z2 + z3  # chained operations stay fast
```

All operators (`+`, `-`, `*`, `/`) are supported. A computation is identical in both modes. Outside the `with` block, arithmetic returns to the default LP mode.

**Key differences from LP mode:**

- LP mode: B is derived via probability-possibility transform using linear programming (works for any trapezoidal shape)
- Fast mode: B is computed analytically using extended triangular distribution convolutions (triangular only, narrower B spread for high-B inputs)

## Scalar operations

```python
z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])

z * 3       # A is scaled, B is preserved
z * 0.5     # Half the fuzzy values
3 * z       # Right-multiplication also works (via __rmul__)
```

Scalar multiplication directly scales A values and copies B unchanged. It does **not** use LP, so it's very fast.

## Power operation

```python
z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])

z ** 2      # Square: A=[1, 4, 9, 16], B unchanged
z ** 0.5    # Square root: A=[1, 1.41, 1.73, 2], B unchanged
z ** 3      # Cube
```

Power applies element-wise to A and copies B.

## Adding zero

Adding `0` (int or float) returns the Z-number unchanged. This enables `sum()`:

```python
znums = [Znum(A=[1,2,3,4], B=[0.1,0.2,0.3,0.4]),
         Znum(A=[2,3,4,5], B=[0.2,0.3,0.4,0.5])]

total = sum(znums)  # Works because sum() starts with 0
```

## Crisp arithmetic

Arithmetic between crisp Z-numbers produces crisp results:

```python
a = Znum.crisp(3)
b = Znum.crisp(4)

print(a + b)   # Znum(A=[7.0, 7.0, 7.0, 7.0], ...)
print(a * b)   # Znum(A=[12.0, 12.0, 12.0, 12.0], ...)
print(a - a)   # Znum(A=[0.0, 0.0, 0.0, 0.0], ...)
print(a / a)   # Znum(A=[1.0, 1.0, 1.0, 1.0], ...)
```

Mixing crisp and fuzzy Z-numbers works naturally:

```python
crisp = Znum.crisp(5)
fuzzy = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])

result = crisp + fuzzy  # A=[6.0, 7.0, 8.0, 9.0]
```

## Chaining operations

Operations can be chained freely:

```python
z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])
z3 = Znum(A=[0.5, 1, 1.5, 2], B=[0.3, 0.4, 0.5, 0.6])

result = (z1 + z2) * z3
result = z1 ** 2 + z2 * 0.5 - z3
result = (z1 * z2) / (z1 + z3)
```

## Precise summation

For summing many Z-numbers with better precision:

```python
from znum import MCDMUtils

znums = [z1, z2, z3]
total = MCDMUtils.accurate_sum(znums)
```

This sums sequentially (left-to-right) to maintain precision, which is equivalent to `sum()` but explicit about the intent.
