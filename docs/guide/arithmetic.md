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

## Fast mode

For performance-critical code, use `Znum.fast()` to skip the LP solver for B computation. Instead, B is computed as element-wise `min(B1, B2)` — the result is only as reliable as the least reliable input. **~19x faster.**

```python
with Znum.fast():
    result = z1 + z2    # B = min(z1.B, z2.B), no LP
    result = z1 * z2    # works for all operators
    result = z1 + z2 + z3  # chained operations stay fast
```

All operators (`+`, `-`, `*`, `/`) are supported. A computation is identical in both modes — only B differs. Outside the `with` block, arithmetic returns to the default LP mode.

**Key differences from LP mode:**

- LP mode: B values reflect how uncertainty compounds through operations (B degrades over many chained operations)
- Fast mode: B never goes below the minimum input B (stays stable through chains)

Use fast mode when you need speed and a conservative lower bound on reliability is acceptable (e.g., in an optimizer evaluating thousands of candidate solutions).

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
