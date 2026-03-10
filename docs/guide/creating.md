# Creating Z-numbers

## Standard constructor

The most general way to create a Z-number:

```python
from znum import Znum

z = Znum(A=[1, 2, 3, 4], B=[0.6, 0.7, 0.8, 0.9])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | list or array | `[1, 2, 3, 4]` | Fuzzy restriction values (trapezoidal) |
| `B` | list or array | `[0.1, 0.2, 0.3, 0.4]` | Reliability values in `[0, 1]` |
| `left` | int | `4` | Intermediate points on the left slope |
| `right` | int | `4` | Intermediate points on the right slope |
| `C` | list or array | `[0, 1, 1, 0]` | Membership function values |

### Validation rules

- **A must be non-decreasing**: `a1 <= a2 <= a3 <= a4`
- **B must be non-decreasing and in [0, 1]**: `0 <= b1 <= b2 <= b3 <= b4 <= 1`
- **A and B must have the same length**

```python
# Valid
Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])     # Standard
Znum(A=[5, 5, 5, 5], B=[1, 1, 1, 1])               # Crisp
Znum(A=[-3, -1, 0, 2], B=[0.5, 0.6, 0.7, 0.8])     # Negative values
Znum(A=[0, 0, 0, 0], B=[0.5, 0.5, 0.5, 0.5])       # Zero

# Invalid - raises exceptions
Znum(A=[4, 3, 2, 1], B=[0.1, 0.2, 0.3, 0.4])   # A not non-decreasing
Znum(A=[1, 2, 3, 4], B=[0.4, 0.3, 0.2, 0.1])   # B not non-decreasing
Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 1.5])   # B > 1
```

## Crisp values

For exact numbers with no fuzziness and full reliability:

```python
z = Znum.crisp(5)
# Equivalent to: Znum(A=[5, 5, 5, 5], B=[1, 1, 1, 1])
```

This is useful when mixing exact and fuzzy values in calculations:

```python
price = Znum(A=[90, 95, 105, 110], B=[0.7, 0.8, 0.9, 1.0])
tax_rate = Znum.crisp(0.2)

tax = price * tax_rate  # Fuzzy tax amount
```

`Znum.crisp()` works with any numeric value:

```python
Znum.crisp(0)       # Zero
Znum.crisp(-3)      # Negative
Znum.crisp(3.14)    # Float
Znum.crisp(1e10)    # Large
Znum.crisp(1e-10)   # Small
```

### Properties of crisp Z-numbers

- `C` (membership) is automatically set to `[1, 1, 1, 1]` since all A values are equal
- `dimension` is always 4
- `is_trapezoid` is always `True`
- No B-value warning is triggered (B = [1,1,1,1] is well above the threshold)

## Default constructor

Calling `Znum()` with no arguments uses defaults:

```python
z = Znum()
print(z)
# Znum(A=[1.0, 2.0, 3.0, 4.0], B=[0.1, 0.2, 0.3, 0.4])
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `A` | ndarray | Fuzzy restriction values (read/write) |
| `B` | ndarray | Reliability values (read/write) |
| `C` | ndarray | Membership function values (read/write) |
| `dimension` | int | Number of corner points (typically 4) |
| `is_trapezoid` | bool | `True` if 4 corner points |
| `is_triangle` | bool | `True` if `A[1] == A[2]` (degenerate trapezoid) |
| `is_even` | bool | `True` if even number of corner points |

## Copy and serialization

```python
z = Znum(A=[1, 2, 3, 4], B=[0.6, 0.7, 0.8, 0.9])

# Deep copy (independent instance)
z2 = z.copy()

# JSON-serializable dict
z.to_json()   # {'A': [1.0, 2.0, 3.0, 4.0], 'B': [0.6, 0.7, 0.8, 0.9]}

# Flat numpy array [A..., B...]
z.to_array()  # array([1. , 2. , 3. , 4. , 0.6, 0.7, 0.8, 0.9])
```

## Near-zero B warning

If `B[-1] < 0.001`, a small epsilon is added to B values to avoid degenerate linear programming solutions. A warning is emitted:

```python
import warnings
z = Znum(A=[1, 2, 3, 4], B=[0, 0, 0, 0.0005])
# UserWarning: B[-1] < 0.001: small epsilon added to B values...
```

This does not apply to `Znum.crisp()` since B = [1, 1, 1, 1].
