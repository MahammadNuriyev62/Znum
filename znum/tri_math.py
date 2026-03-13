"""Analytical engine for triangular Z-number arithmetic (Li et al. 2023).

Replaces LP-based B computation with closed-form extended triangular
distribution. Activated via Znum.fast() context manager for triangular
Z-numbers; non-triangular inputs fall back to LP.

Reference: Li et al. "The arithmetic of triangular Z-numbers with reduced
calculation complexity using an extension of triangular distribution",
Information Sciences 647, 2023. DOI: 10.1016/j.ins.2023.119477
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

# numpy.trapz was removed in NumPy 2.0, replaced by numpy.trapezoid
_trapz = getattr(np, 'trapezoid', None) or np.trapz

if TYPE_CHECKING:
    from znum.core import Znum


# ---- Extended Triangular PDF (Definition 20, Eq. 32, Eq. 36) ----

def make_ext_tri_params(a: float, m: float, b: float, v: float) -> tuple[float, float]:
    """Compute parameters for extended triangular PDF.

    Args:
        a, m, b: Triangular fuzzy number vertices (a <= m <= b).
        v: Reliability value in [0, 1].

    Returns:
        (beta, h) where:
        - beta = 2(1-v)/(b-a) — PDF value at endpoints a and b
        - h = 2v/(b-a) — PDF value at peak m
    """
    w = b - a
    if w <= 0:
        raise ValueError(f"Degenerate triangular: a={a}, b={b} (width={w})")
    beta = 2.0 * (1.0 - v) / w
    h = 2.0 * v / w
    return beta, h


def eval_ext_tri_pdf(a: float, m: float, b: float,
                     beta: float, h: float,
                     x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate extended triangular PDF at points x.

    PDF shape (Eq. 32):
        f(x) = beta + (h-beta)*(x-a)/(m-a),  a <= x <= m
        f(x) = beta + (h-beta)*(b-x)/(b-m),  m < x <= b
        f(x) = 0,                              otherwise

    Properties:
        v=1: peaked triangle (beta=0, h=2/(b-a))
        v=0: bathtub shape (beta=2/(b-a), h=0)
        v=0.5: uniform (beta=h=1/(b-a))
        Always integrates to 1.
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)

    # Left ramp [a, m]
    left_mask = (x >= a) & (x <= m)
    if m > a:
        result[left_mask] = beta + (h - beta) * (x[left_mask] - a) / (m - a)
    else:
        result[left_mask] = h  # degenerate: a == m

    # Right ramp (m, b]
    right_mask = (x > m) & (x <= b)
    if b > m:
        result[right_mask] = beta + (h - beta) * (b - x[right_mask]) / (b - m)
    else:
        result[right_mask] = h  # degenerate: m == b

    return result


# ---- Analytical matrix (drop-in for Math.get_matrix) ----

def analytical_matrix(z: Znum) -> np.ndarray:
    """Compute probability distribution matrix analytically.

    Drop-in replacement for LP-based Math.get_matrix(). For each B
    intermediate value v_j, evaluates the extended triangular PDF at
    A intermediate points and normalizes to a probability distribution.

    Returns:
        Matrix of shape (n_a_points, n_b_points), same as LP version.
    """
    a_vals = np.asarray(z.A_int["value"])
    b_vals = np.asarray(z.B_int["value"])
    n_a = len(a_vals)
    n_b = len(b_vals)

    a_tri = z.A_tri
    a, m, b = float(a_tri[0]), float(a_tri[1]), float(a_tri[2])

    # Handle crisp case (all A values equal)
    if a == b:
        return np.full((n_a, n_b), 1.0 / n_a)

    matrix = np.empty((n_a, n_b))

    for j, v_j in enumerate(b_vals):
        v_j = float(np.clip(v_j, 0.0, 1.0))
        beta, h = make_ext_tri_params(a, m, b, v_j)
        col = eval_ext_tri_pdf(a, m, b, beta, h, a_vals)

        col_sum = col.sum()
        if col_sum > 0:
            col = col / col_sum
        else:
            col = np.full(n_a, 1.0 / n_a)

        matrix[:, j] = col

    return matrix


# ---- A computation (closed-form for triangular) ----

def compute_A_tri(A1_tri: NDArray, A2_tri: NDArray, operation: int) -> NDArray:
    """Closed-form A result for triangular fuzzy number arithmetic.

    Args:
        A1_tri, A2_tri: Triangular (a, m, b) arrays.
        operation: Math.Operations constant.

    Returns:
        Result triangular (a, m, b) array.
    """
    from znum.math_ops import Math

    a1, m1, b1 = float(A1_tri[0]), float(A1_tri[1]), float(A1_tri[2])
    a2, m2, b2 = float(A2_tri[0]), float(A2_tri[1]), float(A2_tri[2])

    if operation == Math.Operations.ADDITION:
        return np.array([a1 + a2, m1 + m2, b1 + b2])

    if operation == Math.Operations.SUBTRACTION:
        return np.array([a1 - b2, m1 - m2, b1 - a2])

    if operation == Math.Operations.MULTIPLICATION:
        corners = [a1 * a2, a1 * b2, b1 * a2, b1 * b2]
        return np.array([min(corners), m1 * m2, max(corners)])

    if operation == Math.Operations.DIVISION:
        if a2 <= 0 <= b2:
            raise ValueError(
                "Division by zero: divisor support contains zero "
                f"[{a2}, {b2}]"
            )
        corners = [a1 / a2, a1 / b2, b1 / a2, b1 / b2]
        return np.array([min(corners), m1 / m2, max(corners)])

    raise ValueError(f"Unknown operation: {operation}")


# ---- Convolution of extended triangular PDFs ----

_N_CONV = 150  # default grid resolution (0.002 accuracy, optimal speed)


def _eval_pdf_2d(a, m, b, beta, h, x_2d: NDArray) -> NDArray:
    """Vectorized PDF eval on 2D array (n_z, n_t). Same formula as eval_ext_tri_pdf."""
    result = np.zeros_like(x_2d)
    if m > a:
        left = (x_2d >= a) & (x_2d <= m)
        result[left] = beta + (h - beta) * (x_2d[left] - a) / (m - a)
    else:
        left = x_2d == m
        result[left] = h
    if b > m:
        right = (x_2d > m) & (x_2d <= b)
        result[right] = beta + (h - beta) * (b - x_2d[right]) / (b - m)
    else:
        right = x_2d == m
        result[right] = h
    return result


def _convolve_addition(a1, m1, b1, beta1, h1,
                       a2, m2, b2, beta2, h2,
                       n_points: int) -> tuple[NDArray, NDArray]:
    """f_{X+Y}(z) = ∫ f_X(t) · f_Y(z-t) dt  — fully vectorized."""
    z_grid = np.linspace(a1 + a2, b1 + b2, n_points)
    t_grid = np.linspace(a1, b1, n_points)
    dt = t_grid[1] - t_grid[0] if n_points > 1 else 1.0
    f1 = eval_ext_tri_pdf(a1, m1, b1, beta1, h1, t_grid)          # (n_t,)
    arg2 = z_grid[:, None] - t_grid[None, :]                       # (n_z, n_t)
    f2 = _eval_pdf_2d(a2, m2, b2, beta2, h2, arg2)                 # (n_z, n_t)
    f_values = _trapz(f1[None, :] * f2, dx=dt, axis=1)           # (n_z,)
    return z_grid, f_values


def _convolve_subtraction(a1, m1, b1, beta1, h1,
                          a2, m2, b2, beta2, h2,
                          n_points: int) -> tuple[NDArray, NDArray]:
    """f_{X-Y}(z) = ∫ f_X(t) · f_Y(t-z) dt  — fully vectorized."""
    z_grid = np.linspace(a1 - b2, b1 - a2, n_points)
    t_grid = np.linspace(a1, b1, n_points)
    dt = t_grid[1] - t_grid[0] if n_points > 1 else 1.0
    f1 = eval_ext_tri_pdf(a1, m1, b1, beta1, h1, t_grid)
    arg2 = t_grid[None, :] - z_grid[:, None]                       # (n_z, n_t)
    f2 = _eval_pdf_2d(a2, m2, b2, beta2, h2, arg2)
    f_values = _trapz(f1[None, :] * f2, dx=dt, axis=1)
    return z_grid, f_values


def _convolve_multiplication(a1, m1, b1, beta1, h1,
                             a2, m2, b2, beta2, h2,
                             n_points: int) -> tuple[NDArray, NDArray]:
    """f_{XY}(z) = ∫ f_X(t) · f_Y(z/t) · (1/|t|) dt  — fully vectorized."""
    corners = [a1 * a2, a1 * b2, b1 * a2, b1 * b2]
    z_grid = np.linspace(min(corners), max(corners), n_points)
    t_grid = np.linspace(a1, b1, n_points)
    dt = t_grid[1] - t_grid[0] if n_points > 1 else 1.0
    f1 = eval_ext_tri_pdf(a1, m1, b1, beta1, h1, t_grid)           # (n_t,)
    safe_t = t_grid.copy()
    zero_mask = np.abs(safe_t) < 1e-15
    safe_t[zero_mask] = 1e-15
    arg2 = z_grid[:, None] / safe_t[None, :]                       # (n_z, n_t)
    f2 = _eval_pdf_2d(a2, m2, b2, beta2, h2, arg2)
    integrand = f1[None, :] * f2 / np.abs(safe_t)[None, :]
    integrand[:, zero_mask] = 0.0
    f_values = _trapz(integrand, dx=dt, axis=1)
    return z_grid, f_values


def _convolve_division(a1, m1, b1, beta1, h1,
                       a2, m2, b2, beta2, h2,
                       n_points: int) -> tuple[NDArray, NDArray]:
    """f_{X/Y}(z) = ∫ f_Y(t) · f_X(z·t) · |t| dt  — fully vectorized."""
    if a2 <= 0 <= b2:
        raise ValueError(
            "Division by zero: divisor support contains zero "
            f"[{a2}, {b2}]"
        )
    corners = [a1 / a2, a1 / b2, b1 / a2, b1 / b2]
    z_grid = np.linspace(min(corners), max(corners), n_points)
    t_grid = np.linspace(a2, b2, n_points)
    dt = t_grid[1] - t_grid[0] if n_points > 1 else 1.0
    f2 = eval_ext_tri_pdf(a2, m2, b2, beta2, h2, t_grid)           # (n_t,)
    arg1 = z_grid[:, None] * t_grid[None, :]                       # (n_z, n_t)
    f1 = _eval_pdf_2d(a1, m1, b1, beta1, h1, arg1)
    integrand = f2[None, :] * f1 * np.abs(t_grid)[None, :]
    f_values = _trapz(integrand, dx=dt, axis=1)
    return z_grid, f_values


_CONVOLVE_DISPATCH = None  # populated lazily to avoid import-time Math access


def _get_dispatch():
    global _CONVOLVE_DISPATCH
    if _CONVOLVE_DISPATCH is None:
        from znum.math_ops import Math
        _CONVOLVE_DISPATCH = {
            Math.Operations.ADDITION: _convolve_addition,
            Math.Operations.SUBTRACTION: _convolve_subtraction,
            Math.Operations.MULTIPLICATION: _convolve_multiplication,
            Math.Operations.DIVISION: _convolve_division,
        }
    return _CONVOLVE_DISPATCH


def convolve_pdfs(a1, m1, b1, v1, a2, m2, b2, v2,
                  operation: int, n_points: int = _N_CONV) -> tuple[NDArray, NDArray]:
    """Convolve two extended triangular PDFs for the given operation.

    Returns:
        (z_grid, f_values): Result PDF evaluated on a grid.
    """
    beta1, h1 = make_ext_tri_params(a1, m1, b1, v1)
    beta2, h2 = make_ext_tri_params(a2, m2, b2, v2)

    dispatch = _get_dispatch()
    fn = dispatch.get(operation)
    if fn is None:
        raise ValueError(f"Unknown operation: {operation}")

    return fn(a1, m1, b1, beta1, h1, a2, m2, b2, beta2, h2, n_points)


# ---- Triangular membership function ----

def _tri_membership(a: float, m: float, b: float, x: NDArray) -> NDArray:
    """Evaluate triangular membership function μ(x) at points x."""
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)

    if m > a:
        left = (x >= a) & (x <= m)
        result[left] = (x[left] - a) / (m - a)
    if b > m:
        right = (x > m) & (x <= b)
        result[right] = (b - x[right]) / (b - m)

    # Exact peak
    peak = x == m
    result[peak] = 1.0
    return result


# ---- B computation (Li et al. Steps 2-5) ----

def compute_base_value(A_result_tri: NDArray,
                       z_grid: NDArray, f_values: NDArray) -> float:
    """Step 4: v₁₂ = ∫ μ_A_result(x) · f₁₂(x) dx.

    Measures how much of the convolved PDF falls under the result's
    triangular membership function. Returns value in [0, 1].
    """
    a_r, m_r, b_r = float(A_result_tri[0]), float(A_result_tri[1]), float(A_result_tri[2])
    mu = _tri_membership(a_r, m_r, b_r, z_grid)
    dz = z_grid[1] - z_grid[0] if len(z_grid) > 1 else 1.0
    val = float(_trapz(mu * f_values, dx=dz))
    return np.clip(val, 0.0, 1.0)


def _batched_convolve_and_base(
    a1, m1, b1, a2, m2, b2,
    B1_tri: NDArray, B2_tri: NDArray,
    A_result_tri: NDArray, operation: int,
    n_points: int,
) -> NDArray:
    """Batch all 3 B-corner convolutions into one tensor operation.

    All 3 corners share the same grids (z_grid, t_grid) and only differ in
    PDF parameters (beta, h). We stack into (3, n_z, n_t) and do a single
    trapz call, eliminating 2/3 of linspace and reducing function overhead.

    Returns:
        Array of 3 base values [v_L, v_M, v_R].
    """
    from znum.math_ops import Math

    # Precompute PDF params for all 3 corners
    betas1, hs1 = np.empty(3), np.empty(3)
    betas2, hs2 = np.empty(3), np.empty(3)
    skip = np.zeros(3, dtype=bool)

    for k in range(3):
        v1_k, v2_k = float(B1_tri[k]), float(B2_tri[k])
        if v1_k <= 0 and v2_k <= 0:
            skip[k] = True
            betas1[k] = hs1[k] = betas2[k] = hs2[k] = 0.0
        else:
            betas1[k], hs1[k] = make_ext_tri_params(a1, m1, b1, v1_k)
            betas2[k], hs2[k] = make_ext_tri_params(a2, m2, b2, v2_k)

    if skip.all():
        return np.zeros(3)

    # Build shared grids once (operation-specific bounds)
    if operation == Math.Operations.ADDITION:
        z_grid = np.linspace(a1 + a2, b1 + b2, n_points)
        t_grid = np.linspace(a1, b1, n_points)
    elif operation == Math.Operations.SUBTRACTION:
        z_grid = np.linspace(a1 - b2, b1 - a2, n_points)
        t_grid = np.linspace(a1, b1, n_points)
    elif operation == Math.Operations.MULTIPLICATION:
        corners = [a1 * a2, a1 * b2, b1 * a2, b1 * b2]
        z_grid = np.linspace(min(corners), max(corners), n_points)
        t_grid = np.linspace(a1, b1, n_points)
    elif operation == Math.Operations.DIVISION:
        if a2 <= 0 <= b2:
            raise ValueError(f"Division by zero: [{a2}, {b2}]")
        corners = [a1 / a2, a1 / b2, b1 / a2, b1 / b2]
        z_grid = np.linspace(min(corners), max(corners), n_points)
        t_grid = np.linspace(a2, b2, n_points)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    dt = t_grid[1] - t_grid[0] if n_points > 1 else 1.0
    n_z, n_t = n_points, n_points

    # Evaluate f1 for all 3 corners → (3, n_t)
    # f1 params differ per corner, but use the same t_grid
    f1_stack = np.empty((3, n_t))
    for k in range(3):
        if skip[k]:
            f1_stack[k] = 0.0
        elif operation == Math.Operations.DIVISION:
            f1_stack[k] = eval_ext_tri_pdf(a2, m2, b2, betas2[k], hs2[k], t_grid)
        else:
            f1_stack[k] = eval_ext_tri_pdf(a1, m1, b1, betas1[k], hs1[k], t_grid)

    # Build shared 2D argument array for f2 → (n_z, n_t)
    if operation == Math.Operations.ADDITION:
        arg2 = z_grid[:, None] - t_grid[None, :]
    elif operation == Math.Operations.SUBTRACTION:
        arg2 = t_grid[None, :] - z_grid[:, None]
    elif operation == Math.Operations.MULTIPLICATION:
        safe_t = t_grid.copy()
        zero_mask = np.abs(safe_t) < 1e-15
        safe_t[zero_mask] = 1e-15
        arg2 = z_grid[:, None] / safe_t[None, :]
    else:  # DIVISION
        arg2 = z_grid[:, None] * t_grid[None, :]

    # Evaluate f2 for all 3 corners → (3, n_z, n_t)
    f2_stack = np.empty((3, n_z, n_t))
    for k in range(3):
        if skip[k]:
            f2_stack[k] = 0.0
        elif operation == Math.Operations.DIVISION:
            f2_stack[k] = _eval_pdf_2d(a1, m1, b1, betas1[k], hs1[k], arg2)
        else:
            f2_stack[k] = _eval_pdf_2d(a2, m2, b2, betas2[k], hs2[k], arg2)

    # Combine and integrate — (3, n_z, n_t)
    integrand = f1_stack[:, None, :] * f2_stack  # (3, n_z, n_t)

    if operation == Math.Operations.MULTIPLICATION:
        inv_t = 1.0 / np.abs(safe_t)
        inv_t[zero_mask] = 0.0
        integrand = integrand * inv_t[None, None, :]
    elif operation == Math.Operations.DIVISION:
        integrand = integrand * np.abs(t_grid)[None, None, :]

    # Single batched trapz: (3, n_z, n_t) → (3, n_z)
    f_all = _trapz(integrand, dx=dt, axis=2)

    # Compute base values: v₁₂ = ∫ μ_A_result(x) · f₁₂(x) dx
    a_r, m_r, b_r = float(A_result_tri[0]), float(A_result_tri[1]), float(A_result_tri[2])
    mu = _tri_membership(a_r, m_r, b_r, z_grid)  # (n_z,)
    dz = z_grid[1] - z_grid[0] if len(z_grid) > 1 else 1.0

    # (3, n_z) * (n_z,) → trapz → (3,)
    result_b = _trapz(f_all * mu[None, :], dx=dz, axis=1)
    result_b[skip] = 0.0
    return np.clip(result_b, 0.0, 1.0)


def compute_B_tri(A1_tri: NDArray, B1_tri: NDArray,
                  A2_tri: NDArray, B2_tri: NDArray,
                  A_result_tri: NDArray, operation: int,
                  n_points: int = _N_CONV) -> NDArray:
    """Full B computation following Li et al. Steps 2-5.

    For triangular B, uses alpha-cut corners (3 convolutions):
        (B1[0], B2[0]) → base_value → B_result[0]  (left support)
        (B1[1], B2[1]) → base_value → B_result[1]  (peak)
        (B1[2], B2[2]) → base_value → B_result[2]  (right support)

    Uses batched tensor convolution for all 3 corners simultaneously.

    Returns:
        Triangular B result as [b_L, b_M, b_R].
    """
    a1, m1, b1 = float(A1_tri[0]), float(A1_tri[1]), float(A1_tri[2])
    a2, m2, b2 = float(A2_tri[0]), float(A2_tri[1]), float(A2_tri[2])

    # Handle crisp A cases (skip convolution)
    if a1 == b1 or a2 == b2:
        return np.minimum(B1_tri, B2_tri)

    result_b = _batched_convolve_and_base(
        a1, m1, b1, a2, m2, b2,
        B1_tri, B2_tri, A_result_tri, operation, n_points,
    )

    # Ensure monotonicity for valid triangular B: b_L <= b_M <= b_R
    if result_b[0] > result_b[1]:
        result_b[0] = result_b[1]
    if result_b[2] < result_b[1]:
        result_b[2] = result_b[1]

    return result_b


# ---- Top-level solve ----

def tri_solve(z1: Znum, z2: Znum, operation: int) -> Znum:
    """Full analytical solve for two triangular Z-numbers.

    Computes both A and B analytically (no LP). Returns a Znum in
    trapezoidal format [a, m, m, b].
    """
    from znum.core import Znum as ZnumCls

    A1_tri, A2_tri = z1.A_tri, z2.A_tri
    B1_tri, B2_tri = z1.B_tri, z2.B_tri

    # Phase 1: A (closed-form)
    A_result = compute_A_tri(A1_tri, A2_tri, operation)

    # Phase 2: B (analytical via convolution)
    B_result = compute_B_tri(
        A1_tri, B1_tri, A2_tri, B2_tri, A_result, operation,
    )

    # Convert to trapezoidal [a, m, m, b]
    a_r, m_r, b_r = A_result
    bl, bm, br = B_result

    return ZnumCls(
        A=[round(float(a_r), 6), round(float(m_r), 6),
           round(float(m_r), 6), round(float(b_r), 6)],
        B=[round(float(bl), 6), round(float(bm), 6),
           round(float(bm), 6), round(float(br), 6)],
    )
