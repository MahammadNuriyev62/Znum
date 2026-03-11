"""
Property-based tests for Znum arithmetic operations.

These tests verify mathematical invariants that must hold regardless of
the specific LP solution. Unlike golden-value tests, they are robust
to floating-point drift from internal refactoring.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from znum import Znum


# --- Test fixtures ---

# Reusable Z-number inputs covering different ranges
ZNUMS = {
    "standard_1": Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4]),
    "standard_2": Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5]),
    "small": Znum(A=[0.5, 1, 1.5, 2], B=[0.05, 0.1, 0.15, 0.2]),
    "fractional_1": Znum(A=[0.1, 0.2, 0.3, 0.4], B=[0.1, 0.2, 0.3, 0.4]),
    "fractional_2": Znum(A=[0.2, 0.3, 0.4, 0.5], B=[0.2, 0.3, 0.4, 0.5]),
    "large_1": Znum(A=[100, 200, 300, 400], B=[0.1, 0.2, 0.3, 0.4]),
    "large_2": Znum(A=[50, 100, 150, 200], B=[0.2, 0.3, 0.4, 0.5]),
    "negative_1": Znum(A=[-4, -3, -2, -1], B=[0.1, 0.2, 0.3, 0.4]),
    "negative_2": Znum(A=[-2, -1, 0, 1], B=[0.15, 0.25, 0.35, 0.45]),
}

# All pairs of Z-numbers for binary operations
ZNUM_PAIRS = [
    ("standard_1", "standard_2"),
    ("standard_1", "small"),
    ("fractional_1", "fractional_2"),
    ("large_1", "large_2"),
    ("negative_1", "standard_1"),
    ("negative_1", "negative_2"),
]

# Pairs where all A values are positive (safe for multiplication/division)
POSITIVE_PAIRS = [
    ("standard_1", "standard_2"),
    ("standard_1", "small"),
    ("fractional_1", "fractional_2"),
    ("large_1", "large_2"),
]


def _is_monotonic(arr):
    """Check that array values are non-decreasing."""
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


# --- Structural validity ---


class TestResultStructure:
    """Every arithmetic result must be a structurally valid Z-number."""

    @pytest.mark.parametrize("name1,name2", ZNUM_PAIRS)
    def test_addition_result_valid(self, name1, name2):
        result = ZNUMS[name1] + ZNUMS[name2]
        assert len(result.A) == 4
        assert len(result.B) == 4

    @pytest.mark.parametrize("name1,name2", ZNUM_PAIRS)
    def test_subtraction_result_valid(self, name1, name2):
        result = ZNUMS[name1] - ZNUMS[name2]
        assert len(result.A) == 4
        assert len(result.B) == 4

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_multiplication_result_valid(self, name1, name2):
        result = ZNUMS[name1] * ZNUMS[name2]
        assert len(result.A) == 4
        assert len(result.B) == 4

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_division_result_valid(self, name1, name2):
        result = ZNUMS[name1] / ZNUMS[name2]
        assert len(result.A) == 4
        assert len(result.B) == 4


# --- A monotonicity ---


class TestAMonotonicity:
    """Result A values must be non-decreasing (valid trapezoid)."""

    @pytest.mark.parametrize("name1,name2", ZNUM_PAIRS)
    def test_addition_A_monotonic(self, name1, name2):
        result = ZNUMS[name1] + ZNUMS[name2]
        assert _is_monotonic(result.A), f"A not monotonic: {result.A}"

    @pytest.mark.parametrize("name1,name2", ZNUM_PAIRS)
    def test_subtraction_A_monotonic(self, name1, name2):
        result = ZNUMS[name1] - ZNUMS[name2]
        assert _is_monotonic(result.A), f"A not monotonic: {result.A}"

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_multiplication_A_monotonic(self, name1, name2):
        result = ZNUMS[name1] * ZNUMS[name2]
        assert _is_monotonic(result.A), f"A not monotonic: {result.A}"

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_division_A_monotonic(self, name1, name2):
        result = ZNUMS[name1] / ZNUMS[name2]
        assert _is_monotonic(result.A), f"A not monotonic: {result.A}"


# --- B monotonicity and non-negativity ---


class TestBProperties:
    """Result B values must be non-decreasing and non-negative."""

    @pytest.mark.parametrize("name1,name2", ZNUM_PAIRS)
    def test_addition_B_monotonic(self, name1, name2):
        result = ZNUMS[name1] + ZNUMS[name2]
        assert _is_monotonic(result.B), f"B not monotonic: {result.B}"

    @pytest.mark.parametrize("name1,name2", ZNUM_PAIRS)
    def test_subtraction_B_monotonic(self, name1, name2):
        result = ZNUMS[name1] - ZNUMS[name2]
        assert _is_monotonic(result.B), f"B not monotonic: {result.B}"

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_multiplication_B_monotonic(self, name1, name2):
        result = ZNUMS[name1] * ZNUMS[name2]
        assert _is_monotonic(result.B), f"B not monotonic: {result.B}"

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_division_B_monotonic(self, name1, name2):
        result = ZNUMS[name1] / ZNUMS[name2]
        assert _is_monotonic(result.B), f"B not monotonic: {result.B}"

    @pytest.mark.parametrize("name1,name2", ZNUM_PAIRS)
    def test_addition_B_nonnegative(self, name1, name2):
        result = ZNUMS[name1] + ZNUMS[name2]
        assert all(result.B >= 0), f"B has negative values: {result.B}"

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_multiplication_B_nonnegative(self, name1, name2):
        result = ZNUMS[name1] * ZNUMS[name2]
        assert all(result.B >= 0), f"B has negative values: {result.B}"


# --- Exact A-value formulas ---


class TestAdditionAExact:
    """Addition A values must equal element-wise sum."""

    @pytest.mark.parametrize("name1,name2", ZNUM_PAIRS)
    def test_A_equals_elementwise_sum(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = z1 + z2
        assert_array_almost_equal(result.A, z1.A + z2.A)


class TestSubtractionAExact:
    """Subtraction A values follow fuzzy interval subtraction."""

    @pytest.mark.parametrize("name1,name2", ZNUM_PAIRS)
    def test_A_equals_interval_subtraction(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = z1 - z2
        expected_A = [
            z1.A[0] - z2.A[3],
            z1.A[1] - z2.A[2],
            z1.A[2] - z2.A[1],
            z1.A[3] - z2.A[0],
        ]
        assert_array_almost_equal(result.A, expected_A)


# --- Scalar and power operations (no LP involvement) ---


class TestScalarOperations:
    """Scalar multiplication and power only affect A, not B."""

    @pytest.mark.parametrize("name", ZNUMS)
    @pytest.mark.parametrize("scalar", [0.5, 2, 3, 10])
    def test_scalar_mul_A(self, name, scalar):
        z = ZNUMS[name]
        result = z * scalar
        assert_array_almost_equal(result.A, z.A * scalar)

    @pytest.mark.parametrize("name", ZNUMS)
    @pytest.mark.parametrize("scalar", [0.5, 2, 3, 10])
    def test_scalar_mul_preserves_B(self, name, scalar):
        z = ZNUMS[name]
        result = z * scalar
        assert_array_equal(result.B, z.B)

    @pytest.mark.parametrize("name", ["standard_1", "standard_2", "small", "large_1"])
    @pytest.mark.parametrize("power", [0.5, 2, 3])
    def test_power_A(self, name, power):
        z = ZNUMS[name]
        result = z**power
        assert_array_almost_equal(result.A, z.A**power)

    @pytest.mark.parametrize("name", ["standard_1", "standard_2", "small", "large_1"])
    @pytest.mark.parametrize("power", [0.5, 2, 3])
    def test_power_preserves_B(self, name, power):
        z = ZNUMS[name]
        result = z**power
        assert_array_equal(result.B, z.B)


# --- Identity operations ---


class TestIdentity:
    """Identity operations must return exact results."""

    @pytest.mark.parametrize("name", ZNUMS)
    def test_add_zero_returns_self(self, name):
        z = ZNUMS[name]
        assert z + 0 is z

    @pytest.mark.parametrize("name", ZNUMS)
    def test_radd_zero_returns_self(self, name):
        z = ZNUMS[name]
        assert 0 + z is z

    @pytest.mark.parametrize("name", ZNUMS)
    def test_scalar_mul_by_one(self, name):
        z = ZNUMS[name]
        result = z * 1
        assert_array_almost_equal(result.A, z.A)
        assert_array_equal(result.B, z.B)

    @pytest.mark.parametrize("name", ["standard_1", "standard_2", "small"])
    def test_power_one(self, name):
        z = ZNUMS[name]
        result = z**1
        assert_array_almost_equal(result.A, z.A)
        assert_array_equal(result.B, z.B)


# --- Commutativity ---


class TestCommutativity:
    """Addition and multiplication A values must be commutative."""

    @pytest.mark.parametrize("name1,name2", ZNUM_PAIRS)
    def test_addition_commutative_A(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        r1 = z1 + z2
        r2 = z2 + z1
        assert_array_almost_equal(r1.A, r2.A)

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_multiplication_commutative_A(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        r1 = z1 * z2
        r2 = z2 * z1
        assert_array_almost_equal(r1.A, r2.A)


# --- Multiplication/Division A bounds ---


class TestMultiplicationABounds:
    """Multiplication A[0] and A[3] must be the min/max of corner products."""

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_A_outer_bounds(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = z1 * z2
        corners = [
            z1.A[0] * z2.A[0],
            z1.A[0] * z2.A[3],
            z1.A[3] * z2.A[0],
            z1.A[3] * z2.A[3],
        ]
        assert result.A[0] == pytest.approx(min(corners), abs=1e-6)
        assert result.A[3] == pytest.approx(max(corners), abs=1e-6)


class TestDivisionABounds:
    """Division A[0] and A[3] must be the min/max of corner quotients."""

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_A_outer_bounds(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = z1 / z2
        corners = [
            z1.A[0] / z2.A[0],
            z1.A[0] / z2.A[3],
            z1.A[3] / z2.A[0],
            z1.A[3] / z2.A[3],
        ]
        assert result.A[0] == pytest.approx(min(corners), abs=1e-6)
        assert result.A[3] == pytest.approx(max(corners), abs=1e-6)


# --- Determinism ---


class TestDeterminism:
    """Same computation must produce identical results."""

    @pytest.mark.parametrize("name1,name2", ZNUM_PAIRS)
    def test_addition_deterministic(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        r1 = z1 + z2
        r2 = z1 + z2
        assert_array_equal(r1.A, r2.A)
        assert_array_equal(r1.B, r2.B)

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_multiplication_deterministic(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        r1 = z1 * z2
        r2 = z1 * z2
        assert_array_equal(r1.A, r2.A)
        assert_array_equal(r1.B, r2.B)


# --- Ordering sanity (result vs operands) ---


class TestResultOrdering:
    """Arithmetic results should have sensible relationships to operands."""

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_sum_A_greater_than_parts(self, name1, name2):
        """Sum of positive Z-numbers has larger A support than either operand."""
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = z1 + z2
        assert result.A[0] >= z1.A[0]
        assert result.A[0] >= z2.A[0]
        assert result.A[3] >= z1.A[3]
        assert result.A[3] >= z2.A[3]

    @pytest.mark.parametrize("name", ["standard_1", "standard_2", "small"])
    def test_double_greater_than_original(self, name):
        """z * 2 should dominate z for positive Z-numbers."""
        z = ZNUMS[name]
        z_double = z * 2
        assert z_double > z

    @pytest.mark.parametrize("name", ["standard_1", "standard_2", "small"])
    def test_sqrt_less_than_original_when_gt_1(self, name):
        """sqrt(z) < z when all A values > 1."""
        z = ZNUMS[name]
        if z.A[0] > 1:
            z_sqrt = z**0.5
            assert z_sqrt < z
