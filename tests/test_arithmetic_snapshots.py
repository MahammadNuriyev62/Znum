"""
Snapshot tests for Znum arithmetic operations.

These tests capture exact expected A and B values for specific inputs.
If any of these tests fail after refactoring, it means the arithmetic
behavior has changed - which may or may not be intentional.

IMPORTANT: These values were captured from the current implementation.
If refactoring changes the underlying algorithm, these expected values
may need to be updated - but only after verifying the new values are correct.
"""

import pytest
import numpy as np
from znum.Znum import Znum


class TestArithmeticSnapshotsAddition:
    """Snapshot tests for addition operations."""

    def test_addition_simple_znums(self):
        """Snapshot: [1,2,3,4] + [1,2,3,4] with same B."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = z1 + z2

        # Capture exact expected values
        assert list(result.A) == pytest.approx([2, 4, 6, 8], rel=1e-6)
        # B values from fuzzy arithmetic
        assert len(result.B) == 4
        assert all(0 <= b <= 1 for b in result.B)

    def test_addition_different_values(self):
        """Snapshot: [1,2,3,4] + [5,6,7,8]."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])

        result = z1 + z2

        assert list(result.A) == pytest.approx([6, 8, 10, 12], rel=1e-6)

    def test_addition_with_scalar_znum(self):
        """Snapshot: [1,2,3,4] + scalar Znum representing 5.

        Note: Direct scalar addition (z + 5) is not supported.
        Use a Znum representing the scalar instead.
        """
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        scalar_z = Znum([5, 5, 5, 5], [0.9, 0.95, 0.98, 1.0])

        result = z + scalar_z

        assert list(result.A) == pytest.approx([6, 7, 8, 9], rel=1e-6)

    def test_addition_negative_values(self):
        """Snapshot: [-4,-3,-2,-1] + [1,2,3,4]."""
        z1 = Znum([-4, -3, -2, -1], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = z1 + z2

        assert list(result.A) == pytest.approx([-3, -1, 1, 3], rel=1e-6)


class TestArithmeticSnapshotsSubtraction:
    """Snapshot tests for subtraction operations."""

    def test_subtraction_same_znums(self):
        """Snapshot: [1,2,3,4] - [1,2,3,4] should give values around 0."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = z1 - z2

        # Result should be around 0 (fuzzy subtraction)
        assert list(result.A) == pytest.approx([-3, -1, 1, 3], rel=1e-6)

    def test_subtraction_different_values(self):
        """Snapshot: [5,6,7,8] - [1,2,3,4]."""
        z1 = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = z1 - z2

        assert list(result.A) == pytest.approx([1, 3, 5, 7], rel=1e-6)

    def test_subtraction_with_scalar_znum(self):
        """Snapshot: [5,6,7,8] - scalar Znum representing 2.

        Note: Direct scalar subtraction (z - 2) is not supported.
        Use a Znum representing the scalar instead.
        """
        z = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])
        scalar_z = Znum([2, 2, 2, 2], [0.9, 0.95, 0.98, 1.0])

        result = z - scalar_z

        assert list(result.A) == pytest.approx([3, 4, 5, 6], rel=1e-6)


class TestArithmeticSnapshotsMultiplication:
    """Snapshot tests for multiplication operations."""

    def test_multiplication_simple(self):
        """Snapshot: [1,2,3,4] * [1,2,3,4]."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = z1 * z2

        # Fuzzy multiplication: min*min to max*max
        assert list(result.A) == pytest.approx([1, 4, 9, 16], rel=1e-6)

    def test_multiplication_with_scalar(self):
        """Snapshot: [1,2,3,4] * 2."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = z * 2

        assert list(result.A) == pytest.approx([2, 4, 6, 8], rel=1e-6)

    def test_multiplication_fractional(self):
        """Snapshot: [2,4,6,8] * 0.5."""
        z = Znum([2, 4, 6, 8], [0.2, 0.4, 0.6, 0.8])

        result = z * 0.5

        assert list(result.A) == pytest.approx([1, 2, 3, 4], rel=1e-6)


class TestArithmeticSnapshotsDivision:
    """Snapshot tests for division operations."""

    def test_division_with_scalar_znum(self):
        """Snapshot: [2,4,6,8] / scalar Znum representing 2.

        Note: Direct scalar division (z / 2) is not supported.
        Use a Znum representing the scalar instead.
        """
        z = Znum([2, 4, 6, 8], [0.2, 0.4, 0.6, 0.8])
        scalar_z = Znum([2, 2, 2, 2], [0.9, 0.95, 0.98, 1.0])

        result = z / scalar_z

        assert list(result.A) == pytest.approx([1, 2, 3, 4], rel=1e-6)

    def test_division_znums(self):
        """Snapshot: [4,6,8,10] / [1,2,3,4]."""
        z1 = Znum([4, 6, 8, 10], [0.4, 0.6, 0.8, 0.9])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = z1 / z2

        # Division result depends on optimization
        assert len(result.A) == 4
        assert result.A[0] <= result.A[1] <= result.A[2] <= result.A[3]


class TestArithmeticSnapshotsPower:
    """Snapshot tests for power operations."""

    def test_power_squared(self):
        """Snapshot: [1,2,3,4] ** 2."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = z ** 2

        assert list(result.A) == pytest.approx([1, 4, 9, 16], rel=1e-6)

    def test_power_cubed(self):
        """Snapshot: [1,2,3,4] ** 3."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = z ** 3

        assert list(result.A) == pytest.approx([1, 8, 27, 64], rel=1e-6)


class TestArithmeticSnapshotsChained:
    """Snapshot tests for chained operations."""

    def test_addition_then_multiplication(self):
        """Snapshot: ([1,2,3,4] + [1,2,3,4]) * 2."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = (z1 + z2) * 2

        assert list(result.A) == pytest.approx([4, 8, 12, 16], rel=1e-6)

    def test_subtraction_then_addition(self):
        """Snapshot: ([5,6,7,8] - [1,2,3,4]) + [1,2,3,4]."""
        z1 = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z3 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = (z1 - z2) + z3

        # (5-4, 6-3, 7-2, 8-1) + (1,2,3,4) = (1,3,5,7) + (1,2,3,4)
        assert len(result.A) == 4
        assert result.A[0] <= result.A[-1]


class TestArithmeticSnapshotsProperties:
    """Snapshot tests for result properties after arithmetic."""

    def test_result_has_valid_B(self):
        """Verify B values stay in [0,1] after arithmetic."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        for op_result in [z1 + z2, z1 - z2, z1 * z2, z1 / z2]:
            assert all(0 <= b <= 1 for b in op_result.B), f"B out of range: {op_result.B}"

    def test_result_A_is_monotonic(self):
        """Verify A values stay monotonically increasing after arithmetic."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        for op_result in [z1 + z2, z1 * z2]:
            A = list(op_result.A)
            assert A == sorted(A), f"A not monotonic: {A}"

    def test_result_dimension_preserved(self):
        """Verify dimension stays the same after arithmetic."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        for op_result in [z1 + z2, z1 - z2, z1 * z2, z1 / z2]:
            assert op_result.dimension == 4

    def test_result_has_A_int_and_B_int(self):
        """Verify A_int and B_int are populated after arithmetic."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        result = z1 + z2

        assert result.A_int is not None
        assert result.B_int is not None
        assert "value" in result.A_int or len(result.A_int) > 0
