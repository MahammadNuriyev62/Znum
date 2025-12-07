"""
Comprehensive E2E tests for Znum comparison operations.

These tests verify that all comparison operations produce correct results.
The expected values were generated from the current (ground truth) implementation.
"""

import pytest
from znum.Znum import Znum


class TestZnumEquality:
    """Tests for Znum equality comparison."""

    def test_equality_self(self):
        """Test that a Z-number equals itself."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        assert z1 == z1

    def test_equality_identical_values(self):
        """Test that two Z-numbers with identical values are equal."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z4 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        assert z1 == z4

    def test_equality_different_A(self):
        """Test that Z-numbers with different A values are not equal."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])
        assert not (z1 == z2)

    def test_equality_different_values(self):
        """Test that Z-numbers with different values are not equal."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z3 = Znum(A=[0.5, 1, 1.5, 2], B=[0.05, 0.1, 0.15, 0.2])
        assert not (z1 == z3)


class TestZnumGreaterThan:
    """Tests for Znum greater than comparison."""

    def test_greater_than_self(self):
        """Test that a Z-number is not greater than itself."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        assert not (z1 > z1)

    def test_greater_than_larger_A(self):
        """Test comparison where second has larger A values."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])
        assert z2 > z1
        assert not (z1 > z2)

    def test_greater_than_smaller_values(self):
        """Test comparison with smaller Z-number."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z3 = Znum(A=[0.5, 1, 1.5, 2], B=[0.05, 0.1, 0.15, 0.2])
        assert z1 > z3
        assert not (z3 > z1)

    def test_greater_than_chain(self):
        """Test transitive greater than relationship."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])
        z3 = Znum(A=[0.5, 1, 1.5, 2], B=[0.05, 0.1, 0.15, 0.2])
        assert z2 > z3  # If z2 > z1 and z1 > z3, then z2 > z3


class TestZnumLessThan:
    """Tests for Znum less than comparison."""

    def test_less_than_self(self):
        """Test that a Z-number is not less than itself."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        assert not (z1 < z1)

    def test_less_than_larger_values(self):
        """Test comparison with larger Z-number."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])
        assert z1 < z2
        assert not (z2 < z1)

    def test_less_than_smaller_values(self):
        """Test comparison with smaller Z-number."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z3 = Znum(A=[0.5, 1, 1.5, 2], B=[0.05, 0.1, 0.15, 0.2])
        assert z3 < z1
        assert not (z1 < z3)


class TestZnumGreaterThanOrEqual:
    """Tests for Znum greater than or equal comparison."""

    def test_greater_than_or_equal_self(self):
        """Test that a Z-number is >= itself."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        assert z1 >= z1

    def test_greater_than_or_equal_identical(self):
        """Test >= with identical Z-numbers."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z4 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        assert z1 >= z4

    def test_greater_than_or_equal_larger(self):
        """Test >= with larger Z-number."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])
        assert z2 >= z1
        assert not (z1 >= z2)

    def test_greater_than_or_equal_smaller(self):
        """Test >= with smaller Z-number."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z3 = Znum(A=[0.5, 1, 1.5, 2], B=[0.05, 0.1, 0.15, 0.2])
        assert z1 >= z3


class TestZnumLessThanOrEqual:
    """Tests for Znum less than or equal comparison."""

    def test_less_than_or_equal_self(self):
        """Test that a Z-number is <= itself."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        assert z1 <= z1

    def test_less_than_or_equal_identical(self):
        """Test <= with identical Z-numbers."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z4 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        assert z1 <= z4

    def test_less_than_or_equal_larger(self):
        """Test <= with larger Z-number."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])
        assert z1 <= z2
        assert not (z2 <= z1)

    def test_less_than_or_equal_smaller(self):
        """Test <= with smaller Z-number."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z3 = Znum(A=[0.5, 1, 1.5, 2], B=[0.05, 0.1, 0.15, 0.2])
        assert z3 <= z1


class TestZnumComplexComparisons:
    """Tests for more complex comparison scenarios."""

    def test_overlapping_znums(self):
        """Test comparison with overlapping Z-number ranges."""
        z5 = Znum(A=[1.5, 2.5, 3.5, 4.5], B=[0.15, 0.25, 0.35, 0.45])
        z6 = Znum(A=[1, 2.5, 3, 4.5], B=[0.1, 0.25, 0.3, 0.45])
        assert z5 > z6
        assert not (z5 < z6)
        assert not (z5 == z6)

    def test_negative_znum_comparisons(self):
        """Test comparisons with negative Z-numbers."""
        z_neg1 = Znum(A=[-4, -3, -2, -1], B=[0.1, 0.2, 0.3, 0.4])
        z_neg2 = Znum(A=[-2, -1, 0, 1], B=[0.15, 0.25, 0.35, 0.45])
        assert not (z_neg1 > z_neg2)
        assert z_neg1 < z_neg2

    def test_negative_vs_positive(self):
        """Test comparison between negative and positive Z-numbers."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z_neg1 = Znum(A=[-4, -3, -2, -1], B=[0.1, 0.2, 0.3, 0.4])
        z_neg2 = Znum(A=[-2, -1, 0, 1], B=[0.15, 0.25, 0.35, 0.45])

        assert not (z_neg2 > z1)
        assert z_neg1 < z1

    def test_comparison_consistency(self):
        """Test that comparisons are consistent (if a > b, then b < a)."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])
        z3 = Znum(A=[0.5, 1, 1.5, 2], B=[0.05, 0.1, 0.15, 0.2])

        # If z2 > z1, then z1 < z2
        assert (z2 > z1) == (z1 < z2)
        # If z1 > z3, then z3 < z1
        assert (z1 > z3) == (z3 < z1)
        # If z2 > z3, then z3 < z2
        assert (z2 > z3) == (z3 < z2)

    def test_equality_consistency(self):
        """Test that equality is reflexive and symmetric."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z4 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])

        # Reflexive
        assert z1 == z1
        # Symmetric
        assert (z1 == z4) == (z4 == z1)


class TestZnumSortingWithComparisons:
    """Tests to verify sorting works correctly with Z-number comparisons."""

    def test_sort_znums_ascending(self):
        """Test that Z-numbers can be sorted in ascending order."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])
        z3 = Znum(A=[0.5, 1, 1.5, 2], B=[0.05, 0.1, 0.15, 0.2])

        znums = [z2, z1, z3]
        sorted_znums = sorted(znums)

        # Should be sorted as z3 < z1 < z2
        assert sorted_znums[0] == z3
        assert sorted_znums[1] == z1
        assert sorted_znums[2] == z2

    def test_sort_znums_descending(self):
        """Test that Z-numbers can be sorted in descending order."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])
        z3 = Znum(A=[0.5, 1, 1.5, 2], B=[0.05, 0.1, 0.15, 0.2])

        znums = [z3, z1, z2]
        sorted_znums = sorted(znums, reverse=True)

        # Should be sorted as z2 > z1 > z3
        assert sorted_znums[0] == z2
        assert sorted_znums[1] == z1
        assert sorted_znums[2] == z3

    def test_max_and_min(self):
        """Test that max() and min() work correctly with Z-numbers."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])
        z3 = Znum(A=[0.5, 1, 1.5, 2], B=[0.05, 0.1, 0.15, 0.2])

        znums = [z1, z2, z3]

        assert max(znums) == z2
        assert min(znums) == z3


class TestZnumLargeValueComparisons:
    """Tests for comparisons with large values."""

    def test_large_value_comparison(self):
        """Test comparison with large Z-number values."""
        z_large1 = Znum(A=[100, 200, 300, 400], B=[0.1, 0.2, 0.3, 0.4])
        z_large2 = Znum(A=[50, 100, 150, 200], B=[0.2, 0.3, 0.4, 0.5])

        assert z_large1 > z_large2
        assert z_large2 < z_large1

    def test_fractional_value_comparison(self):
        """Test comparison with fractional Z-number values."""
        z_frac1 = Znum(A=[0.1, 0.2, 0.3, 0.4], B=[0.1, 0.2, 0.3, 0.4])
        z_frac2 = Znum(A=[0.2, 0.3, 0.4, 0.5], B=[0.2, 0.3, 0.4, 0.5])

        assert z_frac2 > z_frac1
        assert z_frac1 < z_frac2


class TestZnumComparisonEdgeCases:
    """Tests for edge cases in comparisons."""

    def test_compare_result_of_operations(self):
        """Test comparing results of arithmetic operations."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])

        result1 = z1 + z2  # A: [3, 5, 7, 9]
        result2 = z1 * 2   # A: [2, 4, 6, 8]

        assert result1 > result2

    def test_compare_after_scalar_multiplication(self):
        """Test that z * 2 > z."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z1_doubled = z1 * 2

        assert z1_doubled > z1

    def test_compare_after_power(self):
        """Test comparison after power operation."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z1_squared = z1 ** 2

        assert z1_squared > z1
