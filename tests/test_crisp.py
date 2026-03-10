"""
Comprehensive tests for Znum.crisp() factory method.

Tests cover construction, properties, arithmetic, comparison, serialization,
and edge cases for crisp (exact) Z-numbers.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from znum import Znum


class TestCrispConstruction:
    """Tests for basic crisp Z-number creation."""

    def test_crisp_positive_integer(self):
        z = Znum.crisp(5)
        assert_array_equal(z.A, [5.0, 5.0, 5.0, 5.0])
        assert_array_equal(z.B, [1.0, 1.0, 1.0, 1.0])

    def test_crisp_zero(self):
        z = Znum.crisp(0)
        assert_array_equal(z.A, [0.0, 0.0, 0.0, 0.0])
        assert_array_equal(z.B, [1.0, 1.0, 1.0, 1.0])

    def test_crisp_negative(self):
        z = Znum.crisp(-3)
        assert_array_equal(z.A, [-3.0, -3.0, -3.0, -3.0])
        assert_array_equal(z.B, [1.0, 1.0, 1.0, 1.0])

    def test_crisp_float(self):
        z = Znum.crisp(3.14)
        assert_array_almost_equal(z.A, [3.14, 3.14, 3.14, 3.14])
        assert_array_equal(z.B, [1.0, 1.0, 1.0, 1.0])

    def test_crisp_very_small(self):
        z = Znum.crisp(1e-10)
        assert_array_almost_equal(z.A, [1e-10] * 4)
        assert_array_equal(z.B, [1.0, 1.0, 1.0, 1.0])

    def test_crisp_very_large(self):
        z = Znum.crisp(1e10)
        assert_array_almost_equal(z.A, [1e10] * 4)
        assert_array_equal(z.B, [1.0, 1.0, 1.0, 1.0])

    def test_crisp_negative_float(self):
        z = Znum.crisp(-2.718)
        assert_array_almost_equal(z.A, [-2.718] * 4)
        assert_array_equal(z.B, [1.0, 1.0, 1.0, 1.0])


class TestCrispProperties:
    """Tests for properties of crisp Z-numbers."""

    def test_dimension_is_4(self):
        z = Znum.crisp(5)
        assert z.dimension == 4

    def test_is_trapezoid(self):
        z = Znum.crisp(5)
        assert z.is_trapezoid is True

    def test_is_even(self):
        z = Znum.crisp(5)
        assert z.is_even is True

    def test_membership_all_ones(self):
        """Crisp values should have C = [1, 1, 1, 1] (all-equal A triggers special case)."""
        z = Znum.crisp(5)
        assert_array_equal(z.C, [1.0, 1.0, 1.0, 1.0])

    def test_membership_all_ones_for_zero(self):
        z = Znum.crisp(0)
        assert_array_equal(z.C, [1.0, 1.0, 1.0, 1.0])

    def test_membership_all_ones_for_negative(self):
        z = Znum.crisp(-7)
        assert_array_equal(z.C, [1.0, 1.0, 1.0, 1.0])

    def test_is_instance_of_znum(self):
        z = Znum.crisp(5)
        assert isinstance(z, Znum)


class TestCrispSerialization:
    """Tests for serialization methods on crisp Z-numbers."""

    def test_to_json(self):
        z = Znum.crisp(5)
        j = z.to_json()
        assert j == {"A": [5.0, 5.0, 5.0, 5.0], "B": [1.0, 1.0, 1.0, 1.0]}

    def test_to_array(self):
        z = Znum.crisp(3)
        arr = z.to_array()
        assert_array_equal(arr, [3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0])

    def test_str_representation(self):
        z = Znum.crisp(5)
        assert str(z) == "Znum(A=[5.0, 5.0, 5.0, 5.0], B=[1.0, 1.0, 1.0, 1.0])"

    def test_repr(self):
        z = Znum.crisp(5)
        assert repr(z) == str(z)

    def test_copy_is_independent(self):
        z = Znum.crisp(5)
        z_copy = z.copy()
        z_copy.A = [10, 10, 10, 10]
        assert_array_equal(z.A, [5.0, 5.0, 5.0, 5.0])


class TestCrispArithmetic:
    """Tests for arithmetic operations involving crisp Z-numbers."""

    def test_crisp_add_crisp(self):
        """Adding two crisp values should give a crisp-like result."""
        z1 = Znum.crisp(3)
        z2 = Znum.crisp(4)
        result = z1 + z2
        # A should be [7, 7, 7, 7]
        assert_array_almost_equal(result.A, [7.0, 7.0, 7.0, 7.0])

    def test_crisp_sub_crisp(self):
        z1 = Znum.crisp(10)
        z2 = Znum.crisp(3)
        result = z1 - z2
        assert_array_almost_equal(result.A, [7.0, 7.0, 7.0, 7.0])

    def test_crisp_mul_crisp(self):
        z1 = Znum.crisp(3)
        z2 = Znum.crisp(4)
        result = z1 * z2
        assert_array_almost_equal(result.A, [12.0, 12.0, 12.0, 12.0])

    def test_crisp_div_crisp(self):
        z1 = Znum.crisp(12)
        z2 = Znum.crisp(4)
        result = z1 / z2
        assert_array_almost_equal(result.A, [3.0, 3.0, 3.0, 3.0])

    def test_crisp_power(self):
        z = Znum.crisp(3)
        result = z ** 2
        assert_array_almost_equal(result.A, [9.0, 9.0, 9.0, 9.0])
        assert_array_equal(result.B, [1.0, 1.0, 1.0, 1.0])

    def test_crisp_scalar_multiply(self):
        z = Znum.crisp(5)
        result = z * 3
        assert_array_almost_equal(result.A, [15.0, 15.0, 15.0, 15.0])
        assert_array_equal(result.B, [1.0, 1.0, 1.0, 1.0])

    def test_crisp_add_zero_int(self):
        z = Znum.crisp(5)
        result = z + 0
        assert_array_equal(result.A, z.A)
        assert_array_equal(result.B, z.B)

    def test_crisp_radd_zero(self):
        z = Znum.crisp(5)
        result = 0 + z
        assert_array_equal(result.A, z.A)

    def test_sum_of_crisp_list(self):
        """Python sum() over a list of crisp Z-numbers."""
        znums = [Znum.crisp(i) for i in range(1, 4)]  # 1, 2, 3
        result = sum(znums)
        assert_array_almost_equal(result.A, [6.0, 6.0, 6.0, 6.0])

    def test_crisp_add_fuzzy(self):
        """A crisp Z-number added to a regular fuzzy Z-number."""
        z_crisp = Znum.crisp(5)
        z_fuzzy = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        result = z_crisp + z_fuzzy
        assert_array_almost_equal(result.A, [6.0, 7.0, 8.0, 9.0])

    def test_crisp_sub_fuzzy(self):
        z_crisp = Znum.crisp(10)
        z_fuzzy = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        result = z_crisp - z_fuzzy
        assert_array_almost_equal(result.A, [6.0, 7.0, 8.0, 9.0])

    def test_crisp_mul_fuzzy(self):
        z_crisp = Znum.crisp(2)
        z_fuzzy = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        result = z_crisp * z_fuzzy
        assert_array_almost_equal(result.A, [2.0, 4.0, 6.0, 8.0])

    def test_crisp_zero_add_fuzzy(self):
        """Adding crisp(0) to a fuzzy Z-number."""
        z_zero = Znum.crisp(0)
        z_fuzzy = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        result = z_zero + z_fuzzy
        assert_array_almost_equal(result.A, [1.0, 2.0, 3.0, 4.0])

    def test_crisp_one_mul_fuzzy(self):
        """Multiplying crisp(1) by a fuzzy Z-number should preserve A values."""
        z_one = Znum.crisp(1)
        z_fuzzy = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        result = z_one * z_fuzzy
        assert_array_almost_equal(result.A, [1.0, 2.0, 3.0, 4.0])


class TestCrispComparison:
    """Tests for comparison operations on crisp Z-numbers."""

    def test_crisp_equal(self):
        z1 = Znum.crisp(5)
        z2 = Znum.crisp(5)
        assert z1 == z2

    def test_crisp_not_equal(self):
        z1 = Znum.crisp(5)
        z2 = Znum.crisp(3)
        assert not (z1 == z2)

    def test_crisp_greater_than(self):
        z1 = Znum.crisp(5)
        z2 = Znum.crisp(3)
        assert z1 > z2

    def test_crisp_less_than(self):
        z1 = Znum.crisp(3)
        z2 = Znum.crisp(5)
        assert z1 < z2

    def test_crisp_greater_equal(self):
        z1 = Znum.crisp(5)
        z2 = Znum.crisp(5)
        assert z1 >= z2

    def test_crisp_less_equal(self):
        z1 = Znum.crisp(5)
        z2 = Znum.crisp(5)
        assert z1 <= z2

    def test_crisp_gt_fuzzy(self):
        """Crisp 10 should be greater than fuzzy [1, 2, 3, 4]."""
        z_crisp = Znum.crisp(10)
        z_fuzzy = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        assert z_crisp > z_fuzzy

    def test_crisp_lt_fuzzy(self):
        """Crisp 0 should be less than fuzzy [1, 2, 3, 4] with high reliability."""
        z_crisp = Znum.crisp(0)
        z_fuzzy = Znum(A=[1, 2, 3, 4], B=[0.7, 0.8, 0.9, 1.0])
        assert z_crisp < z_fuzzy

    def test_crisp_sorting(self):
        """Sorting a list of crisp Z-numbers should match natural ordering."""
        values = [5, 1, 3, 2, 4]
        znums = [Znum.crisp(v) for v in values]
        sorted_znums = sorted(znums)
        sorted_values = [z.A[0] for z in sorted_znums]
        assert sorted_values == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_crisp_chained_comparison(self):
        z1 = Znum.crisp(1)
        z2 = Znum.crisp(2)
        z3 = Znum.crisp(3)
        assert z1 < z2 < z3

    def test_crisp_negative_ordering(self):
        z1 = Znum.crisp(-5)
        z2 = Znum.crisp(-1)
        z3 = Znum.crisp(0)
        assert z1 < z2 < z3

    def test_crisp_not_equal_to_non_znum(self):
        z = Znum.crisp(5)
        assert z != 5
        assert z != "5"
        assert z != [5, 5, 5, 5]


class TestCrispEdgeCases:
    """Edge case and robustness tests."""

    def test_crisp_no_b_warning(self):
        """B = [1,1,1,1] should NOT trigger the near-zero B warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            z = Znum.crisp(5)  # Should not raise

    def test_crisp_returns_new_instance(self):
        """Each call should return a new independent instance."""
        z1 = Znum.crisp(5)
        z2 = Znum.crisp(5)
        assert z1 is not z2
        z1.A = [10, 10, 10, 10]
        assert_array_equal(z2.A, [5.0, 5.0, 5.0, 5.0])

    def test_crisp_a_int_computed(self):
        """Intermediate representations should be computed."""
        z = Znum.crisp(5)
        assert z.A_int is not None
        assert "value" in z.A_int
        assert "memb" in z.A_int

    def test_crisp_b_int_computed(self):
        z = Znum.crisp(5)
        assert z.B_int is not None
        assert "value" in z.B_int
        assert "memb" in z.B_int

    def test_multiple_crisp_arithmetic_chain(self):
        """Chain several operations: (crisp(2) + crisp(3)) * crisp(4)."""
        result = (Znum.crisp(2) + Znum.crisp(3)) * Znum.crisp(4)
        assert_array_almost_equal(result.A, [20.0, 20.0, 20.0, 20.0])

    def test_crisp_sub_self_is_zero(self):
        """Subtracting a crisp value from itself should give zero."""
        z = Znum.crisp(7)
        result = z - z
        assert_array_almost_equal(result.A, [0.0, 0.0, 0.0, 0.0])

    def test_crisp_div_self_is_one(self):
        """Dividing a crisp value by itself should give one."""
        z = Znum.crisp(7)
        result = z / z
        assert_array_almost_equal(result.A, [1.0, 1.0, 1.0, 1.0])
