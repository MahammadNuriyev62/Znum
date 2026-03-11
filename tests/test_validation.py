"""
Tests for input validation (valid.py) and custom exceptions.

Covers: invalid A/B detection, B boundary checks, validation decorators,
and the near-zero B epsilon adjustment in core.py.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from znum import Znum
from znum.dist import Dist
from znum.exceptions import (
    InvalidAPartOfZnumException,
    InvalidBPartOfZnumException,
    ZnumMustBeEvenException,
    ZnumsMustBeInSameDimensionException,
)


class TestInvalidA:
    """Validation must reject non-monotonic A values."""

    def test_decreasing_A(self):
        with pytest.raises(InvalidAPartOfZnumException):
            Znum(A=[4, 3, 2, 1], B=[0.1, 0.2, 0.3, 0.4])

    def test_non_monotonic_A_middle(self):
        with pytest.raises(InvalidAPartOfZnumException):
            Znum(A=[1, 5, 3, 4], B=[0.1, 0.2, 0.3, 0.4])

    def test_non_monotonic_A_last(self):
        with pytest.raises(InvalidAPartOfZnumException):
            Znum(A=[1, 2, 4, 3], B=[0.1, 0.2, 0.3, 0.4])

    def test_valid_equal_A_accepted(self):
        """Equal adjacent values (plateau) should be accepted."""
        z = Znum(A=[2, 2, 3, 3], B=[0.1, 0.2, 0.3, 0.4])
        assert_array_almost_equal(z.A, [2.0, 2.0, 3.0, 3.0])

    def test_all_equal_A_accepted(self):
        z = Znum(A=[5, 5, 5, 5], B=[0.1, 0.2, 0.3, 0.4])
        assert_array_almost_equal(z.A, [5.0, 5.0, 5.0, 5.0])


class TestInvalidB:
    """Validation must reject non-monotonic or out-of-range B values."""

    def test_decreasing_B(self):
        with pytest.raises(InvalidBPartOfZnumException):
            Znum(A=[1, 2, 3, 4], B=[0.9, 0.1, 0.5, 0.3])

    def test_non_monotonic_B_middle(self):
        with pytest.raises(InvalidBPartOfZnumException):
            Znum(A=[1, 2, 3, 4], B=[0.1, 0.5, 0.3, 0.4])

    def test_B_above_one(self):
        with pytest.raises(InvalidBPartOfZnumException):
            Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 1.5])

    def test_B_below_zero(self):
        with pytest.raises(InvalidBPartOfZnumException):
            Znum(A=[1, 2, 3, 4], B=[-0.1, 0.2, 0.3, 0.4])

    def test_B_all_zero(self):
        """B=[0,0,0,0] is non-decreasing and within [0,1] — should be accepted."""
        z = Znum(A=[1, 2, 3, 4], B=[0, 0, 0, 0])
        assert_array_almost_equal(z.B, [0, 0, 0, 0], decimal=4)

    def test_B_all_one(self):
        z = Znum(A=[1, 2, 3, 4], B=[1, 1, 1, 1])
        assert_array_almost_equal(z.B, [1, 1, 1, 1])

    def test_B_exactly_boundary(self):
        """B=[0, 0.5, 0.5, 1] should be valid."""
        z = Znum(A=[1, 2, 3, 4], B=[0, 0.5, 0.5, 1])
        assert_array_almost_equal(z.B, [0, 0.5, 0.5, 1])


class TestNearZeroBEpsilon:
    """The constructor adds epsilon to B when B[-1] < threshold."""

    def test_warning_emitted(self):
        with pytest.warns(UserWarning, match="small epsilon added"):
            Znum(A=[1, 2, 3, 4], B=[0.0001, 0.0002, 0.0003, 0.0004])

    def test_no_warning_for_normal_B(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])

    def test_epsilon_adjustment_applied(self):
        """B values should be slightly increased after epsilon adjustment."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = Znum(A=[1, 2, 3, 4], B=[0.0001, 0.0002, 0.0003, 0.0004])
        # Each B[i] gets += epsilon * (i+1), so B should be strictly > original
        assert all(z.B[i] > 0.0001 * (i + 1) for i in range(4))
        # B should still be monotonically non-decreasing
        assert all(z.B[i] <= z.B[i + 1] for i in range(3))


class TestValidationDecorators:
    """Tests for check_if_znums_are_even and check_if_znums_are_in_same_dimension."""

    def test_hellinger_rejects_odd_dimension(self):
        """Hellinger distance requires even-dimension Z-numbers."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        # Create a Z-number with odd dimension by bypassing normal construction
        z_odd = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z_odd._A = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ZnumMustBeEvenException):
            Dist.Hellinger.calculate(z1, z_odd)

    def test_hellinger_rejects_different_dimensions(self):
        """Hellinger distance requires same-dimension Z-numbers."""
        z4 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z6 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z6._A = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        with pytest.raises(ZnumsMustBeInSameDimensionException):
            Dist.Hellinger.calculate(z4, z6)
