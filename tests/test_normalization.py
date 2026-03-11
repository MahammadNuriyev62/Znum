"""
Tests for MCDMUtils normalization functions (utils.py).

Covers: normalize_benefit, normalize_cost, normalize_weight, and error paths.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from znum import Znum
from znum.utils import MCDMUtils


class TestNormalizeBenefit:
    """Tests for MCDMUtils.normalize_benefit."""

    def test_basic_normalization(self):
        """All A values should be divided by the global max."""
        z1 = Znum(A=[2, 4, 6, 8], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        MCDMUtils.normalize_benefit([z1, z2])
        # Max A is 8
        assert_array_almost_equal(z1.A, [2 / 8, 4 / 8, 6 / 8, 8 / 8])
        assert_array_almost_equal(z2.A, [1 / 8, 2 / 8, 3 / 8, 4 / 8])

    def test_max_becomes_one(self):
        """The maximum A value should become 1.0 after normalization."""
        z1 = Znum(A=[5, 6, 7, 10], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        MCDMUtils.normalize_benefit([z1, z2])
        all_a = list(z1.A) + list(z2.A)
        assert max(all_a) == pytest.approx(1.0)

    def test_preserves_B(self):
        """Normalization should not modify B values."""
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        original_B = z.B.copy()
        MCDMUtils.normalize_benefit([z])
        assert_array_almost_equal(z.B, original_B)

    def test_all_zero_raises(self):
        """All-zero A values should raise ValueError."""
        z = Znum(A=[0, 0, 0, 0], B=[0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="all A values are zero"):
            MCDMUtils.normalize_benefit([z])


class TestNormalizeCost:
    """Tests for MCDMUtils.normalize_cost."""

    def test_basic_normalization(self):
        """Cost normalization: min(A) / each A, reversed."""
        z1 = Znum(A=[2, 4, 6, 8], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        MCDMUtils.normalize_cost([z1, z2])
        # Min A is 1. For z1: reversed([1/2, 1/4, 1/6, 1/8]) = [1/8, 1/6, 1/4, 1/2]
        assert_array_almost_equal(z1.A, [1 / 8, 1 / 6, 1 / 4, 1 / 2])
        # For z2: reversed([1/1, 1/2, 1/3, 1/4]) = [1/4, 1/3, 1/2, 1/1]
        assert_array_almost_equal(z2.A, [1 / 4, 1 / 3, 1 / 2, 1 / 1])

    def test_preserves_B(self):
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        original_B = z.B.copy()
        MCDMUtils.normalize_cost([z])
        assert_array_almost_equal(z.B, original_B)

    def test_zero_in_A_raises(self):
        """A containing zero should raise ValueError (division by zero)."""
        z = Znum(A=[0, 1, 2, 3], B=[0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="A contains zero"):
            MCDMUtils.normalize_cost([z])

    def test_result_is_non_decreasing(self):
        """After cost normalization, A should still be non-decreasing."""
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        MCDMUtils.normalize_cost([z])
        assert all(z.A[i] <= z.A[i + 1] for i in range(len(z.A) - 1))


class TestNormalize:
    """Tests for the normalize dispatcher."""

    def test_benefit_routing(self):
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        MCDMUtils.normalize([z], MCDMUtils.CriteriaType.BENEFIT)
        # After benefit normalization, max A should be 1.0
        assert max(z.A) == pytest.approx(1.0)

    def test_cost_routing(self):
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        MCDMUtils.normalize([z], MCDMUtils.CriteriaType.COST)
        # After cost normalization, should be non-decreasing
        assert all(z.A[i] <= z.A[i + 1] for i in range(len(z.A) - 1))

    def test_unknown_type_falls_back_to_cost(self):
        """Unknown criteria type should default to cost normalization."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        MCDMUtils.normalize([z1], "UNKNOWN")
        MCDMUtils.normalize([z2], MCDMUtils.CriteriaType.COST)
        assert_array_almost_equal(z1.A, z2.A)
