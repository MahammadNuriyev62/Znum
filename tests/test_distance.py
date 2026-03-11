"""
Tests for distance functions (dist.py).

Covers: Simple distance, Hellinger distance components, formula correctness,
and ideal Z-number construction.
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from znum import Znum
from znum.dist import Dist
from znum.math_ops import Math


class TestSimpleDistance:
    """Tests for Dist.Simple.calculate."""

    def test_distance_to_self_center(self):
        """Distance from [1,2,3,4] to its center-ish value."""
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        d = Dist.Simple.calculate(z, 0)
        # sum(|0 - p| for p in [1,2,3,4,0.1,0.2,0.3,0.4]) * 0.5
        expected = sum(abs(0 - p) for p in [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]) * 0.5
        assert d == pytest.approx(expected)

    def test_distance_to_zero(self):
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        d = Dist.Simple.calculate(z, 0)
        assert d > 0

    def test_distance_symmetry_property(self):
        """Distance uses A + B (numpy element-wise sum), not concatenation."""
        z = Znum(A=[0, 0, 0, 0], B=[0.5, 0.5, 0.5, 0.5])
        d_pos = Dist.Simple.calculate(z, 5)
        d_neg = Dist.Simple.calculate(z, -5)
        # A + B = [0.5, 0.5, 0.5, 0.5]. sum(|5-0.5|)*0.5 = 9.0, sum(|-5-0.5|)*0.5 = 11.0
        assert d_pos == pytest.approx(9.0)
        assert d_neg == pytest.approx(11.0)

    def test_distance_increases_with_distance(self):
        z = Znum(A=[5, 5, 5, 5], B=[0.5, 0.5, 0.5, 0.5])
        d1 = Dist.Simple.calculate(z, 5)
        d2 = Dist.Simple.calculate(z, 10)
        d3 = Dist.Simple.calculate(z, 100)
        assert d1 < d2 < d3

    def test_crisp_at_same_value(self):
        """Distance of crisp(v) to v: A+B=[6,6,6,6], dist = sum(|5-6|)*0.5 = 2."""
        z = Znum.crisp(5)
        d = Dist.Simple.calculate(z, 5)
        # A=[5,5,5,5], B=[1,1,1,1], so A+B=[6,6,6,6]
        assert d == pytest.approx(sum(abs(5 - 6) for _ in range(4)) * 0.5)


class TestHellingerFormula:
    """Tests for Dist.Hellinger._formula_hellinger."""

    def test_identical_distributions(self):
        """Hellinger distance between identical distributions is 0."""
        P = [0.25, 0.25, 0.25, 0.25]
        Q = [0.25, 0.25, 0.25, 0.25]
        assert Dist.Hellinger._formula_hellinger(P, Q) == pytest.approx(0.0)

    def test_non_negative(self):
        P = [0.1, 0.2, 0.3, 0.4]
        Q = [0.4, 0.3, 0.2, 0.1]
        assert Dist.Hellinger._formula_hellinger(P, Q) >= 0

    def test_symmetric(self):
        P = [0.1, 0.2, 0.3, 0.4]
        Q = [0.4, 0.3, 0.2, 0.1]
        assert Dist.Hellinger._formula_hellinger(P, Q) == pytest.approx(
            Dist.Hellinger._formula_hellinger(Q, P)
        )

    def test_known_value(self):
        """Hand-computed Hellinger distance."""
        P = [1.0, 0.0]
        Q = [0.0, 1.0]
        # sqrt((1-0)^2 + (0-1)^2) / sqrt(2) = sqrt(2) / sqrt(2) = 1
        assert Dist.Hellinger._formula_hellinger(P, Q) == pytest.approx(1.0)

    def test_all_zeros(self):
        P = [0, 0, 0, 0]
        Q = [0, 0, 0, 0]
        assert Dist.Hellinger._formula_hellinger(P, Q) == pytest.approx(0.0)


class TestFormulaQ:
    """Tests for Dist.Hellinger._formula_q."""

    def test_identical_midpoints(self):
        """Same midpoints → distance 0."""
        assert Dist.Hellinger._formula_q(1, 1, 3, 3) == pytest.approx(0.0)

    def test_known_value(self):
        """|(1+3)/2 - (2+4)/2| = |2 - 3| = 1."""
        assert Dist.Hellinger._formula_q(1, 2, 3, 4) == pytest.approx(1.0)

    def test_symmetric(self):
        d1 = Dist.Hellinger._formula_q(1, 5, 3, 7)
        d2 = Dist.Hellinger._formula_q(5, 1, 7, 3)
        assert d1 == pytest.approx(d2)

    def test_non_negative(self):
        assert Dist.Hellinger._formula_q(10, 1, 2, 20) >= 0


class TestHellingerCalculate:
    """Integration tests for the full Hellinger distance calculation."""

    def test_distance_to_self_is_zero(self):
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        d = Dist.Hellinger.calculate(z, z)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_non_negative(self):
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[5, 6, 7, 8], B=[0.5, 0.6, 0.7, 0.8])
        d = Dist.Hellinger.calculate(z1, z2)
        assert d >= 0

    def test_farther_znums_larger_distance(self):
        z_base = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        z_near = Znum(A=[2, 3, 4, 5], B=[0.5, 0.6, 0.7, 0.8])
        z_far = Znum(A=[10, 11, 12, 13], B=[0.5, 0.6, 0.7, 0.8])
        d_near = Dist.Hellinger.calculate(z_base, z_near)
        d_far = Dist.Hellinger.calculate(z_base, z_far)
        assert d_far > d_near

    def test_crisp_distance_to_self(self):
        z = Znum.crisp(5)
        d = Dist.Hellinger.calculate(z, z)
        assert d == pytest.approx(0.0, abs=1e-10)


class TestCalculateAB:
    """Tests for the A and B component calculation."""

    def test_identical_znums_give_zero(self):
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        result = Dist.Hellinger._calculate_ab(z, z)
        assert result["A"] == pytest.approx(0.0)
        assert result["B"] == pytest.approx(0.0)

    def test_returns_dict_with_A_and_B_keys(self):
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])
        result = Dist.Hellinger._calculate_ab(z1, z2)
        assert "A" in result
        assert "B" in result
        assert result["A"] >= 0
        assert result["B"] >= 0


class TestIdealFromZnum:
    """Tests for Dist.Hellinger.get_ideal_from_znum."""

    def test_ideal_has_correct_A(self):
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        ideal = Dist.Hellinger.get_ideal_from_znum(z, value=0)
        assert_array_almost_equal(ideal.A, [0, 0, 0, 0])

    def test_ideal_has_near_zero_B(self):
        """Ideal with value=0 has B~0, but epsilon is added for LP stability."""
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        ideal = Dist.Hellinger.get_ideal_from_znum(z, value=0)
        # B should be very close to zero (only epsilon added)
        assert all(ideal.B[i] < 1e-4 for i in range(4))

    def test_ideal_with_nonzero_value(self):
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        ideal = Dist.Hellinger.get_ideal_from_znum(z, value=1)
        assert_array_almost_equal(ideal.A, [1, 1, 1, 1])
        assert_array_almost_equal(ideal.B, [1, 1, 1, 1])

    def test_ideal_has_same_dimension(self):
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        ideal = Dist.Hellinger.get_ideal_from_znum(z)
        assert ideal.dimension == z.dimension

    def test_ideal_intermediate_sizes_match(self):
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        ideal = Dist.Hellinger.get_ideal_from_znum(z)
        assert len(ideal.A_int["value"]) == len(z.A_int["value"])
        assert len(ideal.B_int["value"]) == len(z.B_int["value"])

    def test_ideal_membership_is_triangular(self):
        """Ideal membership should be symmetric triangular."""
        z = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        ideal = Dist.Hellinger.get_ideal_from_znum(z)
        memb = ideal.A_int["memb"]
        # Should peak at 1.0
        assert max(memb) == pytest.approx(1.0)
        # Should be symmetric
        n = len(memb)
        for i in range(n // 2):
            assert memb[i] == pytest.approx(memb[n - 1 - i])


class TestDefaultMembership:
    """Tests for Math.get_default_membership."""

    def test_even_size(self):
        result = Math.get_default_membership(4)
        assert len(result) == 4
        assert result[0] == pytest.approx(0.0)
        assert max(result) == pytest.approx(1.0)

    def test_odd_size(self):
        result = Math.get_default_membership(5)
        assert len(result) == 5
        assert max(result) == pytest.approx(1.0)

    def test_symmetric(self):
        for size in [4, 5, 6, 7, 10]:
            result = Math.get_default_membership(size)
            for i in range(len(result) // 2):
                assert result[i] == pytest.approx(result[len(result) - 1 - i])

    def test_peaks_at_one(self):
        for size in [4, 5, 6, 10]:
            result = Math.get_default_membership(size)
            assert max(result) == pytest.approx(1.0)

    def test_starts_at_zero(self):
        for size in [4, 5, 6, 10]:
            result = Math.get_default_membership(size)
            assert result[0] == pytest.approx(0.0)
            assert result[-1] == pytest.approx(0.0)
