import pytest
import numpy as np
from znum.Znum import Znum
from znum.Dist import Dist


# =============================================================================
# Dist.Simple Tests
# =============================================================================

class TestDistSimple:
    """Test Dist.Simple class."""

    def test_coef_value(self):
        """Test _COEF constant."""
        assert Dist.Simple._COEF == 0.5

    def test_calculate_returns_number(self):
        """Test that calculate returns a number."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = Dist.Simple.calculate(z, 0)

        assert isinstance(result, (int, float))

    def test_calculate_with_n_0(self):
        """Test calculate with n=0."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = Dist.Simple.calculate(z, 0)

        # Note: z.A + z.B is numpy array addition, not concatenation
        # z.A + z.B = [1.1, 2.2, 3.3, 4.4]
        # sum(abs(0 - p) for p in [1.1, 2.2, 3.3, 4.4]) = 1.1 + 2.2 + 3.3 + 4.4 = 11.0
        # 11.0 * 0.5 = 5.5
        expected = 5.5
        assert result == pytest.approx(expected)

    def test_calculate_with_n_1(self):
        """Test calculate with n=1."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = Dist.Simple.calculate(z, 1)

        # Note: z.A + z.B is numpy array addition, not concatenation
        # z.A + z.B = [1.1, 2.2, 3.3, 4.4]
        # sum(abs(1 - p) for p in [1.1, 2.2, 3.3, 4.4]) = 0.1 + 1.2 + 2.3 + 3.4 = 7.0
        # 7.0 * 0.5 = 3.5
        expected = 3.5
        assert result == pytest.approx(expected)

    def test_calculate_distance_to_self_values(self):
        """Test calculate when n matches a value in A."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = Dist.Simple.calculate(z, 2)  # 2 is in A

        assert result >= 0

    def test_calculate_symmetric(self):
        """Test that distance is non-negative."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result_0 = Dist.Simple.calculate(z, 0)
        result_1 = Dist.Simple.calculate(z, 1)

        assert result_0 >= 0
        assert result_1 >= 0

    def test_calculate_with_negative_A(self):
        """Test calculate with negative A values."""
        z = Znum([-4, -3, -2, -1], [0.1, 0.2, 0.3, 0.4])
        result = Dist.Simple.calculate(z, 0)

        assert result >= 0

    def test_calculate_increases_with_distance(self):
        """Test that distance increases as n moves away from Znum values."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result_2 = Dist.Simple.calculate(z, 2)
        result_10 = Dist.Simple.calculate(z, 10)

        # Distance to 10 should be greater than distance to 2
        assert result_10 > result_2


# =============================================================================
# Dist.Hellinger Coefficient Tests
# =============================================================================

class TestDistHellingerCoefficients:
    """Test Dist.Hellinger coefficient constants."""

    def test_coef_A(self):
        assert Dist.Hellinger._COEF_A == 0.5

    def test_coef_B(self):
        assert Dist.Hellinger._COEF_B == 0.25

    def test_coef_H(self):
        assert Dist.Hellinger._COEF_H == 0.25

    def test_coefficients_sum_to_1(self):
        total = Dist.Hellinger._COEF_A + Dist.Hellinger._COEF_B + Dist.Hellinger._COEF_H
        assert total == 1.0


# =============================================================================
# Dist.Hellinger._formula_hellinger Tests
# =============================================================================

class TestDistHellingerFormulaHellinger:
    """Test Dist.Hellinger._formula_hellinger static method."""

    def test_identical_distributions(self):
        """Test Hellinger distance for identical distributions."""
        P = [0.25, 0.25, 0.25, 0.25]
        Q = [0.25, 0.25, 0.25, 0.25]

        result = Dist.Hellinger._formula_hellinger(P, Q)

        assert result == pytest.approx(0.0)

    def test_returns_non_negative(self):
        """Test that Hellinger distance is non-negative."""
        P = [0.1, 0.2, 0.3, 0.4]
        Q = [0.4, 0.3, 0.2, 0.1]

        result = Dist.Hellinger._formula_hellinger(P, Q)

        assert result >= 0

    def test_returns_value_in_range(self):
        """Test that Hellinger distance is in valid range [0, 1]."""
        P = [0.1, 0.2, 0.3, 0.4]
        Q = [0.4, 0.3, 0.2, 0.1]

        result = Dist.Hellinger._formula_hellinger(P, Q)

        assert 0 <= result <= 1


# =============================================================================
# Dist.Hellinger._formula_q Tests
# =============================================================================

class TestDistHellingerFormulaQ:
    """Test Dist.Hellinger._formula_q static method."""

    def test_identical_values(self):
        """Test formula_q when all values are equal."""
        result = Dist.Hellinger._formula_q(1, 1, 1, 1)

        assert result == 0

    def test_returns_non_negative(self):
        """Test that formula_q returns non-negative value."""
        result = Dist.Hellinger._formula_q(1, 2, 3, 4)

        assert result >= 0

    def test_calculation(self):
        """Test formula_q calculation."""
        # Q = abs((znum1_half1_q + znum1_half2_q)/2 - (znum2_half1_q + znum2_half2_q)/2)
        result = Dist.Hellinger._formula_q(2, 4, 6, 8)

        # ((2 + 6) / 2) - ((4 + 8) / 2) = 4 - 6 = -2, abs = 2
        expected = abs((2 + 6) / 2 - (4 + 8) / 2)
        assert result == pytest.approx(expected)


# =============================================================================
# Dist.Hellinger.calculate Tests
# =============================================================================

class TestDistHellingerCalculate:
    """Test Dist.Hellinger.calculate static method."""

    def test_returns_number(self):
        """Test that calculate returns a number."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        result = Dist.Hellinger.calculate(z1, z2)

        assert isinstance(result, (int, float))

    def test_identical_znums(self):
        """Test calculate with identical Znums."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = Dist.Hellinger.calculate(z1, z2)

        # Distance should be close to 0 for identical Znums
        assert result == pytest.approx(0.0, abs=0.01)

    def test_different_znums(self):
        """Test calculate with different Znums."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])

        result = Dist.Hellinger.calculate(z1, z2)

        # Distance should be positive
        assert result > 0

    def test_returns_non_negative(self):
        """Test that calculate returns non-negative value."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        result = Dist.Hellinger.calculate(z1, z2)

        assert result >= 0


# =============================================================================
# Dist.Hellinger.get_ideal_from_znum Tests
# =============================================================================

class TestDistHellingerGetIdealFromZnum:
    """Test Dist.Hellinger.get_ideal_from_znum static method."""

    def test_returns_znum(self):
        """Test that get_ideal_from_znum returns a Znum."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = Dist.Hellinger.get_ideal_from_znum(z, 1)

        assert isinstance(result, Znum)

    def test_default_value_0(self):
        """Test get_ideal_from_znum with default value 0."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = Dist.Hellinger.get_ideal_from_znum(z, 0)

        # A should all be 0
        assert list(result.A) == [0, 0, 0, 0]
        # B values are approximately 0 (may have tiny floating point offsets)
        assert list(result.B) == pytest.approx([0, 0, 0, 0], abs=1e-5)

    def test_value_1(self):
        """Test get_ideal_from_znum with value 1."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = Dist.Hellinger.get_ideal_from_znum(z, 1)

        # A and B should all be 1
        assert list(result.A) == [1, 1, 1, 1]
        assert list(result.B) == [1, 1, 1, 1]

    def test_preserves_dimension(self):
        """Test that ideal Znum has same dimension as original."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = Dist.Hellinger.get_ideal_from_znum(z, 0)

        assert result.dimension == z.dimension

    def test_has_A_int_and_B_int(self):
        """Test that ideal Znum has A_int and B_int."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = Dist.Hellinger.get_ideal_from_znum(z, 1)

        assert result.A_int is not None
        assert result.B_int is not None


# =============================================================================
# Dist.Hellinger._calculate_H Tests
# =============================================================================

class TestDistHellingerCalculateH:
    """Test Dist.Hellinger._calculate_H static method."""

    def test_returns_number(self):
        """Test that _calculate_H returns a number."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        result = Dist.Hellinger._calculate_H(z1, z2)

        assert isinstance(result, (int, float, np.floating))

    def test_identical_znums_returns_zero(self):
        """Test that _calculate_H returns 0 for identical Znums."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = Dist.Hellinger._calculate_H(z1, z2)

        assert result == pytest.approx(0.0, abs=0.01)


# =============================================================================
# Dist.Hellinger._calculate_AB Tests
# =============================================================================

class TestDistHellingerCalculateAB:
    """Test Dist.Hellinger._calculate_AB static method."""

    def test_returns_dict(self):
        """Test that _calculate_AB returns a dictionary."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        result = Dist.Hellinger._calculate_AB(z1, z2)

        assert isinstance(result, dict)
        assert "A" in result
        assert "B" in result

    def test_identical_znums(self):
        """Test _calculate_AB for identical Znums."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = Dist.Hellinger._calculate_AB(z1, z2)

        # For identical Znums, A and B distances should be 0
        assert result["A"] == pytest.approx(0.0)
        assert result["B"] == pytest.approx(0.0)


# =============================================================================
# Integration Tests
# =============================================================================

class TestDistIntegration:
    """Integration tests for Dist module."""

    def test_simple_vs_hellinger(self):
        """Test that both distance methods work and return valid results."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        simple_dist = Dist.Simple.calculate(z1, 0)
        hellinger_dist = Dist.Hellinger.calculate(z1, z2)

        assert simple_dist >= 0
        assert hellinger_dist >= 0

    def test_distance_to_ideal(self):
        """Test distance calculation to ideal Znum."""
        z = Znum([0.5, 0.6, 0.7, 0.8], [0.4, 0.5, 0.6, 0.7])
        ideal_1 = Dist.Hellinger.get_ideal_from_znum(z, 1)
        ideal_0 = Dist.Hellinger.get_ideal_from_znum(z, 0)

        dist_to_1 = Dist.Hellinger.calculate(z, ideal_1)
        dist_to_0 = Dist.Hellinger.calculate(z, ideal_0)

        # Both distances should be valid
        assert dist_to_1 >= 0
        assert dist_to_0 >= 0

    def test_multiple_znums(self):
        """Test distance calculations for multiple Znum pairs."""
        znums = [
            Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
            Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5]),
            Znum([3, 4, 5, 6], [0.3, 0.4, 0.5, 0.6]),
        ]

        for i, z1 in enumerate(znums):
            for j, z2 in enumerate(znums):
                result = Dist.Hellinger.calculate(z1, z2)
                assert result >= 0

                if i == j:
                    # Same Znum should have distance close to 0
                    assert result == pytest.approx(0.0, abs=0.01)
