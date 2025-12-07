import pytest
import numpy as np
from znum.Znum import Znum
from znum.Sort import Sort


# =============================================================================
# Sort.NXF_OPTIONS Tests
# =============================================================================

class TestSortNxfOptions:
    """Test Sort.NXF_OPTIONS constants."""

    def test_nbF_option(self):
        assert Sort.NXF_OPTIONS["nbF"] == "nbF"

    def test_neF_option(self):
        assert Sort.NXF_OPTIONS["neF"] == "neF"

    def test_nwF_option(self):
        assert Sort.NXF_OPTIONS["nwF"] == "nwF"

    def test_all_options_exist(self):
        assert len(Sort.NXF_OPTIONS) == 3


# =============================================================================
# Sort.NXF Tests
# =============================================================================

class TestSortNxf:
    """Test Sort.NXF dictionary."""

    def test_nbF_values(self):
        assert Sort.NXF["nbF"] == (-1, -1, -0.3, -0.1)

    def test_neF_values(self):
        assert Sort.NXF["neF"] == (-0.3, -0.1, 0.1, 0.3)

    def test_nwF_values(self):
        assert Sort.NXF["nwF"] == (0.1, 0.3, 1, 1)


# =============================================================================
# Sort.normalization Tests
# =============================================================================

class TestSortNormalization:
    """Test Sort.normalization static method."""

    def test_returns_two_lists(self):
        """Test that normalization returns two lists."""
        q1 = [1, 2, 3, 4]
        q2 = [5, 6, 7, 8]

        result = Sort.normalization(q1, q2)

        assert len(result) == 2
        assert len(result[0]) == 4
        assert len(result[1]) == 4

    def test_min_is_zero(self):
        """Test that minimum value normalizes to 0."""
        q1 = [1, 2, 3, 4]
        q2 = [5, 6, 7, 8]

        norm1, norm2 = Sort.normalization(q1, q2)

        assert norm1[0] == 0.0  # 1 is the min

    def test_max_is_one(self):
        """Test that maximum value normalizes to 1."""
        q1 = [1, 2, 3, 4]
        q2 = [5, 6, 7, 8]

        norm1, norm2 = Sort.normalization(q1, q2)

        assert norm2[3] == 1.0  # 8 is the max

    def test_all_same_values(self):
        """Test normalization when all values are the same."""
        q1 = [5, 5, 5, 5]
        q2 = [5, 5, 5, 5]

        norm1, norm2 = Sort.normalization(q1, q2)

        # Should return zeros to avoid division by zero
        assert norm1 == [0, 0, 0, 0]
        assert norm2 == [0, 0, 0, 0]

    def test_preserves_order(self):
        """Test that normalization preserves order."""
        q1 = [1, 3, 5, 7]
        q2 = [2, 4, 6, 8]

        norm1, norm2 = Sort.normalization(q1, q2)

        # Check order is preserved
        for i in range(len(norm1) - 1):
            assert norm1[i] <= norm1[i + 1]
            assert norm2[i] <= norm2[i + 1]

    def test_values_in_0_1_range(self):
        """Test that all normalized values are in [0, 1]."""
        q1 = [10, 20, 30, 40]
        q2 = [15, 25, 35, 45]

        norm1, norm2 = Sort.normalization(q1, q2)

        for val in norm1 + norm2:
            assert 0 <= val <= 1

    def test_negative_values(self):
        """Test normalization with negative values."""
        q1 = [-8, -4, -2, -1]
        q2 = [1, 2, 4, 8]

        norm1, norm2 = Sort.normalization(q1, q2)

        # -8 should be 0, 8 should be 1
        assert norm1[0] == 0.0
        assert norm2[3] == 1.0


# =============================================================================
# Sort.get_intermediate Tests
# =============================================================================

class TestSortGetIntermediate:
    """Test Sort.get_intermediate static method."""

    def test_returns_list(self):
        """Test that get_intermediate returns a list."""
        normQ1 = [0.0, 0.2, 0.4, 0.6]
        normQ2 = [0.1, 0.3, 0.5, 1.0]

        result = Sort.get_intermediate(normQ1, normQ2)

        assert isinstance(result, list)

    def test_correct_length(self):
        """Test that get_intermediate returns correct length."""
        normQ1 = [0.0, 0.2, 0.4, 0.6]
        normQ2 = [0.1, 0.3, 0.5, 1.0]

        result = Sort.get_intermediate(normQ1, normQ2)

        assert len(result) == 4

    def test_calculation(self):
        """Test get_intermediate calculation."""
        normQ1 = [0.0, 0.2, 0.4, 0.6]
        normQ2 = [0.1, 0.3, 0.5, 1.0]

        result = Sort.get_intermediate(normQ1, normQ2)

        # result[i] = normQ1[i] - normQ2[len-i-1]
        # result[0] = 0.0 - 1.0 = -1.0
        # result[1] = 0.2 - 0.5 = -0.3
        # result[2] = 0.4 - 0.3 = 0.1
        # result[3] = 0.6 - 0.1 = 0.5
        assert result[0] == pytest.approx(-1.0)
        assert result[1] == pytest.approx(-0.3)
        assert result[2] == pytest.approx(0.1)
        assert result[3] == pytest.approx(0.5)


# =============================================================================
# Sort.formula_nxF_Q_possibility Tests
# =============================================================================

class TestSortFormulaNxFQPossibility:
    """Test Sort.formula_nxF_Q_possibility static method."""

    def test_returns_0_for_disjoint(self):
        """Test that completely disjoint intervals return 0."""
        # When a2 < b1 and no overlap
        result = Sort.formula_nxF_Q_possibility(
            alpha_l=0.5, a1=0.0, a2=0.5, alpha_r=0.5,
            betta_l=0.5, b1=2.0, b2=2.5, betta_r=0.5
        )
        assert result == 0

    def test_returns_1_for_overlap(self):
        """Test that overlapping intervals return 1."""
        # When max(a1, b1) <= min(a2, b2)
        result = Sort.formula_nxF_Q_possibility(
            alpha_l=0.5, a1=0.0, a2=1.0, alpha_r=0.5,
            betta_l=0.5, b1=0.0, b2=1.0, betta_r=0.5
        )
        assert result == 1

    def test_returns_value_between_0_and_1(self):
        """Test partial overlap returns value between 0 and 1."""
        result = Sort.formula_nxF_Q_possibility(
            alpha_l=0.5, a1=0.5, a2=1.0, alpha_r=0.5,
            betta_l=0.5, b1=0.0, b2=0.6, betta_r=0.5
        )
        assert 0 <= result <= 1


# =============================================================================
# Sort.nxF_Q_possibility Tests
# =============================================================================

class TestSortNxFQPossibility:
    """Test Sort.nxF_Q_possibility static method."""

    def test_nbF_option(self):
        """Test nxF_Q_possibility with nbF option."""
        intermediateA = [-0.5, -0.3, 0.1, 0.3]
        result = Sort.nxF_Q_possibility(intermediateA, "nbF")

        assert 0 <= result <= 1

    def test_neF_option(self):
        """Test nxF_Q_possibility with neF option."""
        intermediateA = [-0.2, -0.1, 0.1, 0.2]
        result = Sort.nxF_Q_possibility(intermediateA, "neF")

        assert 0 <= result <= 1

    def test_nwF_option(self):
        """Test nxF_Q_possibility with nwF option."""
        intermediateA = [0.1, 0.2, 0.3, 0.4]
        result = Sort.nxF_Q_possibility(intermediateA, "nwF")

        assert 0 <= result <= 1


# =============================================================================
# Sort.nxF_Q Tests
# =============================================================================

class TestSortNxFQ:
    """Test Sort.nxF_Q static method."""

    def test_returns_value(self):
        """Test that nxF_Q returns a value."""
        possibilities = {"nbF": 0.3, "neF": 0.5, "nwF": 0.8}
        result = Sort.nxF_Q(possibilities, "nbF")

        assert isinstance(result, (int, float))

    def test_normalized_value(self):
        """Test that nxF_Q normalizes correctly."""
        possibilities = {"nbF": 0.25, "neF": 0.25, "nwF": 0.5}

        result_nbF = Sort.nxF_Q(possibilities, "nbF")
        result_neF = Sort.nxF_Q(possibilities, "neF")
        result_nwF = Sort.nxF_Q(possibilities, "nwF")

        # Sum should be 1 (normalized)
        total = result_nbF + result_neF + result_nwF
        assert total == pytest.approx(1.0)


# =============================================================================
# Sort.final_sum Tests
# =============================================================================

class TestSortFinalSum:
    """Test Sort.final_sum static method."""

    def test_returns_number(self):
        """Test that final_sum returns a number."""
        nxF_Qs = {
            "A": {"nbF": 0.2, "neF": 0.3, "nwF": 0.5},
            "B": {"nbF": 0.1, "neF": 0.4, "nwF": 0.5}
        }
        result = Sort.final_sum(nxF_Qs)

        assert isinstance(result, (int, float))

    def test_result_in_range(self):
        """Test that final_sum result is in valid range."""
        nxF_Qs = {
            "A": {"nbF": 0.3, "neF": 0.4, "nwF": 0.3},
            "B": {"nbF": 0.3, "neF": 0.4, "nwF": 0.3}
        }
        result = Sort.final_sum(nxF_Qs)

        # Result should be between 0 and 1 (or some reasonable bound)
        assert -1 <= result <= 2


# =============================================================================
# Sort.solver_main Tests
# =============================================================================

class TestSortSolverMain:
    """Test Sort.solver_main static method."""

    def test_returns_tuple(self):
        """Test that solver_main returns a tuple."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        result = Sort.solver_main(z1, z2)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_d_plus_do_equals_1(self):
        """Test that d + do = 1."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        d, do = Sort.solver_main(z1, z2)

        assert d + do == pytest.approx(1.0)

    def test_identical_znums(self):
        """Test solver_main with identical Znums."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        d, do = Sort.solver_main(z1, z2)

        # For identical Znums, do should be 1
        assert do == 1.0

    def test_clearly_greater(self):
        """Test solver_main when one is clearly greater."""
        z_small = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z_large = Znum([10, 20, 30, 40], [0.1, 0.2, 0.3, 0.4])

        d_small, do_small = Sort.solver_main(z_small, z_large)
        d_large, do_large = Sort.solver_main(z_large, z_small)

        # Large should dominate small
        assert do_large > do_small

    def test_negative_vs_positive(self):
        """Test solver_main with negative vs positive."""
        z_neg = Znum([-10, -5, -2, -1], [0.1, 0.2, 0.3, 0.4])
        z_pos = Znum([1, 2, 5, 10], [0.1, 0.2, 0.3, 0.4])

        d, do = Sort.solver_main(z_pos, z_neg)

        # Positive should dominate negative
        assert do > 0.5

    def test_same_A_different_B(self):
        """Test solver_main with same A but different B."""
        z_low_B = Znum([1, 2, 3, 4], [0.1, 0.15, 0.2, 0.25])
        z_high_B = Znum([1, 2, 3, 4], [0.5, 0.6, 0.7, 0.8])

        d, do = Sort.solver_main(z_high_B, z_low_B)

        # Higher B should indicate greater reliability/dominance
        assert do > 0.5


# =============================================================================
# Integration Tests
# =============================================================================

class TestSortIntegration:
    """Integration tests for Sort module."""

    def test_comparison_chain(self):
        """Test that comparison chain is consistent."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4])
        z3 = Znum([3, 4, 5, 6], [0.1, 0.2, 0.3, 0.4])

        _, do_12 = Sort.solver_main(z2, z1)  # z2 vs z1
        _, do_23 = Sort.solver_main(z3, z2)  # z3 vs z2
        _, do_13 = Sort.solver_main(z3, z1)  # z3 vs z1

        # z2 > z1, z3 > z2, so z3 should dominate z1
        assert do_12 > 0.5  # z2 > z1
        assert do_23 > 0.5  # z3 > z2
        assert do_13 > 0.5  # z3 > z1 (transitivity)

    def test_normalization_with_overlapping_ranges(self):
        """Test normalization when Znum ranges overlap."""
        z1 = Znum([1, 3, 5, 7], [0.2, 0.3, 0.4, 0.5])
        z2 = Znum([2, 4, 6, 8], [0.2, 0.3, 0.4, 0.5])

        norm1, norm2 = Sort.normalization(z1.A, z2.A)

        # All values should be in [0, 1]
        for val in list(norm1) + list(norm2):
            assert 0 <= val <= 1

    def test_solver_returns_valid_for_all_comparison_pairs(self):
        """Test that solver returns valid results for various pairs."""
        znums = [
            Znum([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]),
            Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
            Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5]),
            Znum([-2, -1, 0, 1], [0.1, 0.2, 0.3, 0.4]),
            Znum([5, 5, 5, 5], [0.3, 0.4, 0.5, 0.6]),  # Exact number
        ]

        for i, z1 in enumerate(znums):
            for j, z2 in enumerate(znums):
                d, do = Sort.solver_main(z1, z2)

                # Both should be valid numbers
                assert isinstance(d, (int, float))
                assert isinstance(do, (int, float))

                # d + do should equal 1
                assert d + do == pytest.approx(1.0)
