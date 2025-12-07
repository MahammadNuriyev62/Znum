import pytest
import numpy as np
from znum.Znum import Znum
from znum.Topsis import Topsis
from znum.Dist import Dist
from znum.Beast import Beast


# =============================================================================
# Topsis.DataType Constants Tests
# =============================================================================

class TestTopsisDataType:
    """Test Topsis.DataType constants."""

    def test_alternative_constant(self):
        """Test ALTERNATIVE constant value."""
        assert Topsis.DataType.ALTERNATIVE == "A"

    def test_criteria_constant(self):
        """Test CRITERIA constant value."""
        assert Topsis.DataType.CRITERIA == "C"

    def test_type_constant(self):
        """Test TYPE constant value."""
        assert Topsis.DataType.TYPE == "TYPE"


# =============================================================================
# Topsis.DistanceMethod Constants Tests
# =============================================================================

class TestTopsisDistanceMethod:
    """Test Topsis.DistanceMethod constants."""

    def test_simple_constant(self):
        """Test SIMPLE constant value."""
        assert Topsis.DistanceMethod.SIMPLE == 1

    def test_hellinger_constant(self):
        """Test HELLINGER constant value."""
        assert Topsis.DistanceMethod.HELLINGER == 2

    def test_constants_are_different(self):
        """Test that SIMPLE and HELLINGER are different."""
        assert Topsis.DistanceMethod.SIMPLE != Topsis.DistanceMethod.HELLINGER


# =============================================================================
# Topsis.weightage Tests
# =============================================================================

class TestTopsisWeightage:
    """Test Topsis.weightage static method."""

    def test_applies_weights_to_row(self):
        """Test that weights are multiplied to each element."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        w1 = Znum([0.5, 0.6, 0.7, 0.8], [0.4, 0.5, 0.6, 0.7])
        w2 = Znum([0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6])

        table_main_part = [[z1, z2]]
        weights = [w1, w2]

        Topsis.weightage(table_main_part, weights)

        # Elements should be modified in place
        assert isinstance(table_main_part[0][0], Znum)
        assert isinstance(table_main_part[0][1], Znum)

    def test_multiple_rows(self):
        """Test weightage with multiple rows."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])
        z3 = Znum([3, 4, 5, 6], [0.3, 0.4, 0.5, 0.6])
        z4 = Znum([4, 5, 6, 7], [0.4, 0.5, 0.6, 0.7])

        w1 = Znum([0.5, 0.6, 0.7, 0.8], [0.4, 0.5, 0.6, 0.7])
        w2 = Znum([0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6])

        table_main_part = [[z1, z2], [z3, z4]]
        weights = [w1, w2]

        Topsis.weightage(table_main_part, weights)

        # All elements should be Znums
        for row in table_main_part:
            for elem in row:
                assert isinstance(elem, Znum)


# =============================================================================
# Topsis.get_table_n Tests
# =============================================================================

class TestTopsisGetTableN:
    """Test Topsis.get_table_n static method."""

    def test_returns_list_of_lists(self):
        """Test that get_table_n returns list of lists."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        table_main_part = [[z1, z2]]

        result = Topsis.get_table_n(table_main_part, lambda znum: 1.0)

        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_applies_distance_solver(self):
        """Test that distance solver is applied to each element."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        table_main_part = [[z1, z2]]

        # Use a simple solver that returns a constant
        result = Topsis.get_table_n(table_main_part, lambda znum: 5.0)

        assert result == [[5.0, 5.0]]

    def test_with_simple_distance(self):
        """Test with Dist.Simple.calculate."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        table_main_part = [[z1]]

        result = Topsis.get_table_n(
            table_main_part,
            lambda znum: Dist.Simple.calculate(znum, 0)
        )

        assert len(result) == 1
        assert len(result[0]) == 1
        assert isinstance(result[0][0], (int, float, np.floating))

    def test_preserves_structure(self):
        """Test that output structure matches input."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])
        z3 = Znum([3, 4, 5, 6], [0.3, 0.4, 0.5, 0.6])
        z4 = Znum([4, 5, 6, 7], [0.4, 0.5, 0.6, 0.7])

        table_main_part = [[z1, z2], [z3, z4]]

        result = Topsis.get_table_n(table_main_part, lambda znum: 1.0)

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2


# =============================================================================
# Topsis.find_extremum Tests
# =============================================================================

class TestTopsisFindExtremum:
    """Test Topsis.find_extremum static method."""

    def test_sums_rows(self):
        """Test that find_extremum sums each row."""
        table_n = [[1, 2, 3], [4, 5, 6]]

        result = Topsis.find_extremum(table_n)

        assert result == [6, 15]

    def test_single_row(self):
        """Test with single row."""
        table_n = [[1, 2, 3, 4]]

        result = Topsis.find_extremum(table_n)

        assert result == [10]

    def test_single_column(self):
        """Test with single column."""
        table_n = [[1], [2], [3]]

        result = Topsis.find_extremum(table_n)

        assert result == [1, 2, 3]

    def test_returns_list(self):
        """Test that result is a list."""
        table_n = [[1, 2], [3, 4]]

        result = Topsis.find_extremum(table_n)

        assert isinstance(result, list)


# =============================================================================
# Topsis.find_distance Tests
# =============================================================================

class TestTopsisFindDistance:
    """Test Topsis.find_distance static method."""

    def test_calculates_ratio(self):
        """Test that distance ratio is calculated correctly."""
        s_best = [1, 2, 3]
        s_worst = [3, 2, 1]

        result = Topsis.find_distance(s_best, s_worst)

        # worst / (best + worst)
        # [3/(1+3), 2/(2+2), 1/(3+1)] = [0.75, 0.5, 0.25]
        assert result == pytest.approx([0.75, 0.5, 0.25])

    def test_equal_best_worst(self):
        """Test when best equals worst."""
        s_best = [1, 1, 1]
        s_worst = [1, 1, 1]

        result = Topsis.find_distance(s_best, s_worst)

        # 1 / (1 + 1) = 0.5
        assert result == pytest.approx([0.5, 0.5, 0.5])

    def test_returns_list(self):
        """Test that result is a list."""
        s_best = [1, 2]
        s_worst = [2, 1]

        result = Topsis.find_distance(s_best, s_worst)

        assert isinstance(result, list)

    def test_values_between_0_and_1(self):
        """Test that all values are between 0 and 1."""
        s_best = [0.5, 1.0, 2.0]
        s_worst = [1.5, 2.0, 3.0]

        result = Topsis.find_distance(s_best, s_worst)

        for val in result:
            assert 0 <= val <= 1


# =============================================================================
# Topsis.solver_main Tests
# =============================================================================

class TestTopsisSolverMain:
    """Test Topsis.solver_main static method."""

    def test_returns_list(self):
        """Test that solver_main returns a list."""
        w1 = Znum([0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7])
        w2 = Znum([0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8])

        z11 = Znum([0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6])
        z12 = Znum([0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7])

        z21 = Znum([0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8])
        z22 = Znum([0.4, 0.5, 0.6, 0.7], [0.6, 0.7, 0.8, 0.9])

        table = [
            [w1, w2],
            [z11, z12],
            [z21, z22],
            ["B", "B"]
        ]

        result = Topsis.solver_main(table)

        assert isinstance(result, list)

    def test_result_length_matches_alternatives(self):
        """Test that result has same length as number of alternatives."""
        w1 = Znum([0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7])

        z11 = Znum([0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6])
        z21 = Znum([0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8])
        z31 = Znum([0.5, 0.6, 0.7, 0.8], [0.6, 0.7, 0.8, 0.9])

        table = [
            [w1],
            [z11],
            [z21],
            [z31],
            ["B"]
        ]

        result = Topsis.solver_main(table)

        # 3 alternatives
        assert len(result) == 3

    def test_with_simple_distance_method(self):
        """Test with SIMPLE distance method."""
        w1 = Znum([0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7])

        z11 = Znum([0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6])
        z21 = Znum([0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8])

        table = [
            [w1],
            [z11],
            [z21],
            ["B"]
        ]

        result = Topsis.solver_main(table, distanceType=Topsis.DistanceMethod.SIMPLE)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_with_hellinger_distance_method(self):
        """Test with HELLINGER distance method."""
        w1 = Znum([0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7])

        z11 = Znum([0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6])
        z21 = Znum([0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8])

        table = [
            [w1],
            [z11],
            [z21],
            ["B"]
        ]

        result = Topsis.solver_main(table, distanceType=Topsis.DistanceMethod.HELLINGER)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_with_weight_normalization(self):
        """Test with weight normalization enabled."""
        w1 = Znum([0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7])
        w2 = Znum([0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8])

        z11 = Znum([0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6])
        z12 = Znum([0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7])

        z21 = Znum([0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8])
        z22 = Znum([0.4, 0.5, 0.6, 0.7], [0.6, 0.7, 0.8, 0.9])

        table = [
            [w1, w2],
            [z11, z12],
            [z21, z22],
            ["B", "B"]
        ]

        result = Topsis.solver_main(table, shouldNormalizeWeight=True)

        assert isinstance(result, list)

    def test_result_values_between_0_and_1(self):
        """Test that result values are between 0 and 1."""
        w1 = Znum([0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7])

        z11 = Znum([0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6])
        z21 = Znum([0.5, 0.6, 0.7, 0.8], [0.6, 0.7, 0.8, 0.9])

        table = [
            [w1],
            [z11],
            [z21],
            ["B"]
        ]

        result = Topsis.solver_main(table)

        for val in result:
            assert 0 <= val <= 1

    def test_with_cost_criteria(self):
        """Test with cost criteria type."""
        w1 = Znum([0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7])

        z11 = Znum([0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6])
        z21 = Znum([0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8])

        table = [
            [w1],
            [z11],
            [z21],
            ["C"]  # Cost criteria
        ]

        result = Topsis.solver_main(table)

        assert isinstance(result, list)
        assert len(result) == 2


# =============================================================================
# Integration Tests
# =============================================================================

class TestTopsisIntegration:
    """Integration tests for Topsis module."""

    def test_complete_workflow(self):
        """Test complete TOPSIS workflow."""
        # Weights
        w1 = Znum([0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8])
        w2 = Znum([0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7])

        # Alternatives
        a1_c1 = Znum([0.6, 0.7, 0.8, 0.9], [0.7, 0.8, 0.85, 0.9])
        a1_c2 = Znum([0.5, 0.6, 0.7, 0.8], [0.6, 0.7, 0.8, 0.85])

        a2_c1 = Znum([0.4, 0.5, 0.6, 0.7], [0.5, 0.6, 0.7, 0.8])
        a2_c2 = Znum([0.7, 0.8, 0.85, 0.9], [0.75, 0.8, 0.85, 0.9])

        a3_c1 = Znum([0.5, 0.6, 0.7, 0.8], [0.6, 0.7, 0.75, 0.8])
        a3_c2 = Znum([0.6, 0.7, 0.75, 0.8], [0.65, 0.7, 0.75, 0.8])

        table = [
            [w1, w2],           # Weights
            [a1_c1, a1_c2],     # Alternative 1
            [a2_c1, a2_c2],     # Alternative 2
            [a3_c1, a3_c2],     # Alternative 3
            ["B", "B"]          # Criteria types (Benefit)
        ]

        result = Topsis.solver_main(table)

        # Should have 3 results (one per alternative)
        assert len(result) == 3

        # All results should be valid ratios
        for r in result:
            assert isinstance(r, (int, float))
            assert 0 <= r <= 1

    def test_ranking_obvious_best(self):
        """Test that obviously best alternative ranks highest."""
        w1 = Znum([0.5, 0.6, 0.7, 0.8], [0.6, 0.7, 0.8, 0.9])

        # Best alternative - high values
        best = Znum([0.8, 0.85, 0.9, 0.95], [0.85, 0.9, 0.92, 0.95])
        # Worst alternative - low values
        worst = Znum([0.1, 0.15, 0.2, 0.25], [0.3, 0.4, 0.5, 0.6])

        table = [
            [w1],
            [best],
            [worst],
            ["B"]
        ]

        result = Topsis.solver_main(table)

        # Higher value = better for TOPSIS (closer to ideal)
        # best should have higher score than worst
        assert result[0] > result[1]

    def test_mixed_criteria_types(self):
        """Test with mixed benefit and cost criteria."""
        w1 = Znum([0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8])
        w2 = Znum([0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8])

        z11 = Znum([0.5, 0.6, 0.7, 0.8], [0.6, 0.7, 0.8, 0.85])
        z12 = Znum([0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8])

        z21 = Znum([0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8])
        z22 = Znum([0.5, 0.6, 0.7, 0.8], [0.6, 0.7, 0.8, 0.85])

        table = [
            [w1, w2],
            [z11, z12],
            [z21, z22],
            ["B", "C"]  # Mixed: Benefit and Cost
        ]

        result = Topsis.solver_main(table)

        assert len(result) == 2
        for r in result:
            assert 0 <= r <= 1
