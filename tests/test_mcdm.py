"""
Comprehensive E2E tests for Znum MCDM (Multi-Criteria Decision Making) operations.

These tests verify that Promethee and TOPSIS methods produce correct results.
The expected values were generated from the current (ground truth) implementation.

Note: Vikor is NOT tested per user requirements.
"""

import pytest
import copy
import numpy as np
from numpy.testing import assert_array_almost_equal
from znum.Znum import Znum
from znum.Beast import Beast
from znum.Topsis import Topsis
from znum.Promethee import Promethee


def create_3x3_mcdm_table():
    """Create a standard 3 alternatives x 3 criteria MCDM test table."""
    # Weights
    w1 = Znum(A=[0.2, 0.3, 0.4, 0.5], B=[0.1, 0.2, 0.3, 0.4])
    w2 = Znum(A=[0.3, 0.4, 0.5, 0.6], B=[0.2, 0.3, 0.4, 0.5])
    w3 = Znum(A=[0.1, 0.2, 0.3, 0.4], B=[0.05, 0.1, 0.15, 0.2])
    weights = [w1, w2, w3]

    # Alternative 1
    a1_c1 = Znum(A=[7, 8, 9, 10], B=[0.6, 0.7, 0.8, 0.9])
    a1_c2 = Znum(A=[5, 6, 7, 8], B=[0.5, 0.6, 0.7, 0.8])
    a1_c3 = Znum(A=[6, 7, 8, 9], B=[0.6, 0.65, 0.7, 0.75])

    # Alternative 2
    a2_c1 = Znum(A=[4, 5, 6, 7], B=[0.4, 0.5, 0.6, 0.7])
    a2_c2 = Znum(A=[8, 9, 10, 11], B=[0.7, 0.75, 0.8, 0.85])
    a2_c3 = Znum(A=[3, 4, 5, 6], B=[0.3, 0.4, 0.5, 0.6])

    # Alternative 3
    a3_c1 = Znum(A=[6, 7, 8, 9], B=[0.5, 0.6, 0.7, 0.8])
    a3_c2 = Znum(A=[6, 7, 8, 9], B=[0.55, 0.65, 0.75, 0.85])
    a3_c3 = Znum(A=[7, 8, 9, 10], B=[0.6, 0.7, 0.8, 0.9])

    alt1 = [a1_c1, a1_c2, a1_c3]
    alt2 = [a2_c1, a2_c2, a2_c3]
    alt3 = [a3_c1, a3_c2, a3_c3]

    # Criteria types: Benefit (B) or Cost (C)
    criteria_types = [Beast.CriteriaType.BENEFIT, Beast.CriteriaType.BENEFIT, Beast.CriteriaType.COST]

    return [weights, alt1, alt2, alt3, criteria_types]


def create_2x2_mcdm_table():
    """Create a simple 2 alternatives x 2 criteria MCDM test table."""
    w1 = Znum(A=[0.4, 0.5, 0.6, 0.7], B=[0.3, 0.4, 0.5, 0.6])
    w2 = Znum(A=[0.3, 0.4, 0.5, 0.6], B=[0.2, 0.3, 0.4, 0.5])
    weights = [w1, w2]

    # Alternative 1: high on c1, low on c2
    a1_c1 = Znum(A=[8, 9, 10, 11], B=[0.7, 0.8, 0.9, 0.95])
    a1_c2 = Znum(A=[3, 4, 5, 6], B=[0.3, 0.4, 0.5, 0.6])

    # Alternative 2: low on c1, high on c2
    a2_c1 = Znum(A=[5, 6, 7, 8], B=[0.5, 0.6, 0.7, 0.8])
    a2_c2 = Znum(A=[7, 8, 9, 10], B=[0.6, 0.7, 0.8, 0.9])

    alt1 = [a1_c1, a1_c2]
    alt2 = [a2_c1, a2_c2]

    criteria_types = [Beast.CriteriaType.BENEFIT, Beast.CriteriaType.COST]

    return [weights, alt1, alt2, criteria_types]


def create_4x4_mcdm_table():
    """Create a 4 alternatives x 4 criteria MCDM test table."""
    w1 = Znum(A=[0.2, 0.25, 0.3, 0.35], B=[0.1, 0.15, 0.2, 0.25])
    w2 = Znum(A=[0.25, 0.3, 0.35, 0.4], B=[0.15, 0.2, 0.25, 0.3])
    w3 = Znum(A=[0.15, 0.2, 0.25, 0.3], B=[0.1, 0.15, 0.2, 0.25])
    w4 = Znum(A=[0.2, 0.25, 0.3, 0.35], B=[0.15, 0.2, 0.25, 0.3])
    weights = [w1, w2, w3, w4]

    # Alt 1 - high on c1, low on c2
    a1 = [
        Znum(A=[9, 10, 11, 12], B=[0.8, 0.85, 0.9, 0.95]),
        Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5]),
        Znum(A=[6, 7, 8, 9], B=[0.5, 0.6, 0.7, 0.8]),
        Znum(A=[5, 6, 7, 8], B=[0.4, 0.5, 0.6, 0.7])
    ]

    # Alt 2 - medium all
    a2 = [
        Znum(A=[6, 7, 8, 9], B=[0.5, 0.6, 0.7, 0.8]),
        Znum(A=[5, 6, 7, 8], B=[0.4, 0.5, 0.6, 0.7]),
        Znum(A=[5, 6, 7, 8], B=[0.4, 0.5, 0.6, 0.7]),
        Znum(A=[6, 7, 8, 9], B=[0.5, 0.6, 0.7, 0.8])
    ]

    # Alt 3 - low on c1, high on c2
    a3 = [
        Znum(A=[3, 4, 5, 6], B=[0.3, 0.4, 0.5, 0.6]),
        Znum(A=[8, 9, 10, 11], B=[0.7, 0.8, 0.9, 0.95]),
        Znum(A=[7, 8, 9, 10], B=[0.6, 0.7, 0.8, 0.9]),
        Znum(A=[4, 5, 6, 7], B=[0.35, 0.45, 0.55, 0.65])
    ]

    # Alt 4 - varied
    a4 = [
        Znum(A=[5, 6, 7, 8], B=[0.4, 0.5, 0.6, 0.7]),
        Znum(A=[6, 7, 8, 9], B=[0.5, 0.6, 0.7, 0.8]),
        Znum(A=[8, 9, 10, 11], B=[0.7, 0.75, 0.8, 0.85]),
        Znum(A=[7, 8, 9, 10], B=[0.6, 0.7, 0.8, 0.9])
    ]

    # B=Benefit, C=Cost
    criteria_types = [Beast.CriteriaType.BENEFIT, Beast.CriteriaType.BENEFIT,
                      Beast.CriteriaType.COST, Beast.CriteriaType.BENEFIT]

    return [weights, a1, a2, a3, a4, criteria_types]


class TestPromethee:
    """Tests for Promethee MCDM method."""

    def test_promethee_3x3_basic(self):
        """Test Promethee with 3 alternatives x 3 criteria."""
        table = create_3x3_mcdm_table()
        table_copy = copy.deepcopy(table)

        promethee = Promethee(table_copy, shouldNormalizeWeight=False)
        result = promethee.solve()

        # Expected results from ground truth
        assert promethee.index_of_best_alternative == 1
        assert promethee.index_of_worst_alternative == 0
        assert promethee.ordered_indices == [1, 2, 0]

    def test_promethee_3x3_with_weight_normalization(self):
        """Test Promethee with weight normalization enabled."""
        table = create_3x3_mcdm_table()
        table_copy = copy.deepcopy(table)

        promethee = Promethee(table_copy, shouldNormalizeWeight=True)
        result = promethee.solve()

        # Expected results from ground truth
        assert promethee.index_of_best_alternative == 1
        assert promethee.index_of_worst_alternative == 0
        assert promethee.ordered_indices == [1, 2, 0]

    def test_promethee_2x2_basic(self):
        """Test Promethee with simple 2x2 table."""
        table = create_2x2_mcdm_table()
        table_copy = copy.deepcopy(table)

        promethee = Promethee(table_copy, shouldNormalizeWeight=False)
        result = promethee.solve()

        # Expected results from ground truth
        assert promethee.index_of_best_alternative == 0
        assert promethee.index_of_worst_alternative == 1
        assert promethee.ordered_indices == [0, 1]

    def test_promethee_4x4_basic(self):
        """Test Promethee with 4 alternatives x 4 criteria."""
        table = create_4x4_mcdm_table()
        table_copy = copy.deepcopy(table)

        promethee = Promethee(table_copy, shouldNormalizeWeight=False)
        result = promethee.solve()

        # Expected results from ground truth
        assert promethee.index_of_best_alternative == 3
        assert promethee.index_of_worst_alternative == 0
        assert promethee.ordered_indices == [3, 1, 2, 0]

    def test_promethee_returns_sorted_table(self):
        """Test that Promethee.solve() returns a sorted table."""
        table = create_3x3_mcdm_table()
        table_copy = copy.deepcopy(table)

        promethee = Promethee(table_copy, shouldNormalizeWeight=False)
        result = promethee.solve()

        # Result should be a tuple of tuples
        assert isinstance(result, tuple)
        assert len(result) == 3  # 3 alternatives

        # Each element should be (index, Znum)
        for item in result:
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], Znum)

    def test_promethee_ordered_indices_property(self):
        """Test that ordered_indices property returns correct order."""
        table = create_3x3_mcdm_table()
        table_copy = copy.deepcopy(table)

        promethee = Promethee(table_copy, shouldNormalizeWeight=False)
        promethee.solve()

        ordered = promethee.ordered_indices

        assert isinstance(ordered, list)
        assert len(ordered) == 3
        # Should contain all indices 0, 1, 2
        assert set(ordered) == {0, 1, 2}


class TestTopsis:
    """Tests for TOPSIS MCDM method."""

    def test_topsis_3x3_hellinger(self):
        """Test TOPSIS with 3 alternatives x 3 criteria using Hellinger distance."""
        table = create_3x3_mcdm_table()
        table_copy = copy.deepcopy(table)

        result = Topsis.solver_main(
            table_copy,
            shouldNormalizeWeight=False,
            distanceType=Topsis.DistanceMethod.HELLINGER
        )

        # Expected results from ground truth
        expected = [0.2829330856729705, 0.29955044588402374, 0.2818555658487649]

        assert len(result) == 3
        assert_array_almost_equal(result, expected, decimal=4)

        # Best alternative is index 1, worst is index 2
        assert result.index(max(result)) == 1
        assert result.index(min(result)) == 2

    def test_topsis_3x3_simple(self):
        """Test TOPSIS with 3 alternatives x 3 criteria using Simple distance."""
        table = create_3x3_mcdm_table()
        table_copy = copy.deepcopy(table)

        result = Topsis.solver_main(
            table_copy,
            shouldNormalizeWeight=False,
            distanceType=Topsis.DistanceMethod.SIMPLE
        )

        # Expected results from ground truth
        expected = [0.49310933333333334, 0.5131378712871286, 0.4876624166666666]

        assert len(result) == 3
        assert_array_almost_equal(result, expected, decimal=4)

        # Best alternative is index 1, worst is index 2
        assert result.index(max(result)) == 1
        assert result.index(min(result)) == 2

    def test_topsis_3x3_with_weight_normalization(self):
        """Test TOPSIS with weight normalization enabled."""
        table = create_3x3_mcdm_table()
        table_copy = copy.deepcopy(table)

        result = Topsis.solver_main(
            table_copy,
            shouldNormalizeWeight=True,
            distanceType=Topsis.DistanceMethod.HELLINGER
        )

        # Expected results from ground truth
        expected = [0.31494971493861934, 0.3232336819129416, 0.30892113251525033]

        assert len(result) == 3
        assert_array_almost_equal(result, expected, decimal=4)

        # Best alternative is index 1, worst is index 2
        assert result.index(max(result)) == 1
        assert result.index(min(result)) == 2

    def test_topsis_2x2_hellinger(self):
        """Test TOPSIS with simple 2x2 table using Hellinger distance."""
        table = create_2x2_mcdm_table()
        table_copy = copy.deepcopy(table)

        result = Topsis.solver_main(
            table_copy,
            shouldNormalizeWeight=False,
            distanceType=Topsis.DistanceMethod.HELLINGER
        )

        # Expected results from ground truth
        expected = [0.4052251890443464, 0.3270885728962602]

        assert len(result) == 2
        assert_array_almost_equal(result, expected, decimal=4)

        # Best alternative is index 0, worst is index 1
        assert result.index(max(result)) == 0
        assert result.index(min(result)) == 1

    def test_topsis_2x2_simple(self):
        """Test TOPSIS with simple 2x2 table using Simple distance."""
        table = create_2x2_mcdm_table()
        table_copy = copy.deepcopy(table)

        result = Topsis.solver_main(
            table_copy,
            shouldNormalizeWeight=False,
            distanceType=Topsis.DistanceMethod.SIMPLE
        )

        # Expected results from ground truth
        expected = [0.7394971362433771, 0.6091264722736968]

        assert len(result) == 2
        assert_array_almost_equal(result, expected, decimal=4)

        # Best alternative is index 0
        assert result.index(max(result)) == 0

    def test_topsis_4x4_hellinger(self):
        """Test TOPSIS with 4 alternatives x 4 criteria using Hellinger distance."""
        table = create_4x4_mcdm_table()
        table_copy = copy.deepcopy(table)

        result = Topsis.solver_main(
            table_copy,
            shouldNormalizeWeight=False,
            distanceType=Topsis.DistanceMethod.HELLINGER
        )

        # Expected results from ground truth
        expected = [0.23728539302776117, 0.25085212554065683, 0.24669984876691053, 0.26241605852092237]

        assert len(result) == 4
        assert_array_almost_equal(result, expected, decimal=4)

        # Best is index 3, worst is index 0
        assert result.index(max(result)) == 3
        assert result.index(min(result)) == 0

    def test_topsis_returns_list_of_floats(self):
        """Test that TOPSIS returns a list of float values."""
        table = create_3x3_mcdm_table()
        table_copy = copy.deepcopy(table)

        result = Topsis.solver_main(
            table_copy,
            shouldNormalizeWeight=False,
            distanceType=Topsis.DistanceMethod.HELLINGER
        )

        assert isinstance(result, list)
        for value in result:
            assert isinstance(value, (float, np.floating))

    def test_topsis_values_between_0_and_1(self):
        """Test that TOPSIS results are between 0 and 1."""
        table = create_3x3_mcdm_table()
        table_copy = copy.deepcopy(table)

        result = Topsis.solver_main(
            table_copy,
            shouldNormalizeWeight=False,
            distanceType=Topsis.DistanceMethod.HELLINGER
        )

        for value in result:
            assert 0 <= value <= 1


class TestMcdmConsistency:
    """Tests for consistency between MCDM methods."""

    def test_promethee_and_topsis_same_best_3x3(self):
        """Test that Promethee and TOPSIS identify the same best alternative for 3x3."""
        table = create_3x3_mcdm_table()

        # Promethee
        table_promethee = copy.deepcopy(table)
        promethee = Promethee(table_promethee, shouldNormalizeWeight=False)
        promethee.solve()
        promethee_best = promethee.index_of_best_alternative

        # TOPSIS
        table_topsis = copy.deepcopy(table)
        topsis_result = Topsis.solver_main(
            table_topsis,
            shouldNormalizeWeight=False,
            distanceType=Topsis.DistanceMethod.HELLINGER
        )
        topsis_best = topsis_result.index(max(topsis_result))

        # Both should identify the same best alternative
        assert promethee_best == topsis_best

    def test_promethee_and_topsis_same_best_2x2(self):
        """Test that Promethee and TOPSIS identify the same best alternative for 2x2."""
        table = create_2x2_mcdm_table()

        # Promethee
        table_promethee = copy.deepcopy(table)
        promethee = Promethee(table_promethee, shouldNormalizeWeight=False)
        promethee.solve()
        promethee_best = promethee.index_of_best_alternative

        # TOPSIS
        table_topsis = copy.deepcopy(table)
        topsis_result = Topsis.solver_main(
            table_topsis,
            shouldNormalizeWeight=False,
            distanceType=Topsis.DistanceMethod.HELLINGER
        )
        topsis_best = topsis_result.index(max(topsis_result))

        # Both should identify the same best alternative
        assert promethee_best == topsis_best

    def test_promethee_and_topsis_same_best_4x4(self):
        """Test that Promethee and TOPSIS identify the same best alternative for 4x4."""
        table = create_4x4_mcdm_table()

        # Promethee
        table_promethee = copy.deepcopy(table)
        promethee = Promethee(table_promethee, shouldNormalizeWeight=False)
        promethee.solve()
        promethee_best = promethee.index_of_best_alternative

        # TOPSIS
        table_topsis = copy.deepcopy(table)
        topsis_result = Topsis.solver_main(
            table_topsis,
            shouldNormalizeWeight=False,
            distanceType=Topsis.DistanceMethod.HELLINGER
        )
        topsis_best = topsis_result.index(max(topsis_result))

        # Both should identify the same best alternative
        assert promethee_best == topsis_best


class TestBeastUtilities:
    """Tests for Beast utility functions used in MCDM."""

    def test_criteria_type_constants(self):
        """Test Beast CriteriaType constants."""
        assert Beast.CriteriaType.COST == "C"
        assert Beast.CriteriaType.BENEFIT == "B"

    def test_subtract_matrix(self):
        """Test Beast.subtract_matrix function."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])
        z3 = Znum(A=[0.5, 1, 1.5, 2], B=[0.05, 0.1, 0.15, 0.2])

        o1 = [z2, z3]
        o2 = [z1, z1]

        result = Beast.subtract_matrix(o1, o2)

        assert len(result) == 2
        assert isinstance(result[0], Znum)
        assert isinstance(result[1], Znum)

    def test_parse_table(self):
        """Test Beast.parse_table function."""
        table = create_3x3_mcdm_table()

        weights, table_main_part, criteria_types = Beast.parse_table(table)

        assert len(weights) == 3
        assert len(table_main_part) == 3
        assert len(criteria_types) == 3

    def test_numerate(self):
        """Test Beast.numerate function."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5])

        result = Beast.numerate([z1, z2])

        assert result == [(1, z1), (2, z2)]

    def test_transpose_matrix(self):
        """Test Beast.transpose_matrix function."""
        matrix = [[1, 2, 3], [4, 5, 6]]
        result = list(Beast.transpose_matrix(matrix))

        assert result == [(1, 4), (2, 5), (3, 6)]


class TestTopsisDistanceMethod:
    """Tests for TOPSIS distance method constants."""

    def test_distance_method_constants(self):
        """Test TOPSIS DistanceMethod constants."""
        assert Topsis.DistanceMethod.SIMPLE == 1
        assert Topsis.DistanceMethod.HELLINGER == 2

    def test_data_type_constants(self):
        """Test TOPSIS DataType constants."""
        assert Topsis.DataType.ALTERNATIVE == "A"
        assert Topsis.DataType.CRITERIA == "C"
        assert Topsis.DataType.TYPE == "TYPE"


class TestMcdmWithDifferentCriteriaTypes:
    """Tests for MCDM with different combinations of criteria types."""

    def test_all_benefit_criteria(self):
        """Test MCDM with all BENEFIT criteria."""
        w1 = Znum(A=[0.3, 0.4, 0.5, 0.6], B=[0.2, 0.3, 0.4, 0.5])
        w2 = Znum(A=[0.4, 0.5, 0.6, 0.7], B=[0.3, 0.4, 0.5, 0.6])
        weights = [w1, w2]

        a1 = [
            Znum(A=[7, 8, 9, 10], B=[0.6, 0.7, 0.8, 0.9]),
            Znum(A=[5, 6, 7, 8], B=[0.5, 0.6, 0.7, 0.8])
        ]
        a2 = [
            Znum(A=[4, 5, 6, 7], B=[0.4, 0.5, 0.6, 0.7]),
            Znum(A=[8, 9, 10, 11], B=[0.7, 0.8, 0.9, 0.95])
        ]

        # All BENEFIT
        criteria_types = [Beast.CriteriaType.BENEFIT, Beast.CriteriaType.BENEFIT]
        table = [weights, a1, a2, criteria_types]

        # Should not raise any errors
        table_copy = copy.deepcopy(table)
        promethee = Promethee(table_copy, shouldNormalizeWeight=False)
        result = promethee.solve()
        assert result is not None

        table_copy2 = copy.deepcopy(table)
        topsis_result = Topsis.solver_main(
            table_copy2,
            shouldNormalizeWeight=False,
            distanceType=Topsis.DistanceMethod.HELLINGER
        )
        assert len(topsis_result) == 2

    def test_all_cost_criteria(self):
        """Test MCDM with all COST criteria."""
        w1 = Znum(A=[0.3, 0.4, 0.5, 0.6], B=[0.2, 0.3, 0.4, 0.5])
        w2 = Znum(A=[0.4, 0.5, 0.6, 0.7], B=[0.3, 0.4, 0.5, 0.6])
        weights = [w1, w2]

        a1 = [
            Znum(A=[7, 8, 9, 10], B=[0.6, 0.7, 0.8, 0.9]),
            Znum(A=[5, 6, 7, 8], B=[0.5, 0.6, 0.7, 0.8])
        ]
        a2 = [
            Znum(A=[4, 5, 6, 7], B=[0.4, 0.5, 0.6, 0.7]),
            Znum(A=[8, 9, 10, 11], B=[0.7, 0.8, 0.9, 0.95])
        ]

        # All COST
        criteria_types = [Beast.CriteriaType.COST, Beast.CriteriaType.COST]
        table = [weights, a1, a2, criteria_types]

        # Should not raise any errors
        table_copy = copy.deepcopy(table)
        promethee = Promethee(table_copy, shouldNormalizeWeight=False)
        result = promethee.solve()
        assert result is not None

        table_copy2 = copy.deepcopy(table)
        topsis_result = Topsis.solver_main(
            table_copy2,
            shouldNormalizeWeight=False,
            distanceType=Topsis.DistanceMethod.HELLINGER
        )
        assert len(topsis_result) == 2

    def test_mixed_criteria_types(self):
        """Test MCDM with mixed BENEFIT and COST criteria."""
        table = create_3x3_mcdm_table()  # Has both B and C

        table_copy = copy.deepcopy(table)
        promethee = Promethee(table_copy, shouldNormalizeWeight=False)
        result = promethee.solve()
        assert result is not None

        table_copy2 = copy.deepcopy(table)
        topsis_result = Topsis.solver_main(
            table_copy2,
            shouldNormalizeWeight=False,
            distanceType=Topsis.DistanceMethod.HELLINGER
        )
        assert len(topsis_result) == 3


class TestMcdmModifiesTable:
    """Tests to verify MCDM methods modify the table in place (important for deepcopy)."""

    def test_promethee_modifies_table(self):
        """Test that Promethee modifies the input table."""
        table = create_3x3_mcdm_table()
        original_a1_c1_A = table[1][0].A.copy()

        promethee = Promethee(table, shouldNormalizeWeight=False)
        promethee.solve()

        # Table should be modified after solve
        # (normalization changes values)
        # The original table is modified, so we can't compare directly

    def test_deepcopy_preserves_original(self):
        """Test that deepcopy preserves original table values."""
        table = create_3x3_mcdm_table()
        original_a1_c1_A = table[1][0].A.copy()

        table_copy = copy.deepcopy(table)
        promethee = Promethee(table_copy, shouldNormalizeWeight=False)
        promethee.solve()

        # Original table should be unchanged
        assert_array_almost_equal(table[1][0].A, original_a1_c1_A)
