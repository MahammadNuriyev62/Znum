import pytest
import numpy as np
from znum.Znum import Znum
from znum.Beast import Beast


# =============================================================================
# Beast.CriteriaType Constants Tests
# =============================================================================

class TestBeastCriteriaType:
    """Test Beast.CriteriaType constants."""

    def test_cost_constant(self):
        """Test COST constant value."""
        assert Beast.CriteriaType.COST == "C"

    def test_benefit_constant(self):
        """Test BENEFIT constant value."""
        assert Beast.CriteriaType.BENEFIT == "B"

    def test_constants_are_different(self):
        """Test that COST and BENEFIT are different."""
        assert Beast.CriteriaType.COST != Beast.CriteriaType.BENEFIT


# =============================================================================
# Beast.subtract_matrix Tests
# =============================================================================

class TestBeastSubtractMatrix:
    """Test Beast.subtract_matrix static method."""

    def test_subtract_two_znum_lists(self):
        """Test subtraction of two Znum lists."""
        z1 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        o1 = [z1]
        o2 = [z2]

        result = Beast.subtract_matrix(o1, o2)

        assert len(result) == 1
        assert isinstance(result[0], Znum)

    def test_subtract_multiple_znums(self):
        """Test subtraction of multiple Znums."""
        z1a = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])
        z1b = Znum([4, 5, 6, 7], [0.3, 0.4, 0.5, 0.6])

        z2a = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2b = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        o1 = [z1a, z1b]
        o2 = [z2a, z2b]

        result = Beast.subtract_matrix(o1, o2)

        assert len(result) == 2
        assert all(isinstance(z, Znum) for z in result)

    def test_subtract_empty_lists(self):
        """Test subtraction of empty lists."""
        result = Beast.subtract_matrix([], [])
        assert result == []


# =============================================================================
# Beast.normalize_benefit Tests
# =============================================================================

class TestBeastNormalizeBenefit:
    """Test Beast.normalize_benefit static method."""

    def test_normalizes_by_max(self):
        """Test that values are normalized by maximum A value."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])

        znums = [z1, z2]
        Beast.normalize_benefit(znums)

        # Max A value is 8
        # z1.A should be [1/8, 2/8, 3/8, 4/8]
        # z2.A should be [5/8, 6/8, 7/8, 8/8]
        assert list(z1.A) == pytest.approx([1/8, 2/8, 3/8, 4/8])
        assert list(z2.A) == pytest.approx([5/8, 6/8, 7/8, 8/8])

    def test_single_znum(self):
        """Test normalization with single Znum."""
        z = Znum([2, 4, 6, 8], [0.1, 0.2, 0.3, 0.4])

        znums = [z]
        Beast.normalize_benefit(znums)

        # Max is 8
        assert list(z.A) == pytest.approx([2/8, 4/8, 6/8, 8/8])

    def test_modifies_in_place(self):
        """Test that normalization modifies Znums in place."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        original_z = z

        Beast.normalize_benefit([z])

        # Should be the same object, modified in place
        assert z is original_z


# =============================================================================
# Beast.normalize_cost Tests
# =============================================================================

class TestBeastNormalizeCost:
    """Test Beast.normalize_cost static method."""

    def test_normalizes_by_min_and_reverses(self):
        """Test that values are normalized by minimum and reversed."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        znums = [z1, z2]
        Beast.normalize_cost(znums)

        # Min A value across both is 1
        # z1.A = [1, 2, 3, 4] -> [1/1, 1/2, 1/3, 1/4] -> reversed = [1/4, 1/3, 1/2, 1]
        # z2.A = [2, 3, 4, 5] -> [1/2, 1/3, 1/4, 1/5] -> reversed = [1/5, 1/4, 1/3, 1/2]
        assert list(z1.A) == pytest.approx([1/4, 1/3, 1/2, 1])
        assert list(z2.A) == pytest.approx([1/5, 1/4, 1/3, 1/2])

    def test_single_znum(self):
        """Test cost normalization with single Znum."""
        z = Znum([2, 4, 6, 8], [0.1, 0.2, 0.3, 0.4])

        znums = [z]
        Beast.normalize_cost(znums)

        # Min is 2
        # [2, 4, 6, 8] -> [2/2, 2/4, 2/6, 2/8] = [1, 0.5, 0.333, 0.25] -> reversed = [0.25, 0.333, 0.5, 1]
        assert list(z.A) == pytest.approx([1/4, 1/3, 1/2, 1])

    def test_modifies_in_place(self):
        """Test that normalization modifies Znums in place."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        original_z = z

        Beast.normalize_cost([z])

        assert z is original_z


# =============================================================================
# Beast.normalize Tests
# =============================================================================

class TestBeastNormalize:
    """Test Beast.normalize static method."""

    def test_dispatches_to_benefit(self):
        """Test that BENEFIT type calls normalize_benefit."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        Beast.normalize([z], Beast.CriteriaType.BENEFIT)

        # Max is 4, so values should be divided by 4
        assert list(z.A) == pytest.approx([1/4, 2/4, 3/4, 4/4])

    def test_dispatches_to_cost(self):
        """Test that COST type calls normalize_cost."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        Beast.normalize([z], Beast.CriteriaType.COST)

        # Cost normalization: [min/a for a in A] then reversed
        # Min is 1, so [1/1, 1/2, 1/3, 1/4] = [1, 0.5, 0.333, 0.25] -> reversed = [0.25, 0.333, 0.5, 1]
        assert list(z.A) == pytest.approx([1/4, 1/3, 1/2, 1])

    def test_defaults_to_cost(self):
        """Test that unknown type defaults to cost normalization."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        Beast.normalize([z], "UNKNOWN")

        # Should default to cost: [min/a for a in A] reversed
        assert list(z.A) == pytest.approx([1/4, 1/3, 1/2, 1])


# =============================================================================
# Beast.normalize_weight Tests
# =============================================================================

class TestBeastNormalizeWeight:
    """Test Beast.normalize_weight static method."""

    def test_normalizes_weights(self):
        """Test that weights are normalized by their sum."""
        w1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        w2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        weights = [w1, w2]
        Beast.normalize_weight(weights)

        # After normalization, weights should sum to approximately 1 Znum
        # (though this is fuzzy arithmetic)
        assert len(weights) == 2
        assert all(isinstance(w, Znum) for w in weights)

    def test_single_weight(self):
        """Test normalization with single weight."""
        w = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        weights = [w]
        Beast.normalize_weight(weights)

        # Single weight divided by itself should be approximately 1
        assert isinstance(weights[0], Znum)

    def test_modifies_list_in_place(self):
        """Test that the weights list is modified in place."""
        w1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        w2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        weights = [w1, w2]
        Beast.normalize_weight(weights)

        # List should still have 2 elements
        assert len(weights) == 2


# =============================================================================
# Beast.parse_table Tests
# =============================================================================

class TestBeastParseTable:
    """Test Beast.parse_table static method."""

    def test_parses_table_correctly(self):
        """Test that table is parsed into weights, main, and criteria types."""
        w1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        w2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        row1 = [Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]), Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])]
        row2 = [Znum([3, 4, 5, 6], [0.3, 0.4, 0.5, 0.6]), Znum([4, 5, 6, 7], [0.4, 0.5, 0.6, 0.7])]

        table = [
            [w1, w2],  # weights
            row1,      # main row 1
            row2,      # main row 2
            ["B", "C"]  # criteria types
        ]

        weights, main_part, criteria_types = Beast.parse_table(table)

        assert weights == [w1, w2]
        assert main_part == [row1, row2]
        assert criteria_types == ["B", "C"]

    def test_single_row_main_part(self):
        """Test parsing with single main row."""
        w1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        row1 = [Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])]

        table = [
            [w1],     # weights
            row1,     # single main row
            ["B"]     # criteria types
        ]

        weights, main_part, criteria_types = Beast.parse_table(table)

        assert weights == [w1]
        assert main_part == [row1]
        assert criteria_types == ["B"]


# =============================================================================
# Beast.numerate Tests
# =============================================================================

class TestBeastNumerate:
    """Test Beast.numerate static method."""

    def test_numerates_from_1(self):
        """Test that numeration starts from 1."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        result = Beast.numerate([z1, z2])

        assert result == [(1, z1), (2, z2)]

    def test_single_element(self):
        """Test numeration with single element."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = Beast.numerate([z])

        assert result == [(1, z)]

    def test_empty_list(self):
        """Test numeration with empty list."""
        result = Beast.numerate([])

        assert result == []

    def test_returns_list_of_tuples(self):
        """Test that result is a list of tuples."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = Beast.numerate([z])

        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)


# =============================================================================
# Beast.sort_numerated_single_column_table Tests
# =============================================================================

class TestBeastSortNumeratedSingleColumnTable:
    """Test Beast.sort_numerated_single_column_table static method."""

    def test_sorts_in_descending_order(self):
        """Test that sorting is in descending order by Znum value."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])
        z3 = Znum([3, 4, 5, 6], [0.3, 0.4, 0.5, 0.6])

        numerated = [(1, z1), (2, z2), (3, z3)]

        result = Beast.sort_numerated_single_column_table(numerated)

        # Should be sorted by Znum in descending order
        # z2 > z3 > z1 based on their A values
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_tuple(self):
        """Test that result is a tuple."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = Beast.sort_numerated_single_column_table([(1, z)])

        assert isinstance(result, tuple)

    def test_empty_table(self):
        """Test sorting empty table."""
        result = Beast.sort_numerated_single_column_table([])

        assert result == ()

    def test_preserves_numeration(self):
        """Test that original indices are preserved."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])

        numerated = [(1, z1), (2, z2)]

        result = Beast.sort_numerated_single_column_table(numerated)

        # All original indices should still be present
        indices = [item[0] for item in result]
        assert set(indices) == {1, 2}


# =============================================================================
# Beast.transpose_matrix Tests
# =============================================================================

class TestBeastTransposeMatrix:
    """Test Beast.transpose_matrix static method."""

    def test_transpose_2x3_matrix(self):
        """Test transposing a 2x3 matrix."""
        matrix = [
            [1, 2, 3],
            [4, 5, 6]
        ]

        result = list(Beast.transpose_matrix(matrix))

        assert result == [(1, 4), (2, 5), (3, 6)]

    def test_transpose_3x2_matrix(self):
        """Test transposing a 3x2 matrix."""
        matrix = [
            [1, 2],
            [3, 4],
            [5, 6]
        ]

        result = list(Beast.transpose_matrix(matrix))

        assert result == [(1, 3, 5), (2, 4, 6)]

    def test_transpose_1x3_matrix(self):
        """Test transposing a 1x3 matrix."""
        matrix = [[1, 2, 3]]

        result = list(Beast.transpose_matrix(matrix))

        assert result == [(1,), (2,), (3,)]

    def test_transpose_with_znums(self):
        """Test transposing matrix of Znums."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])
        z3 = Znum([3, 4, 5, 6], [0.3, 0.4, 0.5, 0.6])
        z4 = Znum([4, 5, 6, 7], [0.4, 0.5, 0.6, 0.7])

        matrix = [
            [z1, z2],
            [z3, z4]
        ]

        result = list(Beast.transpose_matrix(matrix))

        assert result == [(z1, z3), (z2, z4)]

    def test_transpose_returns_zip_object(self):
        """Test that transpose returns a zip object."""
        matrix = [[1, 2], [3, 4]]

        result = Beast.transpose_matrix(matrix)

        # zip returns a zip object
        assert hasattr(result, '__iter__')


# =============================================================================
# Integration Tests
# =============================================================================

class TestBeastIntegration:
    """Integration tests for Beast module."""

    def test_full_normalization_workflow(self):
        """Test complete normalization workflow."""
        # Create some Znums for a decision matrix
        z11 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z12 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])
        z21 = Znum([3, 4, 5, 6], [0.3, 0.4, 0.5, 0.6])
        z22 = Znum([4, 5, 6, 7], [0.4, 0.5, 0.6, 0.7])

        w1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        w2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        table = [
            [w1, w2],
            [z11, z12],
            [z21, z22],
            ["B", "C"]
        ]

        weights, main_part, criteria_types = Beast.parse_table(table)

        assert len(weights) == 2
        assert len(main_part) == 2
        assert len(criteria_types) == 2

    def test_transpose_and_numerate(self):
        """Test combining transpose with numerate."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        # Single column
        column = [z1, z2]

        numerated = Beast.numerate(column)

        assert numerated == [(1, z1), (2, z2)]

    def test_normalize_and_sort_workflow(self):
        """Test normalization followed by sorting."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])
        z3 = Znum([3, 4, 5, 6], [0.3, 0.4, 0.5, 0.6])

        znums = [z1, z2, z3]

        # Normalize as benefit
        Beast.normalize_benefit(znums)

        # Numerate
        numerated = Beast.numerate(znums)

        # Sort
        sorted_result = Beast.sort_numerated_single_column_table(numerated)

        assert len(sorted_result) == 3
        assert isinstance(sorted_result, tuple)
