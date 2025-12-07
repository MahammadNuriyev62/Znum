import pytest
import numpy as np
from znum.Znum import Znum
from znum.Math import Math


# =============================================================================
# Math.Operations Tests
# =============================================================================

class TestMathOperations:
    """Test Math.Operations constants."""

    def test_addition_constant(self):
        assert Math.Operations.ADDITION == 1

    def test_subtraction_constant(self):
        assert Math.Operations.SUBTRACTION == 2

    def test_division_constant(self):
        assert Math.Operations.DIVISION == 3

    def test_multiplication_constant(self):
        assert Math.Operations.MULTIPLICATION == 4


# =============================================================================
# Math.QIntermediate Tests
# =============================================================================

class TestMathQIntermediate:
    """Test Math.QIntermediate constants."""

    def test_value_constant(self):
        assert Math.QIntermediate.VALUE == "value"

    def test_membership_constant(self):
        assert Math.QIntermediate.MEMBERSHIP == "memb"


# =============================================================================
# Math.operationFunctions Tests
# =============================================================================

class TestMathOperationFunctions:
    """Test Math.operationFunctions dictionary."""

    def test_addition_function(self):
        func = Math.operationFunctions[Math.Operations.ADDITION]
        assert func(3, 2) == 5

    def test_subtraction_function(self):
        func = Math.operationFunctions[Math.Operations.SUBTRACTION]
        assert func(5, 3) == 2

    def test_multiplication_function(self):
        func = Math.operationFunctions[Math.Operations.MULTIPLICATION]
        assert func(4, 3) == 12

    def test_division_function(self):
        func = Math.operationFunctions[Math.Operations.DIVISION]
        assert func(10, 2) == 5

    def test_all_operations_exist(self):
        assert len(Math.operationFunctions) == 4


# =============================================================================
# Math.get_default_membership Tests
# =============================================================================

class TestMathGetDefaultMembership:
    """Test Math.get_default_membership static method."""

    def test_size_4(self):
        result = Math.get_default_membership(4)
        assert len(result) == 4
        # Should be [0, 1, 1, 0] for size 4
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 1
        assert result[3] == 0

    def test_size_6(self):
        result = Math.get_default_membership(6)
        assert len(result) == 6
        # Symmetric, peaks in middle
        assert result[0] == 0
        assert result[-1] == 0

    def test_size_8(self):
        result = Math.get_default_membership(8)
        assert len(result) == 8
        assert result[0] == 0
        assert result[-1] == 0

    def test_symmetry(self):
        result = Math.get_default_membership(6)
        # Check symmetry
        for i in range(len(result) // 2):
            assert result[i] == result[-(i + 1)]


# =============================================================================
# Math Instance Method Tests
# =============================================================================

class TestMathInstanceMethods:
    """Test Math instance methods."""

    def test_get_intermediate_structure(self):
        """Test that get_intermediate returns correct structure."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = z.math.get_intermediate(z.A)

        assert "value" in result
        assert "memb" in result
        assert len(result["value"]) == len(result["memb"])

    def test_get_intermediate_A(self):
        """Test get_intermediate for A values."""
        z = Znum([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
        result = z.math.get_intermediate(z.A)

        # Values should include interpolated points
        assert len(result["value"]) > 4

    def test_get_intermediate_B(self):
        """Test get_intermediate for B values."""
        z = Znum([1, 2, 3, 4], [0.0, 0.25, 0.75, 1.0])
        result = z.math.get_intermediate(z.B)

        assert len(result["value"]) > 4

    def test_get_membership(self):
        """Test get_membership method."""
        z = Znum([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
        # Get membership at a point within the trapezoid
        result = z.math.get_membership(z.A, 1.5)

        # Should return a valid membership value
        assert 0 <= result <= 1

    def test_get_y_interpolation(self):
        """Test get_y linear interpolation."""
        z = Znum([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
        xs = np.array([0, 1, 2, 3])
        ys = np.array([0, 1, 1, 0])

        # Test at known point
        result = z.math.get_y(0.5, xs, ys)
        assert result == 0.5  # Linear interpolation between (0,0) and (1,1)

    def test_get_y_at_peak(self):
        """Test get_y at peak membership."""
        z = Znum([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
        xs = np.array([0, 1, 2, 3])
        ys = np.array([0, 1, 1, 0])

        result = z.math.get_y(1.5, xs, ys)
        assert result == 1.0  # Should be at peak


# =============================================================================
# Math.get_matrix Tests
# =============================================================================

class TestMathGetMatrix:
    """Test Math.get_matrix method."""

    def test_get_matrix_returns_array(self):
        """Test that get_matrix returns numpy array."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = z.math.get_matrix()

        assert isinstance(result, np.ndarray)

    def test_get_matrix_shape(self):
        """Test get_matrix output shape."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = z.math.get_matrix()

        # Should be a 2D array with multiple rows
        assert len(result.shape) == 2

    def test_get_matrix_values_in_range(self):
        """Test that get_matrix values are in valid range [0, 1]."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = z.math.get_matrix()

        # All probabilities should be between 0 and 1
        assert np.all(result >= -0.001)  # Allow small numerical errors
        assert np.all(result <= 1.001)


# =============================================================================
# Math.get_i37 Tests
# =============================================================================

class TestMathGetI37:
    """Test Math.get_i37 static method."""

    def test_get_i37_returns_scalar(self):
        """Test that get_i37 returns a scalar."""
        Q_int = {
            "value": np.array([1, 2, 3, 4]),
            "memb": np.array([0.1, 0.3, 0.3, 0.1])
        }
        result = Math.get_i37(Q_int)

        assert isinstance(result, (int, float, np.floating))

    def test_get_i37_weighted_average(self):
        """Test that get_i37 computes weighted average correctly."""
        Q_int = {
            "value": np.array([1, 2, 3, 4]),
            "memb": np.array([0.25, 0.25, 0.25, 0.25])
        }
        result = Math.get_i37(Q_int)

        # With equal weights, should be simple average
        expected = (1 + 2 + 3 + 4) / 4
        assert result == pytest.approx(expected)


# =============================================================================
# Math.get_Q_from_matrix Tests
# =============================================================================

class TestMathGetQFromMatrix:
    """Test Math.get_Q_from_matrix static method."""

    def test_returns_list_of_4(self):
        """Test that get_Q_from_matrix returns 4-element list."""
        matrix = [
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 0.5]
        ]
        result = Math.get_Q_from_matrix(matrix)

        assert len(result) == 4

    def test_extracts_min_max(self):
        """Test that min and max are extracted correctly."""
        matrix = [
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 0.5]
        ]
        result = Math.get_Q_from_matrix(matrix)

        assert result[0] == 1.0  # Min
        assert result[3] == 4.0  # Max


# =============================================================================
# Math.get_matrix_main Tests
# =============================================================================

class TestMathGetMatrixMain:
    """Test Math.get_matrix_main static method."""

    def test_addition_matrix(self):
        """Test get_matrix_main for addition."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        result = Math.get_matrix_main(z1, z2, Math.Operations.ADDITION)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_subtraction_matrix(self):
        """Test get_matrix_main for subtraction."""
        z1 = Znum([5, 6, 7, 8], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.2, 0.3, 0.4, 0.5])

        result = Math.get_matrix_main(z1, z2, Math.Operations.SUBTRACTION)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_multiplication_matrix(self):
        """Test get_matrix_main for multiplication."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        result = Math.get_matrix_main(z1, z2, Math.Operations.MULTIPLICATION)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_division_matrix(self):
        """Test get_matrix_main for division."""
        z1 = Znum([4, 6, 8, 10], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.2, 0.3, 0.4, 0.5])

        result = Math.get_matrix_main(z1, z2, Math.Operations.DIVISION)

        assert isinstance(result, list)
        assert len(result) > 0


# =============================================================================
# Math.get_minimized_matrix Tests
# =============================================================================

class TestMathGetMinimizedMatrix:
    """Test Math.get_minimized_matrix static method."""

    def test_aggregates_duplicate_keys(self):
        """Test that minimized_matrix aggregates rows with same first column."""
        matrix = [
            [1.0, 0.5, 0.1, 0.2],
            [1.0, 0.3, 0.2, 0.3],
            [2.0, 0.6, 0.3, 0.4]
        ]
        result = Math.get_minimized_matrix(matrix)

        # Should have 2 unique keys (1.0 and 2.0)
        first_cols = [row[0] for row in result]
        assert len(set(first_cols)) == 2

    def test_takes_max_membership(self):
        """Test that max membership is taken for duplicate keys."""
        matrix = [
            [1.0, 0.5, 0.1, 0.2],
            [1.0, 0.8, 0.2, 0.3]
        ]
        result = Math.get_minimized_matrix(matrix)

        # Find the row with key 1.0
        for row in result:
            if row[0] == 1.0:
                assert row[1] == 0.8  # Max of 0.5 and 0.8


# =============================================================================
# Math.get_prob_pos Tests
# =============================================================================

class TestMathGetProbPos:
    """Test Math.get_prob_pos static method."""

    def test_returns_list(self):
        """Test that get_prob_pos returns a list."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        # Create a simplified matrix for testing
        matrix = [
            [3.0, 0.5, 0.1, 0.2, 0.15, 0.25],
            [5.0, 0.7, 0.2, 0.3, 0.25, 0.35],
            [7.0, 0.6, 0.15, 0.25, 0.2, 0.3]
        ]

        result = Math.get_prob_pos(matrix, z1, z2)

        assert isinstance(result, list)


# =============================================================================
# Math.z_solver_main Tests
# =============================================================================

class TestMathZSolverMain:
    """Test Math.z_solver_main static method."""

    def test_addition_returns_znum(self):
        """Test that z_solver_main returns Znum for addition."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        result = Math.z_solver_main(z1, z2, Math.Operations.ADDITION)

        assert isinstance(result, Znum)

    def test_subtraction_returns_znum(self):
        """Test that z_solver_main returns Znum for subtraction."""
        z1 = Znum([5, 6, 7, 8], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.2, 0.3, 0.4, 0.5])

        result = Math.z_solver_main(z1, z2, Math.Operations.SUBTRACTION)

        assert isinstance(result, Znum)

    def test_multiplication_returns_znum(self):
        """Test that z_solver_main returns Znum for multiplication."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        result = Math.z_solver_main(z1, z2, Math.Operations.MULTIPLICATION)

        assert isinstance(result, Znum)

    def test_division_returns_znum(self):
        """Test that z_solver_main returns Znum for division."""
        z1 = Znum([4, 6, 8, 10], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.2, 0.3, 0.4, 0.5])

        result = Math.z_solver_main(z1, z2, Math.Operations.DIVISION)

        assert isinstance(result, Znum)

    def test_addition_result_values(self):
        """Test that addition produces correct A bounds."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

        result = Math.z_solver_main(z1, z2, Math.Operations.ADDITION)

        # Min should be 1+2=3, Max should be 4+5=9
        assert result.A[0] == 3.0
        assert result.A[3] == 9.0

    def test_subtraction_result_values(self):
        """Test that subtraction produces correct A bounds."""
        z1 = Znum([5, 6, 7, 8], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.2, 0.3, 0.4, 0.5])

        result = Math.z_solver_main(z1, z2, Math.Operations.SUBTRACTION)

        # Min should be 5-4=1, Max should be 8-1=7
        assert result.A[0] == 1.0
        assert result.A[3] == 7.0

    def test_multiplication_result_values(self):
        """Test that multiplication produces correct A bounds."""
        z1 = Znum([2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([3, 4, 5, 6], [0.2, 0.3, 0.4, 0.5])

        result = Math.z_solver_main(z1, z2, Math.Operations.MULTIPLICATION)

        # Min should be 2*3=6, Max should be 5*6=30
        assert result.A[0] == 6.0
        assert result.A[3] == 30.0

    def test_division_result_values(self):
        """Test that division produces correct A bounds."""
        z1 = Znum([6, 8, 10, 12], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 6], [0.2, 0.3, 0.4, 0.5])

        result = Math.z_solver_main(z1, z2, Math.Operations.DIVISION)

        # Min should be 6/6=1, Max should be 12/2=6
        assert result.A[0] == 1.0
        assert result.A[3] == 6.0


# =============================================================================
# Math Constants Tests
# =============================================================================

class TestMathConstants:
    """Test Math class constants."""

    def test_method_constant(self):
        """Test METHOD constant."""
        assert Math.METHOD == "simplex"

    def test_precision_constant(self):
        """Test PRECISION constant."""
        assert Math.PRECISION == 6
