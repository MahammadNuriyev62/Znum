import pytest
import numpy as np
from znum.Znum import Znum


# =============================================================================
# Construction Tests
# =============================================================================

class TestZnumConstruction:
    """Test Znum construction and initialization."""

    def test_default_construction(self):
        """Test construction with default values."""
        z = Znum()
        assert len(z.A) == 4
        assert len(z.B) == 4
        assert len(z.C) == 4
        np.testing.assert_array_equal(z.A, [1, 2, 3, 4])
        np.testing.assert_array_almost_equal(z.B, [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_equal(z.C, [0, 1, 1, 0])

    def test_construction_with_A_only(self):
        """Test construction with only A provided."""
        z = Znum(A=[5, 6, 7, 8])
        np.testing.assert_array_equal(z.A, [5, 6, 7, 8])
        # B should be default
        np.testing.assert_array_almost_equal(z.B, [0.1, 0.2, 0.3, 0.4])

    def test_construction_with_B_only(self):
        """Test construction with only B provided."""
        z = Znum(B=[0.2, 0.4, 0.6, 0.8])
        # A should be default
        np.testing.assert_array_equal(z.A, [1, 2, 3, 4])
        np.testing.assert_array_almost_equal(z.B, [0.2, 0.4, 0.6, 0.8])

    def test_construction_with_A_and_B(self):
        """Test construction with both A and B provided."""
        z = Znum(A=[10, 20, 30, 40], B=[0.1, 0.3, 0.5, 0.7])
        np.testing.assert_array_equal(z.A, [10, 20, 30, 40])
        np.testing.assert_array_almost_equal(z.B, [0.1, 0.3, 0.5, 0.7])

    def test_construction_with_lists(self):
        """Test construction with Python lists."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert isinstance(z.A, np.ndarray)
        assert isinstance(z.B, np.ndarray)

    def test_construction_with_numpy_arrays(self):
        """Test construction with numpy arrays."""
        a = np.array([1, 2, 3, 4])
        b = np.array([0.1, 0.2, 0.3, 0.4])
        z = Znum(A=a, B=b)
        np.testing.assert_array_equal(z.A, a)

    def test_construction_with_floats(self):
        """Test construction with float values."""
        z = Znum([1.5, 2.5, 3.5, 4.5], [0.15, 0.25, 0.35, 0.45])
        assert z.A[0] == 1.5
        assert z.B[0] == 0.15

    def test_construction_with_negative_A(self):
        """Test construction with negative A values."""
        z = Znum([-4, -3, -2, -1], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_equal(z.A, [-4, -3, -2, -1])

    def test_construction_with_zero_A(self):
        """Test construction with zero in A."""
        z = Znum([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
        assert z.A[0] == 0

    def test_construction_with_equal_A_elements(self):
        """Test construction when all A elements are equal (exact number)."""
        z = Znum([5, 5, 5, 5], [0.1, 0.2, 0.3, 0.4])
        # When all A are equal, C should be all ones
        np.testing.assert_array_equal(z.C, [1, 1, 1, 1])

    def test_construction_with_very_small_B(self):
        """Test construction with very small B values."""
        z = Znum([1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0])
        # B should be adjusted slightly to avoid issues
        assert z.B[-1] > 0


# =============================================================================
# Property Tests
# =============================================================================

class TestZnumProperties:
    """Test Znum property getters and setters."""

    def test_dimension_property(self):
        """Test dimension property."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert z.dimension == 4

    def test_A_property_getter(self):
        """Test A property getter."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert isinstance(z.A, np.ndarray)
        np.testing.assert_array_equal(z.A, [1, 2, 3, 4])

    def test_B_property_getter(self):
        """Test B property getter."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert isinstance(z.B, np.ndarray)
        np.testing.assert_array_almost_equal(z.B, [0.1, 0.2, 0.3, 0.4])

    def test_C_property_getter(self):
        """Test C property getter."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert isinstance(z.C, np.ndarray)

    def test_A_property_setter(self):
        """Test A property setter."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z.A = [5, 6, 7, 8]
        np.testing.assert_array_equal(z.A, [5, 6, 7, 8])

    def test_B_property_setter(self):
        """Test B property setter."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z.B = [0.5, 0.6, 0.7, 0.8]
        np.testing.assert_array_almost_equal(z.B, [0.5, 0.6, 0.7, 0.8])

    def test_C_property_setter(self):
        """Test C property setter."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z.C = [0.5, 1, 1, 0.5]
        np.testing.assert_array_equal(z.C, [0.5, 1, 1, 0.5])


# =============================================================================
# Copy Tests
# =============================================================================

class TestZnumCopy:
    """Test Znum copy functionality."""

    def test_copy_creates_new_instance(self):
        """Test that copy creates a new Znum instance."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = z1.copy()
        assert z1 is not z2

    def test_copy_has_same_values(self):
        """Test that copy has the same A and B values."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = z1.copy()
        np.testing.assert_array_equal(z1.A, z2.A)
        np.testing.assert_array_equal(z1.B, z2.B)

    def test_copy_is_independent(self):
        """Test that modifying copy doesn't affect original."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = z1.copy()
        z2.A = [5, 6, 7, 8]
        # Original should be unchanged
        np.testing.assert_array_equal(z1.A, [1, 2, 3, 4])

    def test_copy_arrays_are_independent(self):
        """Test that copy's arrays are independent from original."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = z1.copy()
        z2.A[0] = 100  # Modify element directly
        # Original should be unchanged
        assert z1.A[0] == 1


# =============================================================================
# Serialization Tests (to_json, to_array)
# =============================================================================

class TestZnumSerialization:
    """Test Znum serialization methods."""

    def test_to_json_returns_dict(self):
        """Test that to_json returns a dictionary."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = z.to_json()
        assert isinstance(result, dict)

    def test_to_json_has_A_key(self):
        """Test that to_json result has 'A' key."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = z.to_json()
        assert "A" in result

    def test_to_json_has_B_key(self):
        """Test that to_json result has 'B' key."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = z.to_json()
        assert "B" in result

    def test_to_json_A_is_list(self):
        """Test that to_json A is a list (not numpy array)."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = z.to_json()
        assert isinstance(result["A"], list)

    def test_to_json_B_is_list(self):
        """Test that to_json B is a list (not numpy array)."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = z.to_json()
        assert isinstance(result["B"], list)

    def test_to_json_preserves_values(self):
        """Test that to_json preserves A and B values."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = z.to_json()
        assert result["A"] == [1.0, 2.0, 3.0, 4.0]
        assert result["B"] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_to_array_returns_ndarray(self):
        """Test that to_array returns a numpy array."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = z.to_array()
        assert isinstance(result, np.ndarray)

    def test_to_array_length(self):
        """Test that to_array returns concatenated A and B."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = z.to_array()
        assert len(result) == 8  # 4 + 4

    def test_to_array_values(self):
        """Test that to_array has correct values."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = z.to_array()
        expected = np.array([1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# String Representation Tests
# =============================================================================

class TestZnumStringRepresentation:
    """Test Znum string representation methods."""

    def test_str_returns_string(self):
        """Test that __str__ returns a string."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = str(z)
        assert isinstance(result, str)

    def test_str_contains_znum(self):
        """Test that __str__ contains 'Znum'."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = str(z)
        assert "Znum" in result

    def test_str_contains_A(self):
        """Test that __str__ contains 'A='."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = str(z)
        assert "A=" in result

    def test_str_contains_B(self):
        """Test that __str__ contains 'B='."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = str(z)
        assert "B=" in result

    def test_repr_returns_string(self):
        """Test that __repr__ returns a string."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = repr(z)
        assert isinstance(result, str)

    def test_repr_equals_str(self):
        """Test that __repr__ equals __str__."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert repr(z) == str(z)


# =============================================================================
# Static Method Tests
# =============================================================================

class TestZnumStaticMethods:
    """Test Znum static methods."""

    def test_get_default_A(self):
        """Test get_default_A static method."""
        result = Znum.get_default_A()
        np.testing.assert_array_equal(result, [1, 2, 3, 4])

    def test_get_default_B(self):
        """Test get_default_B static method."""
        result = Znum.get_default_B()
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3, 0.4])

    def test_get_default_C(self):
        """Test get_default_C static method."""
        result = Znum.get_default_C()
        np.testing.assert_array_equal(result, [0, 1, 1, 0])


# =============================================================================
# Class Attribute Tests
# =============================================================================

class TestZnumClassAttributes:
    """Test Znum class attributes."""

    def test_has_vikor_attribute(self):
        """Test that Znum has Vikor class attribute."""
        assert hasattr(Znum, "Vikor")

    def test_has_topsis_attribute(self):
        """Test that Znum has Topsis class attribute."""
        assert hasattr(Znum, "Topsis")

    def test_has_sort_attribute(self):
        """Test that Znum has Sort class attribute."""
        assert hasattr(Znum, "Sort")

    def test_has_promethee_attribute(self):
        """Test that Znum has Promethee class attribute."""
        assert hasattr(Znum, "Promethee")

    def test_has_beast_attribute(self):
        """Test that Znum has Beast class attribute."""
        assert hasattr(Znum, "Beast")

    def test_has_math_attribute(self):
        """Test that Znum has Math class attribute."""
        assert hasattr(Znum, "Math")

    def test_has_dist_attribute(self):
        """Test that Znum has Dist class attribute."""
        assert hasattr(Znum, "Dist")


# =============================================================================
# Instance Attribute Tests
# =============================================================================

class TestZnumInstanceAttributes:
    """Test Znum instance attributes."""

    def test_has_math_instance(self):
        """Test that Znum instance has math attribute."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert hasattr(z, "math")

    def test_has_valid_instance(self):
        """Test that Znum instance has valid attribute."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert hasattr(z, "valid")

    def test_has_type_instance(self):
        """Test that Znum instance has type attribute."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert hasattr(z, "type")

    def test_has_A_int(self):
        """Test that Znum instance has A_int attribute."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert hasattr(z, "A_int")

    def test_has_B_int(self):
        """Test that Znum instance has B_int attribute."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert hasattr(z, "B_int")

    def test_has_left_right(self):
        """Test that Znum instance has left and right attributes."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert hasattr(z, "left")
        assert hasattr(z, "right")
        assert z.left == 4
        assert z.right == 4


# =============================================================================
# Edge Cases and Special Values Tests
# =============================================================================

class TestZnumEdgeCases:
    """Test edge cases and special values."""

    def test_very_large_A_values(self):
        """Test with very large A values."""
        z = Znum([1e6, 2e6, 3e6, 4e6], [0.1, 0.2, 0.3, 0.4])
        assert z.A[0] == 1e6
        assert z.A[3] == 4e6

    def test_very_small_A_values(self):
        """Test with very small A values."""
        z = Znum([1e-6, 2e-6, 3e-6, 4e-6], [0.1, 0.2, 0.3, 0.4])
        assert z.A[0] == pytest.approx(1e-6)
        assert z.A[3] == pytest.approx(4e-6)

    def test_mixed_positive_negative_A(self):
        """Test with mixed positive and negative A values."""
        z = Znum([-2, -1, 1, 2], [0.1, 0.2, 0.3, 0.4])
        assert z.A[0] == -2
        assert z.A[3] == 2

    def test_B_values_at_boundaries(self):
        """Test with B values at 0 and 1 boundaries."""
        z = Znum([1, 2, 3, 4], [0.001, 0.5, 0.5, 0.999])
        assert z.B[0] >= 0
        assert z.B[3] <= 1

    def test_equal_middle_A_values(self):
        """Test with equal middle A values (triangle-like)."""
        z = Znum([1, 3, 3, 5], [0.1, 0.2, 0.3, 0.4])
        assert z.A[1] == z.A[2]

    def test_znum_with_custom_left_right(self):
        """Test Znum with custom left and right discretization."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], left=8, right=8)
        assert z.left == 8
        assert z.right == 8


# =============================================================================
# Integration Tests
# =============================================================================

class TestZnumIntegration:
    """Integration tests combining multiple features."""

    def test_copy_and_modify(self):
        """Test copying and modifying a Znum."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = z1.copy()
        z2.A = [5, 6, 7, 8]

        # Verify z1 unchanged, z2 modified
        np.testing.assert_array_equal(z1.A, [1, 2, 3, 4])
        np.testing.assert_array_equal(z2.A, [5, 6, 7, 8])

    def test_arithmetic_then_serialize(self):
        """Test arithmetic operation followed by serialization."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])
        result = z1 + z2

        # Serialize and verify
        json_result = result.to_json()
        assert "A" in json_result
        assert "B" in json_result
        assert json_result["A"][0] == 3.0  # 1 + 2

    def test_comparison_after_arithmetic(self):
        """Test comparison after arithmetic operation."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])
        z3 = z1 + z2

        # z3 should be greater than both z1 and z2
        assert z3 > z1
        assert z3 > z2

    def test_sum_multiple_znums_then_compare(self):
        """Test summing multiple Znums and comparing."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z3 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        total = sum([z1, z2, z3])

        assert isinstance(total, Znum)
        assert total > z1
        assert total > z2
        assert total > z3
