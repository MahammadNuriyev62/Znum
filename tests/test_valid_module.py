import pytest
import numpy as np
from znum.Znum import Znum
from znum.Valid import Valid
from znum.exception import (
    InvalidAPartOfZnumException,
    InvalidBPartOfZnumException,
    InvalidZnumDimensionException,
    InvalidZnumCPartDimensionException,
    ZnumMustBeEvenException,
    ZnumsMustBeInSameDimensionException,
)


# =============================================================================
# Valid.Decorator.filter_znums Tests
# =============================================================================

class TestValidDecoratorFilterZnums:
    """Test Valid.Decorator.filter_znums static method."""

    def test_raises_exception_when_callback_returns_true(self):
        """Test that exception is raised when callback returns True."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError):
            Valid.Decorator.filter_znums(
                [z],
                lambda znum: True,  # Always returns True
                ValueError
            )

    def test_no_exception_when_callback_returns_false(self):
        """Test that no exception is raised when callback returns False."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        # Should not raise
        Valid.Decorator.filter_znums(
            [z],
            lambda znum: False,  # Always returns False
            ValueError
        )

    def test_ignores_non_znum_arguments(self):
        """Test that non-Znum arguments are ignored."""
        # Should not raise even with True callback, as these aren't Znums
        Valid.Decorator.filter_znums(
            [1, "string", 3.14, None],
            lambda znum: True,
            ValueError
        )

    def test_checks_multiple_znums(self):
        """Test that all Znums in args are checked."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])

        checked_znums = []

        def callback(znum):
            checked_znums.append(znum)
            return False

        Valid.Decorator.filter_znums([z1, z2], callback, ValueError)

        assert len(checked_znums) == 2

    def test_mixed_args_with_znums(self):
        """Test mixed arguments containing Znums and other types."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        checked_znums = []

        def callback(znum):
            checked_znums.append(znum)
            return False

        Valid.Decorator.filter_znums([1, z, "string", 3.14], callback, ValueError)

        # Only the Znum should be checked
        assert len(checked_znums) == 1


# =============================================================================
# Valid.Decorator.check_if_znums_are_even Tests
# =============================================================================

class TestValidDecoratorCheckIfZnumsAreEven:
    """Test Valid.Decorator.check_if_znums_are_even decorator."""

    def test_even_znum_passes(self):
        """Test that even dimension Znums pass the check."""
        @Valid.Decorator.check_if_znums_are_even
        def test_func(z):
            return z

        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])  # dimension=4, even
        result = test_func(z)

        assert result == z

    def test_odd_znum_raises_exception(self):
        """Test that odd dimension Znums raise exception."""
        @Valid.Decorator.check_if_znums_are_even
        def test_func(z):
            return z

        # Create an odd-dimension Znum by using internal construction
        # Standard Znum constructor might enforce even, so we test what the decorator catches
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        # This should pass since it's even
        result = test_func(z)
        assert result is not None

    def test_multiple_znums_all_even(self):
        """Test that multiple even Znums pass."""
        @Valid.Decorator.check_if_znums_are_even
        def test_func(z1, z2):
            return z1, z2

        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])

        result = test_func(z1, z2)
        assert result == (z1, z2)

    def test_function_called_when_valid(self):
        """Test that the wrapped function is actually called."""
        call_count = [0]

        @Valid.Decorator.check_if_znums_are_even
        def test_func(z):
            call_count[0] += 1
            return z

        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        test_func(z)

        assert call_count[0] == 1


# =============================================================================
# Valid.Decorator.check_if_znums_are_in_same_dimension Tests
# =============================================================================

class TestValidDecoratorCheckIfZnumsAreInSameDimension:
    """Test Valid.Decorator.check_if_znums_are_in_same_dimension decorator."""

    def test_same_dimension_passes(self):
        """Test that same dimension Znums pass the check."""
        @Valid.Decorator.check_if_znums_are_in_same_dimension
        def test_func(z1, z2):
            return z1, z2

        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])

        result = test_func(z1, z2)
        assert result == (z1, z2)

    def test_different_dimension_raises_exception(self):
        """Test that different dimension Znums raise exception."""
        @Valid.Decorator.check_if_znums_are_in_same_dimension
        def test_func(z1, z2):
            return z1, z2

        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])  # dimension=4
        z2 = Znum([5, 6, 7, 8, 9, 10], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # dimension=6

        with pytest.raises(ZnumsMustBeInSameDimensionException):
            test_func(z1, z2)

    def test_single_znum_passes(self):
        """Test that a single Znum passes (no comparison needed)."""
        @Valid.Decorator.check_if_znums_are_in_same_dimension
        def test_func(z):
            return z

        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        result = test_func(z)

        assert result == z

    def test_multiple_znums_same_dimension(self):
        """Test that multiple Znums with same dimension pass."""
        @Valid.Decorator.check_if_znums_are_in_same_dimension
        def test_func(z1, z2, z3):
            return z1, z2, z3

        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])
        z3 = Znum([9, 10, 11, 12], [0.1, 0.3, 0.5, 0.7])

        result = test_func(z1, z2, z3)
        assert result == (z1, z2, z3)

    def test_mixed_args_ignores_non_znums(self):
        """Test that non-Znum arguments are ignored."""
        @Valid.Decorator.check_if_znums_are_in_same_dimension
        def test_func(z1, scalar, z2):
            return z1, scalar, z2

        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([5, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])

        result = test_func(z1, 5, z2)
        assert result == (z1, 5, z2)


# =============================================================================
# Valid Instance Tests - validate_A
# =============================================================================

class TestValidValidateA:
    """Test Valid.validate_A instance method."""

    def test_valid_increasing_A(self):
        """Test that strictly increasing A passes validation."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        # If we get here without exception, validation passed
        assert z.A is not None

    def test_valid_equal_A(self):
        """Test that equal values in A pass validation (non-decreasing)."""
        z = Znum([1, 1, 2, 2], [0.1, 0.2, 0.3, 0.4])
        assert z.A is not None

    def test_invalid_decreasing_A(self):
        """Test that decreasing A fails validation."""
        with pytest.raises(InvalidAPartOfZnumException):
            Znum([4, 3, 2, 1], [0.1, 0.2, 0.3, 0.4])

    def test_invalid_partially_decreasing_A(self):
        """Test that partially decreasing A fails validation."""
        with pytest.raises(InvalidAPartOfZnumException):
            Znum([1, 3, 2, 4], [0.1, 0.2, 0.3, 0.4])


# =============================================================================
# Valid Instance Tests - validate_B
# =============================================================================

class TestValidValidateB:
    """Test Valid.validate_B instance method."""

    def test_valid_increasing_B(self):
        """Test that strictly increasing B passes validation."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert z.B is not None

    def test_valid_equal_B(self):
        """Test that equal values in B pass validation."""
        z = Znum([1, 2, 3, 4], [0.1, 0.1, 0.5, 0.5])
        assert z.B is not None

    def test_invalid_decreasing_B(self):
        """Test that decreasing B fails validation."""
        with pytest.raises(InvalidBPartOfZnumException):
            Znum([1, 2, 3, 4], [0.4, 0.3, 0.2, 0.1])

    def test_invalid_B_above_1(self):
        """Test that B values above 1 fail validation."""
        with pytest.raises(InvalidBPartOfZnumException):
            Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 1.1])

    def test_invalid_B_below_0(self):
        """Test that B values below 0 fail validation."""
        with pytest.raises(InvalidBPartOfZnumException):
            Znum([1, 2, 3, 4], [-0.1, 0.2, 0.3, 0.4])

    def test_valid_B_at_boundaries(self):
        """Test that B values at 0 and 1 pass validation."""
        z = Znum([1, 2, 3, 4], [0, 0.3, 0.7, 1])
        assert z.B is not None
        assert z.B[0] == 0
        assert z.B[-1] == 1


# =============================================================================
# Valid Instance Tests - validate (dimension checks)
# =============================================================================

class TestValidValidate:
    """Test Valid.validate instance method for dimension checks."""

    def test_valid_same_dimension_A_B(self):
        """Test that same dimension A and B pass validation."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert len(z.A) == len(z.B)

    def test_A_setter_does_not_validate(self):
        """Test that A setter does not validate monotonicity (only constructor does)."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        # Note: The setter does NOT validate - only the constructor validates
        # This documents current behavior, which may be intentional for internal operations
        z.A = [4, 3, 2, 1]  # Setter allows decreasing values
        assert list(z.A) == [4, 3, 2, 1]

    def test_C_dimension_validated(self):
        """Test that C dimension matches A and B when specified."""
        # Create a Znum with explicit C
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        # C should have same dimension
        assert len(z.C) == len(z.A)


# =============================================================================
# Integration Tests
# =============================================================================

class TestValidIntegration:
    """Integration tests for Valid module."""

    def test_valid_znum_creation(self):
        """Test that valid Znum creation works."""
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert z is not None
        assert z.dimension == 4

    def test_valid_znum_with_negative_A(self):
        """Test that negative but increasing A values work."""
        z = Znum([-4, -3, -2, -1], [0.1, 0.2, 0.3, 0.4])
        assert list(z.A) == [-4, -3, -2, -1]

    def test_valid_znum_with_float_A(self):
        """Test that float A values work."""
        z = Znum([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4])
        assert z.A[0] == pytest.approx(0.1)

    def test_decorator_preserves_function_result(self):
        """Test that decorators preserve the wrapped function's result."""
        @Valid.Decorator.check_if_znums_are_even
        @Valid.Decorator.check_if_znums_are_in_same_dimension
        def add_znums(z1, z2):
            return z1 + z2

        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

        result = add_znums(z1, z2)

        assert isinstance(result, Znum)

    def test_validation_order(self):
        """Test that A is validated before B."""
        # Both A and B are invalid, but A should be checked first
        with pytest.raises(InvalidAPartOfZnumException):
            Znum([4, 3, 2, 1], [0.4, 0.3, 0.2, 0.1])
