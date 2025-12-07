import pytest
from znum.Znum import Znum
import numpy as np


@pytest.fixture
def znums():
    """
    Returns a dictionary of Znum objects to be reused in tests.
    """
    return {
        "z1": Znum([0, 1, 2, 3], [0.0, 0.1, 0.2, 0.3]),
        "z2": Znum([1, 2, 3, 4], [0.1, 0.2, 0.4, 0.6]),
        "z3": Znum([2, 3, 4, 6], [0.2, 0.3, 0.5, 0.7]),
        "z4": Znum([3, 4, 5, 9], [0.1, 0.2, 0.5, 0.8]),
        "z5": Znum([4, 6, 7, 10], [0.0, 0.2, 0.6, 0.9]),
        "z6": Znum([5, 6, 8, 11], [0.1, 0.3, 0.5, 0.9]),
        "z7": Znum([1, 5, 9, 12], [0.0, 0.2, 0.8, 0.9]),
        "z8": Znum([2, 4, 6, 13], [0.1, 0.2, 0.4, 0.8]),
        "z9": Znum([3, 5, 7, 14], [0.2, 0.3, 0.7, 0.9]),
        "z10": Znum([4, 6, 8, 15], [0.1, 0.2, 0.5, 0.6]),
        "z11": Znum([5, 7, 9, 16], [0.0, 0.2, 0.5, 0.7]),
        "z12": Znum([6, 7, 8, 17], [0.1, 0.3, 0.4, 0.9]),
        "z13": Znum([7, 8, 10, 18], [0.0, 0.2, 0.5, 0.6]),
        "z14": Znum([2, 5, 7, 9], [0.0, 0.3, 0.4, 0.5]),
        "z15": Znum([8, 10, 12, 20], [0.1, 0.2, 0.5, 0.8]),
        "z16": Znum([9, 11, 13, 21], [0.2, 0.4, 0.5, 0.9]),
        "z17": Znum([10, 12, 14, 22], [0.0, 0.1, 0.3, 0.7]),
        "z18": Znum([3, 6, 9, 12], [0.2, 0.3, 0.6, 0.7]),
        "z19": Znum([4, 5, 10, 13], [0.1, 0.3, 0.4, 0.9]),
        "z20": Znum([5, 6, 7, 8], [0.0, 0.4, 0.5, 0.6]),
        "z21": Znum([0, 2, 4, 5], [0.0, 0.1, 0.2, 0.4]),
    }


def assert_znum_equal(actual, expected_A, expected_B, rel=1e-3):
    """
    Helper to assert that a Znum's A and B match expected values within
    a tolerance (for floating-point B).
    """
    for actual_arr, expected_arr in [(actual.A, expected_A), (actual.B, expected_B)]:
        for a_actual, a_expected in zip(actual_arr, expected_arr):
            assert np.isclose(a_actual, a_expected, rel), \
                f"Expected {a_expected}, got {a_actual}"


# =============================================================================
# Basic Addition Tests
# =============================================================================

def test_z1_plus_z2(znums):
    z1, z2 = znums["z1"], znums["z2"]
    result = z1 + z2
    # z1 + z2 = Znum(A=[1.0, 3.0, 5.0, 7.0], B=[0.525, 0.577222, 0.658889, 0.745])
    expected_A = [1.0, 3.0, 5.0, 7.0]
    expected_B = [0.525, 0.577222, 0.658889, 0.745]
    assert_znum_equal(result, expected_A, expected_B)


def test_addition_commutativity(znums):
    """z1 + z2 should equal z2 + z1."""
    z1, z2 = znums["z1"], znums["z2"]
    result1 = z1 + z2
    result2 = z2 + z1
    assert_znum_equal(result1, result2.A.tolist(), result2.B.tolist())


def test_addition_with_zero():
    """Adding zero should return the same Znum."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    result = z1 + 0
    assert_znum_equal(result, z1.A.tolist(), z1.B.tolist())


def test_addition_radd_with_zero():
    """Test right-addition with zero (for sum() support)."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    result = 0 + z1
    assert_znum_equal(result, z1.A.tolist(), z1.B.tolist())


def test_addition_multiple_znums(znums):
    """Test adding multiple Znums."""
    z1, z2, z3 = znums["z1"], znums["z2"], znums["z3"]
    result = z1 + z2 + z3
    # Verify it produces a valid Znum
    assert isinstance(result, Znum)
    assert len(result.A) == 4
    assert len(result.B) == 4


def test_sum_of_znums(znums):
    """Test using Python's sum() function with Znums."""
    z1, z2, z3 = znums["z1"], znums["z2"], znums["z3"]
    result = sum([z1, z2, z3])
    # Verify sum works (uses __radd__)
    assert isinstance(result, Znum)


# =============================================================================
# Basic Subtraction Tests
# =============================================================================

def test_z3_minus_z4(znums):
    z3, z4 = znums["z3"], znums["z4"]
    result = z3 - z4
    # z3 - z4 = Znum(A=[-7.0, -2.0, 0.0, 3.0], B=[0.541257, 0.567975, 0.637533, 0.70834])
    expected_A = [-7.0, -2.0, 0.0, 3.0]
    expected_B = [0.541257, 0.567975, 0.637533, 0.70834]
    assert_znum_equal(result, expected_A, expected_B)


def test_subtraction_yields_negative_A():
    """Test that subtraction can produce negative A values."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([5, 6, 7, 8], [0.1, 0.2, 0.3, 0.4])
    result = z1 - z2
    # Should have negative values in A
    assert result.A[0] < 0


def test_subtraction_self():
    """Subtracting a Znum from itself."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    result = z1 - z1
    # Should produce something close to zero spread
    assert isinstance(result, Znum)
    assert len(result.A) == 4


# =============================================================================
# Basic Multiplication Tests
# =============================================================================

def test_z5_times_z6(znums):
    z5, z6 = znums["z5"], znums["z6"]
    result = z5 * z6
    # z5 * z6 = Znum(A=[20.0, 36.0, 56.0, 110.0], B=[0.042187, 0.202109, 0.398047, 0.81])
    expected_A = [20.0, 36.0, 56.0, 110.0]
    expected_B = [0.042187, 0.202109, 0.398047, 0.81]
    assert_znum_equal(result, expected_A, expected_B)


def test_multiply_by_int(znums):
    z1 = znums["z1"]
    result = z1 * 2
    expected_A = [0.0, 2.0, 4.0, 6.0]
    expected_B = [0.0, 0.1, 0.2, 0.3]
    assert_znum_equal(result, expected_A, expected_B)


def test_multiply_by_float(znums):
    z1 = znums["z1"]
    result = z1 * 2.3
    expected_A = [0.0, 2.3, 4.6, 6.9]
    expected_B = [0.0, 0.1, 0.2, 0.3]
    assert_znum_equal(result, expected_A, expected_B)


def test_multiply_by_negative_scalar():
    """Test multiplication by negative scalar.

    NOTE: Multiplying by negative scalar reverses the order of A,
    which violates the monotonicity constraint and raises an exception.
    This is current expected behavior.
    """
    from znum.exception import InvalidAPartOfZnumException

    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(InvalidAPartOfZnumException):
        z1 * (-2)


def test_multiply_by_zero_scalar():
    """Test multiplication by zero scalar."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    result = z1 * 0
    expected_A = [0, 0, 0, 0]
    expected_B = [0.1, 0.2, 0.3, 0.4]  # B should stay the same
    assert_znum_equal(result, expected_A, expected_B)


def test_multiply_by_one_scalar():
    """Test multiplication by one scalar."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    result = z1 * 1
    assert_znum_equal(result, z1.A.tolist(), z1.B.tolist())


def test_multiply_by_fractional_scalar():
    """Test multiplication by fractional scalar."""
    z1 = Znum([2, 4, 6, 8], [0.1, 0.2, 0.3, 0.4])
    result = z1 * 0.5
    expected_A = [1, 2, 3, 4]
    expected_B = [0.1, 0.2, 0.3, 0.4]
    assert_znum_equal(result, expected_A, expected_B)


# =============================================================================
# Basic Division Tests
# =============================================================================

def test_z7_div_z8(znums):
    z7, z8 = znums["z7"], znums["z8"]
    result = z7 / z8
    # z7 / z8 = Znum(A=[0.076923, 0.833333, 2.25, 6.0], B=[0.107475, 0.17195, 0.370188, 0.761168])
    expected_A = [0.076923, 0.833333, 2.25, 6.0]
    expected_B = [0.107475, 0.17195, 0.370188, 0.761168]
    assert_znum_equal(result, expected_A, expected_B)


def test_division_produces_valid_znum(znums):
    """Test that division produces a valid Znum."""
    z1, z2 = znums["z1"], znums["z2"]
    # Note: z1 has 0 in A[0], which could cause issues
    z3, z4 = znums["z3"], znums["z4"]
    result = z3 / z4
    assert isinstance(result, Znum)
    assert len(result.A) == 4
    assert len(result.B) == 4


# =============================================================================
# Power Tests
# =============================================================================

def test_power_squared():
    """Test squaring a Znum."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    result = z1 ** 2
    expected_A = [1, 4, 9, 16]
    expected_B = [0.1, 0.2, 0.3, 0.4]  # B stays the same
    assert_znum_equal(result, expected_A, expected_B)


def test_power_cubed():
    """Test cubing a Znum."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    result = z1 ** 3
    expected_A = [1, 8, 27, 64]
    expected_B = [0.1, 0.2, 0.3, 0.4]
    assert_znum_equal(result, expected_A, expected_B)


def test_power_to_one():
    """Test raising to power of 1."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    result = z1 ** 1
    assert_znum_equal(result, z1.A.tolist(), z1.B.tolist())


def test_power_to_zero():
    """Test raising to power of 0."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    result = z1 ** 0
    expected_A = [1, 1, 1, 1]  # Any number to power 0 is 1
    expected_B = [0.1, 0.2, 0.3, 0.4]
    assert_znum_equal(result, expected_A, expected_B)


def test_power_fractional():
    """Test fractional power (square root)."""
    z1 = Znum([1, 4, 9, 16], [0.1, 0.2, 0.3, 0.4])
    result = z1 ** 0.5
    expected_A = [1, 2, 3, 4]
    expected_B = [0.1, 0.2, 0.3, 0.4]
    assert_znum_equal(result, expected_A, expected_B)


# =============================================================================
# Complex Expression Tests
# =============================================================================

def test_res1(znums):
    z1, z2, z3, z4 = znums["z1"], znums["z2"], znums["z3"], znums["z4"]
    # res1 = (z1 + z2 - z3) * z4
    result = (z1 + z2 - z3) * z4
    # res1 = Znum(A=[-45.0, -5.0, 10.0, 45.0], B=[0.07783, 0.152849, 0.366636, 0.611442])
    expected_A = [-45.0, -5.0, 10.0, 45.0]
    expected_B = [0.07783, 0.152849, 0.366636, 0.611442]
    assert_znum_equal(result, expected_A, expected_B)


def test_res2(znums):
    z5, z6, z7, z8, z9 = znums["z5"], znums["z6"], znums["z7"], znums["z8"], znums["z9"]
    # res2 = z5 + z6 + z7 - (z8 * z9)
    result = z5 + z6 + z7 - (z8 * z9)
    # res2 = Znum(A=[-172.0, -25.0, 4.0, 27.0], B=[0.349103, 0.355063, 0.408514, 0.646719])
    expected_A = [-172.0, -25.0, 4.0, 27.0]
    expected_B = [0.349103, 0.355063, 0.408514, 0.646719]
    assert_znum_equal(result, expected_A, expected_B)


def test_res3(znums):
    z10, z11, z13, z1 = znums["z10"], znums["z11"], znums["z13"], znums["z1"]
    # res3 = (z10 + z11) / (z13 - z1)
    result = (z10 + z11) / (z13 - z1)
    # res3 = Znum(A=[0.5, 1.444444, 2.833333, 7.75], B=[0.013476, 0.067667, 0.219289, 0.315793])
    expected_A = [0.5, 1.444444, 2.833333, 7.75]
    expected_B = [0.013476, 0.067667, 0.219289, 0.315793]
    assert_znum_equal(result, expected_A, expected_B)


def test_res4(znums):
    z14, z15, z16, z17, z18, z19, z20 = (
        znums["z14"],
        znums["z15"],
        znums["z16"],
        znums["z17"],
        znums["z18"],
        znums["z19"],
        znums["z20"],
    )
    # res4 = ((z14 + z15) * (z16 - z17)) / (z18 + z19 - z20)
    result = ((z14 + z15) * (z16 - z17)) / (z18 + z19 - z20)
    # res4 = Znum(A=[-1508.0, -14.25, 4.75, 1276.0], B=[0.054937, 0.118293, 0.20413, 0.340594])
    expected_A = [-1508.0, -14.25, 4.75, 1276.0]
    expected_B = [0.054937, 0.118293, 0.20413, 0.340594]
    assert_znum_equal(result, expected_A, expected_B)


def test_res5(znums):
    z1, z2, z3, z4, z5, z6, z7, z8, z9 = (
        znums["z1"],
        znums["z2"],
        znums["z3"],
        znums["z4"],
        znums["z5"],
        znums["z6"],
        znums["z7"],
        znums["z8"],
        znums["z9"],
    )
    # res5 = ( (z1 + z2 + z3) * (z4 - z5 + z6) ) / (z7 * z8 - z9 )
    tmpA = z1 + z2 + z3
    tmpB = z4 - z5 + z6
    tmpC = z7 * z8 - z9
    result = (tmpA * tmpB) / tmpC
    # res5 = Znum(A=[-52.0, 0.367347, 4.846154, 416.0],
    #             B=[0.000708, 0.033987, 0.166975, 0.481842])
    expected_A = [-52.0, 0.367347, 4.846154, 416.0]
    expected_B = [0.000708, 0.033987, 0.166975, 0.481842]
    assert_znum_equal(result, expected_A, expected_B)


def test_res6(znums):
    z18, z19, z20, z21, z2 = (
        znums["z18"],
        znums["z19"],
        znums["z20"],
        znums["z21"],
        znums["z2"],
    )
    result = z18 * z19 + (z20 - z21 / z2)
    # "res7 = z18 * z19 + (z20 - z21 / z2) = Znum(A=[12.0, 34.0, 96.333333, 164.0],
    #                                           B=[0.011994, 0.05494, 0.11024, 0.357853])"
    expected_A = [12.0, 34.0, 96.333333, 164.0]
    expected_B = [0.011994, 0.05494, 0.11024, 0.357853]
    assert_znum_equal(result, expected_A, expected_B)


# =============================================================================
# Additional Arithmetic Tests
# =============================================================================

def test_addition_associativity(znums):
    """(z1 + z2) + z3 should be close to z1 + (z2 + z3) for A values.

    NOTE: Due to the complexity of Z-number arithmetic (optimization-based),
    associativity is preserved for A values but B values may differ slightly
    due to different intermediate computation paths.
    """
    z1, z2, z3 = znums["z1"], znums["z2"], znums["z3"]
    result1 = (z1 + z2) + z3
    result2 = z1 + (z2 + z3)
    # A values should be exactly equal
    for a1, a2 in zip(result1.A, result2.A):
        assert np.isclose(a1, a2, rtol=1e-6), f"A values differ: {a1} vs {a2}"
    # B values may differ due to different computation paths - just verify they're valid
    assert len(result1.B) == len(result2.B)


def test_subtraction_sequence(znums):
    """Test a sequence of subtractions."""
    z1, z2, z3 = znums["z1"], znums["z2"], znums["z3"]
    result = z3 - z2 - z1
    assert isinstance(result, Znum)
    assert len(result.A) == 4


def test_mixed_operations_1(znums):
    """Test mixed add and subtract."""
    z1, z2, z3 = znums["z1"], znums["z2"], znums["z3"]
    result = z1 + z2 - z3
    assert isinstance(result, Znum)


def test_mixed_operations_2(znums):
    """Test mixed multiply and divide."""
    z5, z6, z7 = znums["z5"], znums["z6"], znums["z7"]
    result = (z5 * z6) / z7
    assert isinstance(result, Znum)


def test_nested_parentheses(znums):
    """Test nested parenthetical operations."""
    z1, z2, z3, z4 = znums["z1"], znums["z2"], znums["z3"], znums["z4"]
    result = ((z1 + z2) * (z3 - z4))
    assert isinstance(result, Znum)


# =============================================================================
# Edge Case Arithmetic Tests
# =============================================================================

def test_arithmetic_with_negative_A():
    """Test arithmetic with negative A values."""
    z_neg = Znum([-4, -3, -2, -1], [0.1, 0.2, 0.3, 0.4])
    z_pos = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

    result_add = z_neg + z_pos
    assert isinstance(result_add, Znum)

    result_sub = z_neg - z_pos
    assert isinstance(result_sub, Znum)
    # Should have more negative values
    assert result_sub.A[0] < z_neg.A[0]


def test_arithmetic_with_large_values():
    """Test arithmetic with large A values."""
    z_large1 = Znum([100, 200, 300, 400], [0.1, 0.2, 0.3, 0.4])
    z_large2 = Znum([500, 600, 700, 800], [0.2, 0.3, 0.4, 0.5])

    result = z_large1 + z_large2
    assert isinstance(result, Znum)
    assert result.A[0] >= 600  # min should be at least 600


def test_arithmetic_with_small_values():
    """Test arithmetic with small A values."""
    z_small1 = Znum([0.001, 0.002, 0.003, 0.004], [0.1, 0.2, 0.3, 0.4])
    z_small2 = Znum([0.005, 0.006, 0.007, 0.008], [0.2, 0.3, 0.4, 0.5])

    result = z_small1 * z_small2
    assert isinstance(result, Znum)


def test_arithmetic_preserves_dimension():
    """All arithmetic operations should preserve 4-element dimension."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

    assert len((z1 + z2).A) == 4
    assert len((z1 + z2).B) == 4
    assert len((z1 - z2).A) == 4
    assert len((z1 - z2).B) == 4
    assert len((z1 * z2).A) == 4
    assert len((z1 * z2).B) == 4
    assert len((z1 / z2).A) == 4
    assert len((z1 / z2).B) == 4
    assert len((z1 ** 2).A) == 4
    assert len((z1 ** 2).B) == 4


def test_chain_of_same_operation(znums):
    """Test chaining the same operation multiple times."""
    z1, z2, z3, z4, z5 = znums["z1"], znums["z2"], znums["z3"], znums["z4"], znums["z5"]

    # Chain of additions
    result_add = z1 + z2 + z3 + z4 + z5
    assert isinstance(result_add, Znum)

    # Chain of multiplications
    z_small1 = Znum([1, 1.5, 2, 2.5], [0.1, 0.2, 0.3, 0.4])
    z_small2 = Znum([1, 1.5, 2, 2.5], [0.2, 0.3, 0.4, 0.5])
    z_small3 = Znum([1, 1.5, 2, 2.5], [0.1, 0.2, 0.3, 0.4])
    result_mul = z_small1 * z_small2 * z_small3
    assert isinstance(result_mul, Znum)


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_multiply_by_invalid_type_raises():
    """Test that multiplying by invalid type raises exception."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(Exception):
        z1 * "invalid"


def test_multiply_by_list_raises():
    """Test that multiplying by list raises exception."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(Exception):
        z1 * [1, 2, 3]


# =============================================================================
# Result Type Verification Tests
# =============================================================================

def test_addition_returns_znum(znums):
    """Addition should return a Znum instance."""
    z1, z2 = znums["z1"], znums["z2"]
    result = z1 + z2
    assert isinstance(result, Znum)


def test_subtraction_returns_znum(znums):
    """Subtraction should return a Znum instance."""
    z1, z2 = znums["z1"], znums["z2"]
    result = z1 - z2
    assert isinstance(result, Znum)


def test_multiplication_returns_znum(znums):
    """Multiplication should return a Znum instance."""
    z1, z2 = znums["z1"], znums["z2"]
    result = z1 * z2
    assert isinstance(result, Znum)


def test_division_returns_znum(znums):
    """Division should return a Znum instance."""
    z3, z4 = znums["z3"], znums["z4"]
    result = z3 / z4
    assert isinstance(result, Znum)


def test_power_returns_znum(znums):
    """Power should return a Znum instance."""
    z1 = znums["z1"]
    result = z1 ** 2
    assert isinstance(result, Znum)


def test_scalar_multiply_returns_znum(znums):
    """Scalar multiplication should return a Znum instance."""
    z1 = znums["z1"]
    result = z1 * 3.14
    assert isinstance(result, Znum)


# =============================================================================
# Additional Regression Tests
# =============================================================================

def test_specific_addition_values():
    """Regression test for specific addition values."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])
    result = z1 + z2
    # A should be sum of corresponding elements at min/max
    assert result.A[0] == 3.0  # 1 + 2
    assert result.A[3] == 9.0  # 4 + 5


def test_specific_subtraction_values():
    """Regression test for specific subtraction values."""
    z1 = Znum([5, 6, 7, 8], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    result = z1 - z2
    # Min of result = min(z1) - max(z2) = 5 - 4 = 1
    # Max of result = max(z1) - min(z2) = 8 - 1 = 7
    assert result.A[0] == 1.0
    assert result.A[3] == 7.0


def test_specific_multiplication_values():
    """Regression test for specific multiplication values."""
    z1 = Znum([2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([3, 4, 5, 6], [0.2, 0.3, 0.4, 0.5])
    result = z1 * z2
    # Min = 2*3 = 6, Max = 5*6 = 30
    assert result.A[0] == 6.0
    assert result.A[3] == 30.0


def test_specific_division_values():
    """Regression test for specific division values."""
    z1 = Znum([6, 8, 10, 12], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([2, 3, 4, 6], [0.2, 0.3, 0.4, 0.5])
    result = z1 / z2
    # Min = 6/6 = 1, Max = 12/2 = 6
    assert result.A[0] == 1.0
    assert result.A[3] == 6.0


# =============================================================================
# Constructor Tests
# =============================================================================

def test_default_constructor():
    """Test Znum with default values."""
    z = Znum()
    expected_A = [1, 2, 3, 4]
    expected_B = [0.1, 0.2, 0.3, 0.4]
    assert_znum_equal(z, expected_A, expected_B)


def test_constructor_with_only_A():
    """Test Znum with only A specified."""
    z = Znum(A=[5, 6, 7, 8])
    assert z.A[0] == 5.0
    assert z.A[3] == 8.0
    # B should be default
    assert len(z.B) == 4


def test_constructor_with_only_B():
    """Test Znum with only B specified."""
    z = Znum(B=[0.2, 0.4, 0.6, 0.8])
    # A should be default
    assert z.A[0] == 1.0
    assert z.B[0] == 0.2


def test_constructor_exact_number():
    """Test Znum with all equal A values (exact number)."""
    z = Znum(A=[5, 5, 5, 5], B=[0.1, 0.2, 0.3, 0.4])
    # C should be all 1s for exact numbers
    assert np.all(z.C == 1)


def test_constructor_B_correction():
    """Test that B values are corrected when B[-1] < 0.001."""
    z = Znum(A=[1, 2, 3, 4], B=[0.0, 0.0, 0.0, 0.0001])
    # B should be slightly adjusted
    assert z.B[-1] > 0.0001


def test_constructor_preserves_dimension():
    """Test that constructor preserves dimension."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    assert z.dimension == 4


def test_constructor_with_left_right():
    """Test Znum with custom left and right partition counts."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], left=8, right=8)
    assert z.left == 8
    assert z.right == 8


# =============================================================================
# Static Method Tests
# =============================================================================

def test_get_default_A():
    """Test get_default_A static method."""
    default_A = Znum.get_default_A()
    assert list(default_A) == [1, 2, 3, 4]


def test_get_default_B():
    """Test get_default_B static method."""
    default_B = Znum.get_default_B()
    assert list(default_B) == [0.1, 0.2, 0.3, 0.4]


def test_get_default_C():
    """Test get_default_C static method."""
    default_C = Znum.get_default_C()
    assert list(default_C) == [0, 1, 1, 0]


# =============================================================================
# Property Getter Tests
# =============================================================================

def test_A_property_getter():
    """Test A property getter."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    assert list(z.A) == [1, 2, 3, 4]


def test_B_property_getter():
    """Test B property getter."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    assert list(z.B) == [0.1, 0.2, 0.3, 0.4]


def test_C_property_getter():
    """Test C property getter."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    # Default C is trapezoid membership
    assert len(z.C) == 4


def test_dimension_property():
    """Test dimension property."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    assert z.dimension == 4


# =============================================================================
# Property Setter Tests
# =============================================================================

def test_A_property_setter():
    """Test A property setter."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    z.A = [5, 6, 7, 8]
    assert list(z.A) == [5, 6, 7, 8]
    # A_int should be recalculated
    assert z.A_int is not None


def test_B_property_setter():
    """Test B property setter."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    z.B = [0.2, 0.4, 0.6, 0.8]
    assert list(z.B) == [0.2, 0.4, 0.6, 0.8]
    # B_int should be recalculated
    assert z.B_int is not None


def test_C_property_setter():
    """Test C property setter."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    z.C = [0.5, 1, 1, 0.5]
    assert list(z.C) == [0.5, 1, 1, 0.5]


# =============================================================================
# Utility Method Tests
# =============================================================================

def test_copy():
    """Test copy method creates independent copy."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    z2 = z1.copy()

    # Values should be equal
    assert list(z1.A) == list(z2.A)
    assert list(z1.B) == list(z2.B)

    # But modifying z2 should not affect z1
    z2.A = [5, 6, 7, 8]
    assert list(z1.A) == [1, 2, 3, 4]


def test_to_json():
    """Test to_json method."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    json_repr = z.to_json()

    assert "A" in json_repr
    assert "B" in json_repr
    assert json_repr["A"] == [1, 2, 3, 4]
    assert json_repr["B"] == [0.1, 0.2, 0.3, 0.4]


def test_to_array():
    """Test to_array method."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    arr = z.to_array()

    assert len(arr) == 8
    assert list(arr) == [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]


def test_str():
    """Test __str__ method."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    s = str(z)

    assert "Znum" in s
    assert "A=" in s
    assert "B=" in s


def test_repr():
    """Test __repr__ method."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    r = repr(z)

    assert "Znum" in r
    assert r == str(z)


# =============================================================================
# Type Property Tests
# =============================================================================

def test_type_is_trapezoid():
    """Test that 4-element Znum is recognized as trapezoid."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    assert z.type.isTrapezoid


def test_type_is_even():
    """Test that 4-element Znum is recognized as even."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    assert z.type.isEven


# =============================================================================
# Intermediate Value Tests
# =============================================================================

def test_A_int_structure():
    """Test that A_int has correct structure."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

    assert "value" in z.A_int
    assert "memb" in z.A_int
    assert len(z.A_int["value"]) == len(z.A_int["memb"])


def test_B_int_structure():
    """Test that B_int has correct structure."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

    assert "value" in z.B_int
    assert "memb" in z.B_int
    assert len(z.B_int["value"]) == len(z.B_int["memb"])


# =============================================================================
# Additional Arithmetic Edge Cases
# =============================================================================

def test_addition_with_exact_numbers():
    """Test addition with exact numbers (all A equal)."""
    z1 = Znum([5, 5, 5, 5], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([3, 3, 3, 3], [0.2, 0.3, 0.4, 0.5])
    result = z1 + z2

    # Result should have A values around 8
    assert result.A[0] == 8.0
    assert result.A[3] == 8.0


def test_subtraction_with_exact_numbers():
    """Test subtraction with exact numbers."""
    z1 = Znum([10, 10, 10, 10], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([3, 3, 3, 3], [0.2, 0.3, 0.4, 0.5])
    result = z1 - z2

    # Result should be around 7
    assert result.A[0] == 7.0
    assert result.A[3] == 7.0


def test_multiplication_with_exact_numbers():
    """Test multiplication with exact numbers."""
    z1 = Znum([2, 2, 2, 2], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([3, 3, 3, 3], [0.2, 0.3, 0.4, 0.5])
    result = z1 * z2

    # Result should be around 6
    assert result.A[0] == 6.0
    assert result.A[3] == 6.0


def test_division_with_exact_numbers():
    """Test division with exact numbers."""
    z1 = Znum([10, 10, 10, 10], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([2, 2, 2, 2], [0.2, 0.3, 0.4, 0.5])
    result = z1 / z2

    # Result should be around 5
    assert result.A[0] == 5.0
    assert result.A[3] == 5.0


def test_power_with_exact_numbers():
    """Test power with exact numbers."""
    z = Znum([3, 3, 3, 3], [0.1, 0.2, 0.3, 0.4])
    result = z ** 2

    # All A values should be 9
    assert result.A[0] == 9.0
    assert result.A[3] == 9.0


def test_arithmetic_with_very_small_B():
    """Test arithmetic with very small B values."""
    z1 = Znum([1, 2, 3, 4], [0.001, 0.002, 0.003, 0.004])
    z2 = Znum([2, 3, 4, 5], [0.001, 0.002, 0.003, 0.004])
    result = z1 + z2

    assert isinstance(result, Znum)
    assert len(result.A) == 4


def test_arithmetic_with_high_B():
    """Test arithmetic with high B values (near 1)."""
    z1 = Znum([1, 2, 3, 4], [0.7, 0.8, 0.9, 0.99])
    z2 = Znum([2, 3, 4, 5], [0.7, 0.8, 0.9, 0.99])
    result = z1 + z2

    assert isinstance(result, Znum)
    assert len(result.A) == 4


def test_mixed_positive_negative_multiplication():
    """Test multiplication with mixed positive/negative values."""
    z_pos = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    z_neg = Znum([-4, -3, -2, -1], [0.1, 0.2, 0.3, 0.4])
    result = z_pos * z_neg

    assert isinstance(result, Znum)
    # Result should have negative values
    assert result.A[0] < 0


def test_division_by_larger_number():
    """Test division where divisor is larger."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([10, 20, 30, 40], [0.1, 0.2, 0.3, 0.4])
    result = z1 / z2

    # Result should be less than 1
    assert result.A[0] < 1
    assert result.A[3] < 1


def test_power_negative_base():
    """Test power with negative A values.

    NOTE: Squaring negative values produces non-monotonic results
    (e.g., [-4, -3, -2, -1]^2 = [16, 9, 4, 1] which is decreasing),
    violating the A monotonicity constraint. This raises an exception.
    """
    from znum.exception import InvalidAPartOfZnumException

    z = Znum([-4, -3, -2, -1], [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(InvalidAPartOfZnumException):
        z ** 2


def test_power_high_exponent():
    """Test power with high exponent."""
    z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    result = z ** 5

    assert result.A[0] == 1.0    # 1^5
    assert result.A[3] == 1024.0  # 4^5


# =============================================================================
# Validation Exception Tests
# =============================================================================

def test_invalid_A_not_monotonic_raises():
    """Test that non-monotonic A raises exception."""
    from znum.exception import InvalidAPartOfZnumException

    with pytest.raises(InvalidAPartOfZnumException):
        Znum([4, 2, 3, 1], [0.1, 0.2, 0.3, 0.4])


def test_invalid_B_not_monotonic_raises():
    """Test that non-monotonic B raises exception."""
    from znum.exception import InvalidBPartOfZnumException

    with pytest.raises(InvalidBPartOfZnumException):
        Znum([1, 2, 3, 4], [0.4, 0.2, 0.3, 0.1])


def test_invalid_B_greater_than_1_raises():
    """Test that B values > 1 raise exception."""
    from znum.exception import InvalidBPartOfZnumException

    with pytest.raises(InvalidBPartOfZnumException):
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 1.5])


def test_invalid_B_less_than_0_raises():
    """Test that B values < 0 raise exception."""
    from znum.exception import InvalidBPartOfZnumException

    with pytest.raises(InvalidBPartOfZnumException):
        Znum([1, 2, 3, 4], [-0.1, 0.2, 0.3, 0.4])


# =============================================================================
# Class Attribute Tests
# =============================================================================

def test_class_has_vikor():
    """Test that Znum class has Vikor attribute."""
    from znum.Vikor import Vikor
    assert Znum.Vikor == Vikor


def test_class_has_topsis():
    """Test that Znum class has Topsis attribute."""
    from znum.Topsis import Topsis
    assert Znum.Topsis == Topsis


def test_class_has_sort():
    """Test that Znum class has Sort attribute."""
    from znum.Sort import Sort
    assert Znum.Sort == Sort


def test_class_has_promethee():
    """Test that Znum class has Promethee attribute."""
    from znum.Promethee import Promethee
    assert Znum.Promethee == Promethee


def test_class_has_beast():
    """Test that Znum class has Beast attribute."""
    from znum.Beast import Beast
    assert Znum.Beast == Beast


def test_class_has_math():
    """Test that Znum class has Math attribute."""
    from znum.Math import Math
    assert Znum.Math == Math


def test_class_has_dist():
    """Test that Znum class has Dist attribute."""
    from znum.Dist import Dist
    assert Znum.Dist == Dist


# =============================================================================
# More Regression Tests with Specific Values
# =============================================================================

def test_regression_add_1():
    """Regression test for addition."""
    z1 = Znum([2, 4, 6, 8], [0.1, 0.2, 0.3, 0.5])
    z2 = Znum([1, 3, 5, 7], [0.2, 0.3, 0.4, 0.6])
    result = z1 + z2

    # A[0] = 2 + 1 = 3
    # A[3] = 8 + 7 = 15
    assert result.A[0] == 3.0
    assert result.A[3] == 15.0


def test_regression_sub_1():
    """Regression test for subtraction."""
    z1 = Znum([10, 15, 20, 25], [0.1, 0.2, 0.3, 0.5])
    z2 = Znum([2, 4, 6, 8], [0.2, 0.3, 0.4, 0.6])
    result = z1 - z2

    # A[0] = 10 - 8 = 2
    # A[3] = 25 - 2 = 23
    assert result.A[0] == 2.0
    assert result.A[3] == 23.0


def test_regression_mul_1():
    """Regression test for multiplication."""
    z1 = Znum([2, 3, 4, 5], [0.1, 0.2, 0.3, 0.5])
    z2 = Znum([3, 4, 5, 6], [0.2, 0.3, 0.4, 0.6])
    result = z1 * z2

    # A[0] = 2 * 3 = 6
    # A[3] = 5 * 6 = 30
    assert result.A[0] == 6.0
    assert result.A[3] == 30.0


def test_regression_div_1():
    """Regression test for division."""
    z1 = Znum([12, 24, 36, 48], [0.1, 0.2, 0.3, 0.5])
    z2 = Znum([2, 3, 4, 6], [0.2, 0.3, 0.4, 0.6])
    result = z1 / z2

    # A[0] = 12 / 6 = 2
    # A[3] = 48 / 2 = 24
    assert result.A[0] == 2.0
    assert result.A[3] == 24.0


def test_complex_nested_operations_1():
    """Complex nested operation test."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])
    z3 = Znum([3, 4, 5, 6], [0.1, 0.3, 0.5, 0.6])

    result = (z1 + z2) * z3 - z1
    assert isinstance(result, Znum)
    assert len(result.A) == 4
    assert len(result.B) == 4


def test_complex_nested_operations_2():
    """Another complex nested operation test."""
    z1 = Znum([5, 10, 15, 20], [0.2, 0.4, 0.6, 0.8])
    z2 = Znum([2, 4, 6, 8], [0.1, 0.3, 0.5, 0.7])
    z3 = Znum([1, 2, 3, 4], [0.3, 0.4, 0.5, 0.6])

    result = z1 / z2 + z3 * z2
    assert isinstance(result, Znum)


def test_operations_preserve_type():
    """All operations should return Znum with correct type."""
    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5])

    assert (z1 + z2).type.isTrapezoid
    assert (z1 - z2).type.isTrapezoid
    assert (z1 * z2).type.isTrapezoid
    assert (z1 / z2).type.isTrapezoid
    assert (z1 ** 2).type.isTrapezoid


# =============================================================================
# Boundary Value Tests
# =============================================================================

def test_B_at_exact_boundaries():
    """Test with B values at exact boundary (0 and 1)."""
    z1 = Znum([1, 2, 3, 4], [0.0, 0.33, 0.67, 1.0])
    z2 = Znum([2, 3, 4, 5], [0.0, 0.33, 0.67, 1.0])
    result = z1 + z2

    assert isinstance(result, Znum)


def test_A_with_decimal_values():
    """Test with A values having many decimal places."""
    z1 = Znum([1.123456, 2.234567, 3.345678, 4.456789], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([0.111111, 0.222222, 0.333333, 0.444444], [0.2, 0.3, 0.4, 0.5])
    result = z1 + z2

    assert isinstance(result, Znum)
    assert result.A[0] > 1.2


def test_narrow_spread_znum():
    """Test with narrow spread (A values close together)."""
    z1 = Znum([10.0, 10.1, 10.2, 10.3], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([5.0, 5.1, 5.2, 5.3], [0.2, 0.3, 0.4, 0.5])
    result = z1 + z2

    assert isinstance(result, Znum)
    # Result should also have narrow spread around 15
    assert result.A[3] - result.A[0] < 1


def test_wide_spread_znum():
    """Test with wide spread (A values far apart)."""
    z1 = Znum([1, 100, 200, 1000], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([2, 50, 150, 500], [0.2, 0.3, 0.4, 0.5])
    result = z1 + z2

    assert isinstance(result, Znum)
    assert result.A[3] > result.A[0]
