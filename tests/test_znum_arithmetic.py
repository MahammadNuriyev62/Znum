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


def assert_znum_equal(actual, expected_A, expected_B, rel=1e-10):
    """
    Helper to assert that a Znum's A and B match expected values within
    a tolerance (for floating-point B).
    """
    for actual, expected in [(actual.A, expected_A), (actual.B, expected_B)]:
        assert np.linalg.norm(actual - expected) < rel


def test_z1_plus_z2(znums):
    z1, z2 = znums["z1"], znums["z2"]
    result = z1 + z2
    # z1 + z2 = Znum(A=[1.0, 3.0, 5.0, 7.0], B=[0.525, 0.577222, 0.658889, 0.745])
    expected_A = [1.0, 3.0, 5.0, 7.0]
    expected_B = [0.525, 0.577222, 0.658889, 0.745]
    assert_znum_equal(result, expected_A, expected_B)


def test_z3_minus_z4(znums):
    z3, z4 = znums["z3"], znums["z4"]
    result = z3 - z4
    # z3 - z4 = Znum(A=[-7.0, -2.0, 0.0, 3.0], B=[0.541257, 0.567975, 0.637533, 0.70834])
    expected_A = [-7.0, -2.0, 0.0, 3.0]
    expected_B = [0.541257, 0.567975, 0.637533, 0.70834]
    assert_znum_equal(result, expected_A, expected_B)


def test_z5_times_z6(znums):
    z5, z6 = znums["z5"], znums["z6"]
    result = z5 * z6
    # z5 * z6 = Znum(A=[20.0, 36.0, 56.0, 110.0], B=[0.042187, 0.202109, 0.398047, 0.81])
    expected_A = [20.0, 36.0, 56.0, 110.0]
    expected_B = [0.042187, 0.202109, 0.398047, 0.81]
    assert_znum_equal(result, expected_A, expected_B)


def test_z7_div_z8(znums):
    z7, z8 = znums["z7"], znums["z8"]
    result = z7 / z8
    # z7 / z8 = Znum(A=[0.076923, 0.833333, 2.25, 6.0], B=[0.107475, 0.17195, 0.370188, 0.761168])
    expected_A = [0.076923, 0.833333, 2.25, 6.0]
    expected_B = [0.107475, 0.17195, 0.370188, 0.761168]
    assert_znum_equal(result, expected_A, expected_B)


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
