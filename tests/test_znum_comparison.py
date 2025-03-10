import pytest
from znum.Znum import Znum


@pytest.fixture
def znums():
    """
    Returns a dictionary of Znum objects to be reused in tests.
    """
    return {
        "z1": Znum([0, 1, 2, 3], [0.0, 0.1, 0.2, 0.3]),
        "z2": Znum([1, 2, 3, 4], [0.1, 0.2, 0.4, 0.6]),
        "z3": Znum([2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5]),
        "z4": Znum([2, 3, 4, 5], [0.4, 0.6, 0.8, 0.9]),
    }


def test_z4_is_greater_than_z3_than_z2_than_z1(znums):
    z1, z2, z3, z4 = znums["z1"], znums["z2"], znums["z3"], znums["z4"]
    assert z4 > z3 > z2 > z1
