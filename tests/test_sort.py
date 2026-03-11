"""
Tests for Sort comparison module (sort.py).

Covers: normalize, NxF possibility/value computation, overlapping fuzzy
comparison, and edge cases.
"""

import pytest

from znum import Znum
from znum.sort import Sort


class TestNormalize:
    """Tests for Sort._normalize."""

    def test_equal_values_return_zeros(self):
        """When all values are the same, normalization returns all zeros."""
        q1 = [5, 5, 5, 5]
        q2 = [5, 5, 5, 5]
        n1, n2 = Sort._normalize(q1, q2)
        assert n1 == [0, 0, 0, 0]
        assert n2 == [0, 0, 0, 0]

    def test_basic_normalization(self):
        """Values should be scaled to [0, 1]."""
        q1 = [0, 10]
        q2 = [5, 5]
        n1, n2 = Sort._normalize(q1, q2)
        assert n1[0] == pytest.approx(0.0)
        assert n1[1] == pytest.approx(1.0)
        assert n2[0] == pytest.approx(0.5)
        assert n2[1] == pytest.approx(0.5)

    def test_preserves_lengths(self):
        q1 = [1, 2, 3, 4]
        q2 = [5, 6, 7, 8]
        n1, n2 = Sort._normalize(q1, q2)
        assert len(n1) == 4
        assert len(n2) == 4

    def test_all_in_unit_interval(self):
        q1 = [1, 2, 3, 4]
        q2 = [5, 6, 7, 8]
        n1, n2 = Sort._normalize(q1, q2)
        for v in n1 + n2:
            assert 0 <= v <= 1


class TestOverlappingFuzzyComparison:
    """Tests for comparing overlapping fuzzy Z-numbers."""

    def test_clearly_greater(self):
        """Z-number with clearly higher A values should be greater."""
        z1 = Znum(A=[7, 8, 9, 10], B=[0.5, 0.6, 0.7, 0.8])
        z2 = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        assert z1 > z2
        assert z2 < z1

    def test_overlapping_greater(self):
        """Partially overlapping Z-numbers: [3,4,5,6] vs [1,2,3,4]."""
        z1 = Znum(A=[3, 4, 5, 6], B=[0.5, 0.6, 0.7, 0.8])
        z2 = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        assert z1 > z2

    def test_heavily_overlapping(self):
        """Heavily overlapping: [2,3,4,5] vs [1,2,3,4] — still ordered."""
        z1 = Znum(A=[2, 3, 4, 5], B=[0.5, 0.6, 0.7, 0.8])
        z2 = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        assert z1 > z2

    def test_identical_are_equal(self):
        z1 = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        z2 = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        assert z1 == z2

    def test_same_A_different_B(self):
        """Same A but different B — comparison should still work."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.7, 0.8, 0.9, 1.0])
        z2 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        # With same A, B difference drives the comparison
        d1, do1 = Sort.solver_main(z1, z2)
        d2, do2 = Sort.solver_main(z2, z1)
        # At minimum, the comparison should not crash and should return valid values
        assert 0 <= d1 <= 1
        assert 0 <= do1 <= 1
        assert d1 + do1 == pytest.approx(1.0)

    def test_comparison_consistency(self):
        """If z1 > z2, then z2 < z1 and not z1 < z2."""
        z1 = Znum(A=[5, 6, 7, 8], B=[0.5, 0.6, 0.7, 0.8])
        z2 = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        assert z1 > z2
        assert z2 < z1
        assert not z1 < z2
        assert not z2 > z1

    def test_ge_le_consistency(self):
        z1 = Znum(A=[5, 6, 7, 8], B=[0.5, 0.6, 0.7, 0.8])
        z2 = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        assert z1 >= z2
        assert z2 <= z1

    def test_equal_implies_ge_and_le(self):
        z1 = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        z2 = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        assert z1 >= z2
        assert z1 <= z2


class TestSolverMainScores:
    """Tests for the raw (d, do) scores from Sort.solver_main."""

    def test_returns_tuple_of_two(self):
        z1 = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        z2 = Znum(A=[5, 6, 7, 8], B=[0.5, 0.6, 0.7, 0.8])
        result = Sort.solver_main(z1, z2)
        assert len(result) == 2

    def test_d_plus_do_equals_one(self):
        z1 = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        z2 = Znum(A=[5, 6, 7, 8], B=[0.5, 0.6, 0.7, 0.8])
        d, do = Sort.solver_main(z1, z2)
        assert d + do == pytest.approx(1.0)

    def test_self_comparison(self):
        """Comparing identical Z-numbers should give do == 1."""
        z = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        d, do = Sort.solver_main(z, z)
        assert do == pytest.approx(1.0)

    def test_dominance_direction(self):
        """Larger Z-number should have higher do when compared to smaller."""
        z_big = Znum(A=[10, 20, 30, 40], B=[0.5, 0.6, 0.7, 0.8])
        z_small = Znum(A=[1, 2, 3, 4], B=[0.5, 0.6, 0.7, 0.8])
        _, do_big_vs_small = Sort.solver_main(z_big, z_small)
        _, do_small_vs_big = Sort.solver_main(z_small, z_big)
        assert do_big_vs_small > do_small_vs_big


class TestNxFPossibility:
    """Tests for the NxF possibility measure."""

    def test_all_branches_reachable(self):
        """Different intermediate values should exercise all 4 branches."""
        # Case: large positive difference (worse region)
        inter_positive = [0.5, 0.6, 0.7, 0.8]
        result = Sort._nxf_possibility(inter_positive, "nwF")
        assert 0 <= result <= 1

        # Case: large negative difference (better region)
        inter_negative = [-0.8, -0.7, -0.6, -0.5]
        result = Sort._nxf_possibility(inter_negative, "nbF")
        assert 0 <= result <= 1

        # Case: near-zero difference (equal region)
        inter_zero = [-0.05, 0.0, 0.0, 0.05]
        result = Sort._nxf_possibility(inter_zero, "neF")
        assert 0 <= result <= 1

    def test_possibility_bounded(self):
        """All possibility values should be in [0, 1]."""
        intermediates = [
            [0.1, 0.2, 0.3, 0.4],
            [-0.4, -0.3, -0.2, -0.1],
            [-0.1, 0.0, 0.0, 0.1],
        ]
        for inter in intermediates:
            for option in ["nbF", "neF", "nwF"]:
                result = Sort._nxf_possibility(inter, option)
                assert 0 <= result <= 1, f"Failed for {inter}, {option}: {result}"
