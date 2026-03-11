"""
Unit tests for Math.get_intermediate.

Verifies that the intermediate value/membership representation matches
the mathematically exact answer (computed via rational arithmetic) to
within 1 ULP (~2.2e-16 for float64).
"""

from fractions import Fraction

import numpy as np
import pytest

from znum import Znum

# Maximum acceptable error: 1 ULP for float64
_ATOL = np.finfo(np.float64).eps  # 2.22e-16


def _exact_intermediate(Q, C, left, right):
    """Compute exact intermediate values using rational arithmetic."""
    Qf = [Fraction(x).limit_denominator(10**15) for x in Q]
    Cf = [Fraction(x).limit_denominator(10**15) for x in C]

    step_l = (Qf[1] - Qf[0]) / left if Qf[0] != Qf[1] else Fraction(0)
    step_r = (Qf[3] - Qf[2]) / right if Qf[2] != Qf[3] else Fraction(0)

    values = [Qf[0] + i * step_l for i in range(left + 1)]
    values += [Qf[2] + i * step_r for i in range(right + 1)]

    left_membs = [Cf[0] + i * (Cf[1] - Cf[0]) / left for i in range(left + 1)]
    right_membs = [Cf[2] + i * (Cf[3] - Cf[2]) / right for i in range(right + 1)]
    membs = left_membs + right_membs
    return [float(v) for v in values], [float(m) for m in membs]


# Test cases: (A, B, left, right) covering various scales
CASES = [
    pytest.param([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], 4, 4, id="integer"),
    pytest.param([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], 4, 4, id="fractional"),
    pytest.param([100, 200, 300, 400], [0.1, 0.2, 0.3, 0.4], 4, 4, id="large"),
    pytest.param([-4, -3, -2, -1], [0.1, 0.2, 0.3, 0.4], 4, 4, id="negative"),
    pytest.param([0, 0, 1, 2], [0.1, 0.2, 0.3, 0.4], 4, 4, id="zero_left"),
    pytest.param([1, 2, 3, 3], [0.1, 0.2, 0.3, 0.4], 4, 4, id="zero_right"),
    pytest.param([5, 5, 5, 5], [0.1, 0.2, 0.3, 0.4], 4, 4, id="crisp"),
    pytest.param([1, 3, 7, 10], [0.1, 0.2, 0.3, 0.4], 4, 4, id="asymmetric"),
    pytest.param([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], 2, 2, id="coarse"),
    pytest.param([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], 8, 8, id="fine"),
    pytest.param([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], 3, 5, id="asymmetric_lr"),
]


class TestGetIntermediateValues:
    """Intermediate x-values must match exact rational computation."""

    @pytest.mark.parametrize("A,B,left,right", CASES)
    def test_values_near_exact(self, A, B, left, right):
        z = Znum(A=A, B=B, left=left, right=right)
        result = z.math.get_intermediate(z.A, z.mu_A)
        exact_vals, _ = _exact_intermediate(A, z.mu_A.tolist(), left, right)
        np.testing.assert_allclose(result["value"], exact_vals, atol=_ATOL)

    @pytest.mark.parametrize("A,B,left,right", CASES)
    def test_value_count(self, A, B, left, right):
        z = Znum(A=A, B=B, left=left, right=right)
        result = z.math.get_intermediate(z.A, z.mu_A)
        assert len(result["value"]) == left + right + 2

    @pytest.mark.parametrize("A,B,left,right", CASES)
    def test_value_endpoints(self, A, B, left, right):
        """First value must be A[0], last of left must be A[1], etc."""
        z = Znum(A=A, B=B, left=left, right=right)
        result = z.math.get_intermediate(z.A, z.mu_A)
        vals = result["value"]
        assert vals[0] == pytest.approx(A[0], abs=_ATOL)
        assert vals[left] == pytest.approx(A[1], abs=_ATOL)
        assert vals[left + 1] == pytest.approx(A[2], abs=_ATOL)
        assert vals[-1] == pytest.approx(A[3], abs=_ATOL)


class TestGetIntermediateMemberships:
    """Intermediate membership values must match exact rational computation."""

    @pytest.mark.parametrize("A,B,left,right", CASES)
    def test_memberships_near_exact(self, A, B, left, right):
        z = Znum(A=A, B=B, left=left, right=right)
        result = z.math.get_intermediate(z.A, z.mu_A)
        _, exact_membs = _exact_intermediate(A, z.mu_A.tolist(), left, right)
        np.testing.assert_allclose(result["memb"], exact_membs, atol=_ATOL)

    @pytest.mark.parametrize("A,B,left,right", CASES)
    def test_membership_endpoints(self, A, B, left, right):
        """Membership at trapezoid corners: 0 at edges, 1 at flat top."""
        z = Znum(A=A, B=B, left=left, right=right)
        result = z.math.get_intermediate(z.A, z.mu_A)
        memb = result["memb"]
        if A[0] != A[1]:  # non-degenerate left slope
            assert memb[0] == pytest.approx(0, abs=_ATOL)
            assert memb[left] == pytest.approx(1, abs=_ATOL)
        if A[2] != A[3]:  # non-degenerate right slope
            assert memb[left + 1] == pytest.approx(1, abs=_ATOL)
            assert memb[-1] == pytest.approx(0, abs=_ATOL)

    @pytest.mark.parametrize("A,B,left,right", CASES)
    def test_membership_bounded_0_1(self, A, B, left, right):
        """All membership values must be in [0, 1]."""
        z = Znum(A=A, B=B, left=left, right=right)
        result = z.math.get_intermediate(z.A, z.mu_A)
        assert np.all(result["memb"] >= -_ATOL)
        assert np.all(result["memb"] <= 1 + _ATOL)

    @pytest.mark.parametrize("A,B,left,right", CASES)
    def test_left_slope_monotonic(self, A, B, left, right):
        """Membership on the left slope must be non-decreasing."""
        z = Znum(A=A, B=B, left=left, right=right)
        result = z.math.get_intermediate(z.A, z.mu_A)
        left_membs = result["memb"][: left + 1]
        assert all(left_membs[i] <= left_membs[i + 1] + _ATOL for i in range(len(left_membs) - 1))

    @pytest.mark.parametrize("A,B,left,right", CASES)
    def test_right_slope_monotonic(self, A, B, left, right):
        """Membership on the right slope must be non-increasing."""
        z = Znum(A=A, B=B, left=left, right=right)
        result = z.math.get_intermediate(z.A, z.mu_A)
        right_membs = result["memb"][left + 1 :]
        assert all(right_membs[i] >= right_membs[i + 1] - _ATOL for i in range(len(right_membs) - 1))


class TestGetIntermediateForB:
    """get_intermediate must also work correctly for B values."""

    @pytest.mark.parametrize("A,B,left,right", CASES)
    def test_B_intermediate_near_exact(self, A, B, left, right):
        z = Znum(A=A, B=B, left=left, right=right)
        result = z.math.get_intermediate(z.B, z.mu_B)
        exact_vals, exact_membs = _exact_intermediate(B, z.mu_B.tolist(), left, right)
        np.testing.assert_allclose(result["value"], exact_vals, atol=_ATOL)
        np.testing.assert_allclose(result["memb"], exact_membs, atol=_ATOL)
