"""
Exhaustive tests for fast_b=True arithmetic mode.

fast_b=True computes B as element-wise min(B1, B2) instead of LP.
A computation is identical in both modes.

Tests cover:
- B equals element-wise min for all operations
- A matches LP mode exactly
- Structural validity (monotonicity, non-negativity, length)
- Commutativity
- Chained operations (B never degrades below inputs)
- Mixed B values (asymmetric reliability)
- Edge cases (identical B, crisp, near-zero B, single-element chains)
- Determinism
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from znum import Znum
from znum.math_ops import Math


# --- Fixtures ---

ZNUMS = {
    "z1": Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4]),
    "z2": Znum(A=[2, 3, 4, 5], B=[0.2, 0.3, 0.4, 0.5]),
    "small": Znum(A=[0.5, 1, 1.5, 2], B=[0.05, 0.1, 0.15, 0.2]),
    "large": Znum(A=[100, 200, 300, 400], B=[0.1, 0.2, 0.3, 0.4]),
    "large2": Znum(A=[50, 100, 150, 200], B=[0.2, 0.3, 0.4, 0.5]),
    "frac1": Znum(A=[0.1, 0.2, 0.3, 0.4], B=[0.1, 0.2, 0.3, 0.4]),
    "frac2": Znum(A=[0.2, 0.3, 0.4, 0.5], B=[0.2, 0.3, 0.4, 0.5]),
    "high_B": Znum(A=[1, 2, 3, 4], B=[0.6, 0.7, 0.8, 0.9]),
    "low_B": Znum(A=[2, 3, 4, 5], B=[0.05, 0.1, 0.15, 0.2]),
    "same_B": Znum(A=[3, 4, 5, 6], B=[0.3, 0.4, 0.5, 0.6]),
    "same_B2": Znum(A=[1, 2, 3, 4], B=[0.3, 0.4, 0.5, 0.6]),
    "neg1": Znum(A=[-4, -3, -2, -1], B=[0.1, 0.2, 0.3, 0.4]),
    "neg2": Znum(A=[-2, -1, 0, 1], B=[0.15, 0.25, 0.35, 0.45]),
    "wide": Znum(A=[1, 5, 8, 12], B=[0.1, 0.3, 0.5, 0.7]),
    "narrow": Znum(A=[4, 5, 5, 6], B=[0.2, 0.4, 0.6, 0.8]),
}

ALL_PAIRS = [
    ("z1", "z2"),
    ("z1", "small"),
    ("z1", "large"),
    ("frac1", "frac2"),
    ("large", "large2"),
    ("high_B", "low_B"),
    ("same_B", "same_B2"),
    ("neg1", "z1"),
    ("neg1", "neg2"),
    ("wide", "narrow"),
]

# Pairs safe for multiplication and division (all A values positive)
POSITIVE_PAIRS = [
    ("z1", "z2"),
    ("z1", "small"),
    ("frac1", "frac2"),
    ("large", "large2"),
    ("high_B", "low_B"),
    ("same_B", "same_B2"),
    ("wide", "narrow"),
]

OPS = [
    ("add", Math.Operations.ADDITION),
    ("sub", Math.Operations.SUBTRACTION),
    ("mul", Math.Operations.MULTIPLICATION),
    ("div", Math.Operations.DIVISION),
]


def _fast(z1, z2, op):
    return Math.z_solver_main(z1, z2, op, fast_b=True)


def _lp(z1, z2, op):
    return Math.z_solver_main(z1, z2, op, fast_b=False)


def _is_monotonic(arr):
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


def _pairs_for_op(op):
    if op in (Math.Operations.MULTIPLICATION, Math.Operations.DIVISION):
        return POSITIVE_PAIRS
    return ALL_PAIRS


# --- Core property: B equals element-wise min ---


class TestBEqualsMin:
    """fast_b=True must produce B = min(B1, B2) element-wise."""

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS)
    def test_addition_B_is_min(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = _fast(z1, z2, Math.Operations.ADDITION)
        assert_array_equal(result.B, np.minimum(z1.B, z2.B))

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS)
    def test_subtraction_B_is_min(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = _fast(z1, z2, Math.Operations.SUBTRACTION)
        assert_array_equal(result.B, np.minimum(z1.B, z2.B))

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_multiplication_B_is_min(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = _fast(z1, z2, Math.Operations.MULTIPLICATION)
        assert_array_equal(result.B, np.minimum(z1.B, z2.B))

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_division_B_is_min(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = _fast(z1, z2, Math.Operations.DIVISION)
        assert_array_equal(result.B, np.minimum(z1.B, z2.B))


# --- A matches LP mode exactly ---


class TestAMatchesLP:
    """A values must be identical regardless of fast_b flag."""

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS)
    def test_addition_A_matches(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        assert_array_equal(
            _fast(z1, z2, Math.Operations.ADDITION).A,
            _lp(z1, z2, Math.Operations.ADDITION).A,
        )

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS)
    def test_subtraction_A_matches(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        assert_array_equal(
            _fast(z1, z2, Math.Operations.SUBTRACTION).A,
            _lp(z1, z2, Math.Operations.SUBTRACTION).A,
        )

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_multiplication_A_matches(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        assert_array_equal(
            _fast(z1, z2, Math.Operations.MULTIPLICATION).A,
            _lp(z1, z2, Math.Operations.MULTIPLICATION).A,
        )

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_division_A_matches(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        assert_array_equal(
            _fast(z1, z2, Math.Operations.DIVISION).A,
            _lp(z1, z2, Math.Operations.DIVISION).A,
        )


# --- Structural validity ---


class TestStructuralValidity:
    """fast_b results must be structurally valid Z-numbers."""

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS)
    @pytest.mark.parametrize("op_name,op", [OPS[0], OPS[1]])
    def test_result_has_4_elements(self, name1, name2, op_name, op):
        result = _fast(ZNUMS[name1], ZNUMS[name2], op)
        assert len(result.A) == 4
        assert len(result.B) == 4

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    @pytest.mark.parametrize("op_name,op", [OPS[2], OPS[3]])
    def test_result_has_4_elements_mul_div(self, name1, name2, op_name, op):
        result = _fast(ZNUMS[name1], ZNUMS[name2], op)
        assert len(result.A) == 4
        assert len(result.B) == 4

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS)
    @pytest.mark.parametrize("op_name,op", [OPS[0], OPS[1]])
    def test_A_monotonic(self, name1, name2, op_name, op):
        result = _fast(ZNUMS[name1], ZNUMS[name2], op)
        assert _is_monotonic(result.A), f"A not monotonic: {result.A}"

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    @pytest.mark.parametrize("op_name,op", [OPS[2], OPS[3]])
    def test_A_monotonic_mul_div(self, name1, name2, op_name, op):
        result = _fast(ZNUMS[name1], ZNUMS[name2], op)
        assert _is_monotonic(result.A), f"A not monotonic: {result.A}"

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS)
    @pytest.mark.parametrize("op_name,op", [OPS[0], OPS[1]])
    def test_B_monotonic(self, name1, name2, op_name, op):
        result = _fast(ZNUMS[name1], ZNUMS[name2], op)
        assert _is_monotonic(result.B), f"B not monotonic: {result.B}"

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    @pytest.mark.parametrize("op_name,op", [OPS[2], OPS[3]])
    def test_B_monotonic_mul_div(self, name1, name2, op_name, op):
        result = _fast(ZNUMS[name1], ZNUMS[name2], op)
        assert _is_monotonic(result.B), f"B not monotonic: {result.B}"

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS)
    @pytest.mark.parametrize("op_name,op", [OPS[0], OPS[1]])
    def test_B_nonnegative(self, name1, name2, op_name, op):
        result = _fast(ZNUMS[name1], ZNUMS[name2], op)
        assert all(result.B >= 0), f"B has negative values: {result.B}"

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    @pytest.mark.parametrize("op_name,op", [OPS[2], OPS[3]])
    def test_B_nonnegative_mul_div(self, name1, name2, op_name, op):
        result = _fast(ZNUMS[name1], ZNUMS[name2], op)
        assert all(result.B >= 0), f"B has negative values: {result.B}"


# --- Commutativity ---


class TestCommutativity:
    """Addition and multiplication should be commutative for both A and B."""

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS)
    def test_addition_commutative(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        r1 = _fast(z1, z2, Math.Operations.ADDITION)
        r2 = _fast(z2, z1, Math.Operations.ADDITION)
        assert_array_almost_equal(r1.A, r2.A)
        assert_array_equal(r1.B, r2.B)

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_multiplication_commutative(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        r1 = _fast(z1, z2, Math.Operations.MULTIPLICATION)
        r2 = _fast(z2, z1, Math.Operations.MULTIPLICATION)
        assert_array_almost_equal(r1.A, r2.A)
        assert_array_equal(r1.B, r2.B)


# --- Chained operations: B stability ---


class TestChainedBStability:
    """With fast_b, B should never go below the min of all inputs' B."""

    def test_chained_addition_B_stable(self):
        """B stays at element-wise min across all inputs."""
        zs = [ZNUMS["z1"], ZNUMS["z2"], ZNUMS["small"], ZNUMS["z1"], ZNUMS["z2"]]
        expected_B = zs[0].B.copy()
        acc = zs[0]
        for z in zs[1:]:
            expected_B = np.minimum(expected_B, z.B)
            acc = _fast(acc, z, Math.Operations.ADDITION)
            assert_array_equal(acc.B, expected_B)

    def test_chained_multiplication_B_stable(self):
        """B stays at element-wise min across all inputs."""
        zs = [ZNUMS["z1"], ZNUMS["z2"], ZNUMS["frac1"], ZNUMS["z1"]]
        expected_B = zs[0].B.copy()
        acc = zs[0]
        for z in zs[1:]:
            expected_B = np.minimum(expected_B, z.B)
            acc = _fast(acc, z, Math.Operations.MULTIPLICATION)
            assert_array_equal(acc.B, expected_B)

    def test_chained_same_B_never_changes(self):
        """When all inputs have same B, result B is always that B."""
        B = [0.3, 0.4, 0.5, 0.6]
        zs = [
            Znum(A=[1, 2, 3, 4], B=B),
            Znum(A=[2, 3, 4, 5], B=B),
            Znum(A=[3, 4, 5, 6], B=B),
            Znum(A=[1, 3, 4, 6], B=B),
            Znum(A=[2, 4, 5, 7], B=B),
        ]
        acc = zs[0]
        for z in zs[1:]:
            acc = _fast(acc, z, Math.Operations.ADDITION)
            assert_array_almost_equal(acc.B, B)

    def test_long_chain_B_stable(self):
        """B never degrades below the minimum input even after many operations."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.2, 0.3, 0.4, 0.5])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.3, 0.4, 0.5, 0.6])
        min_B = np.minimum(z1.B, z2.B)

        acc = z1
        for i in range(20):
            acc = _fast(acc, z2, Math.Operations.ADDITION)
            assert_array_equal(acc.B, min_B)

    def test_mixed_ops_chain(self):
        """B stays at running min through mixed operations."""
        z1 = ZNUMS["z1"]
        z2 = ZNUMS["z2"]
        z3 = ZNUMS["frac2"]

        r = _fast(z1, z2, Math.Operations.ADDITION)
        assert_array_equal(r.B, np.minimum(z1.B, z2.B))

        r = _fast(r, z3, Math.Operations.MULTIPLICATION)
        expected = np.minimum(np.minimum(z1.B, z2.B), z3.B)
        assert_array_equal(r.B, expected)


# --- Asymmetric B values ---


class TestAsymmetricB:
    """Test behavior when B values differ significantly between inputs."""

    def test_high_low_B_takes_low(self):
        """When one B is much higher, result takes the lower one."""
        z_high = ZNUMS["high_B"]  # B=[0.6, 0.7, 0.8, 0.9]
        z_low = ZNUMS["low_B"]   # B=[0.05, 0.1, 0.15, 0.2]
        result = _fast(z_high, z_low, Math.Operations.ADDITION)
        assert_array_equal(result.B, z_low.B)

    def test_low_high_B_order_irrelevant(self):
        """min is symmetric, so order doesn't matter."""
        z_high = ZNUMS["high_B"]
        z_low = ZNUMS["low_B"]
        r1 = _fast(z_high, z_low, Math.Operations.ADDITION)
        r2 = _fast(z_low, z_high, Math.Operations.ADDITION)
        assert_array_equal(r1.B, r2.B)

    def test_partial_overlap_B(self):
        """When one B is lower on some elements and higher on others, min picks per-element."""
        z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.3, 0.5, 0.7])
        z2 = Znum(A=[2, 3, 4, 5], B=[0.2, 0.2, 0.6, 0.6])
        result = _fast(z1, z2, Math.Operations.ADDITION)
        assert_array_equal(result.B, [0.1, 0.2, 0.5, 0.6])


# --- Identical B ---


class TestIdenticalB:
    """When both inputs have the same B, result B must equal that B."""

    @pytest.mark.parametrize("op_name,op", OPS[:2])
    def test_same_B_preserved(self, op_name, op):
        z1 = ZNUMS["same_B"]
        z2 = ZNUMS["same_B2"]
        result = _fast(z1, z2, op)
        assert_array_equal(result.B, z1.B)

    def test_same_B_preserved_multiplication(self):
        z1 = ZNUMS["same_B2"]  # A=[1,2,3,4]
        z2 = ZNUMS["same_B"]   # A=[3,4,5,6]
        result = _fast(z1, z2, Math.Operations.MULTIPLICATION)
        assert_array_equal(result.B, z1.B)


# --- Crisp Z-numbers ---


class TestCrisp:
    """Tests with crisp (exact-value, full-reliability) Z-numbers."""

    def test_crisp_plus_crisp(self):
        z1 = Znum.crisp(3)
        z2 = Znum.crisp(5)
        result = _fast(z1, z2, Math.Operations.ADDITION)
        assert_array_almost_equal(result.A, [8, 8, 8, 8])
        assert_array_equal(result.B, [1, 1, 1, 1])

    def test_crisp_plus_fuzzy(self):
        z_crisp = Znum.crisp(3)
        z_fuzzy = Znum(A=[1, 2, 3, 4], B=[0.2, 0.3, 0.4, 0.5])
        result = _fast(z_crisp, z_fuzzy, Math.Operations.ADDITION)
        # B should be min(crisp B=[1,1,1,1], fuzzy B) = fuzzy B
        assert_array_equal(result.B, z_fuzzy.B)

    def test_crisp_times_crisp(self):
        z1 = Znum.crisp(4)
        z2 = Znum.crisp(7)
        result = _fast(z1, z2, Math.Operations.MULTIPLICATION)
        assert_array_almost_equal(result.A, [28, 28, 28, 28])
        assert_array_equal(result.B, [1, 1, 1, 1])

    def test_crisp_div_crisp(self):
        z1 = Znum.crisp(10)
        z2 = Znum.crisp(2)
        result = _fast(z1, z2, Math.Operations.DIVISION)
        assert_array_almost_equal(result.A, [5, 5, 5, 5])
        assert_array_equal(result.B, [1, 1, 1, 1])

    def test_crisp_minus_fuzzy(self):
        z_crisp = Znum.crisp(10)
        z_fuzzy = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
        result = _fast(z_crisp, z_fuzzy, Math.Operations.SUBTRACTION)
        assert_array_equal(result.B, z_fuzzy.B)


# --- Determinism ---


class TestDeterminism:
    """Same computation must produce identical results every time."""

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS[:5])
    def test_addition_deterministic(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        r1 = _fast(z1, z2, Math.Operations.ADDITION)
        r2 = _fast(z1, z2, Math.Operations.ADDITION)
        assert_array_equal(r1.A, r2.A)
        assert_array_equal(r1.B, r2.B)

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS[:4])
    def test_multiplication_deterministic(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        r1 = _fast(z1, z2, Math.Operations.MULTIPLICATION)
        r2 = _fast(z1, z2, Math.Operations.MULTIPLICATION)
        assert_array_equal(r1.A, r2.A)
        assert_array_equal(r1.B, r2.B)


# --- B bounds ---


class TestBBounds:
    """Result B must always be bounded by input B values."""

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS)
    @pytest.mark.parametrize("op_name,op", [OPS[0], OPS[1]])
    def test_B_leq_both_inputs(self, name1, name2, op_name, op):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = _fast(z1, z2, op)
        assert all(result.B <= z1.B + 1e-10)
        assert all(result.B <= z2.B + 1e-10)

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    @pytest.mark.parametrize("op_name,op", [OPS[2], OPS[3]])
    def test_B_leq_both_inputs_mul_div(self, name1, name2, op_name, op):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = _fast(z1, z2, op)
        assert all(result.B <= z1.B + 1e-10)
        assert all(result.B <= z2.B + 1e-10)

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS)
    def test_B_between_zero_and_one(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = _fast(z1, z2, Math.Operations.ADDITION)
        assert all(result.B >= 0)
        assert all(result.B <= 1)


# --- A exact formulas (same as LP mode) ---


class TestAExactFormulas:
    """A values must follow the same formulas as LP mode."""

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS)
    def test_addition_A_elementwise_sum(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = _fast(z1, z2, Math.Operations.ADDITION)
        assert_array_almost_equal(result.A, z1.A + z2.A)

    @pytest.mark.parametrize("name1,name2", ALL_PAIRS)
    def test_subtraction_A_interval(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = _fast(z1, z2, Math.Operations.SUBTRACTION)
        expected_A = [
            z1.A[0] - z2.A[3],
            z1.A[1] - z2.A[2],
            z1.A[2] - z2.A[1],
            z1.A[3] - z2.A[0],
        ]
        assert_array_almost_equal(result.A, expected_A)

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_multiplication_A_outer_bounds(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = _fast(z1, z2, Math.Operations.MULTIPLICATION)
        corners = [
            z1.A[0] * z2.A[0], z1.A[0] * z2.A[3],
            z1.A[3] * z2.A[0], z1.A[3] * z2.A[3],
        ]
        assert result.A[0] == pytest.approx(min(corners), abs=1e-6)
        assert result.A[3] == pytest.approx(max(corners), abs=1e-6)

    @pytest.mark.parametrize("name1,name2", POSITIVE_PAIRS)
    def test_division_A_outer_bounds(self, name1, name2):
        z1, z2 = ZNUMS[name1], ZNUMS[name2]
        result = _fast(z1, z2, Math.Operations.DIVISION)
        corners = [
            z1.A[0] / z2.A[0], z1.A[0] / z2.A[3],
            z1.A[3] / z2.A[0], z1.A[3] / z2.A[3],
        ]
        assert result.A[0] == pytest.approx(min(corners), abs=1e-6)
        assert result.A[3] == pytest.approx(max(corners), abs=1e-6)


# --- Negative Z-numbers ---


class TestNegativeZnums:
    """fast_b works correctly with negative A values."""

    def test_negative_addition(self):
        z1 = ZNUMS["neg1"]  # A=[-4,-3,-2,-1]
        z2 = ZNUMS["z1"]    # A=[1,2,3,4]
        result = _fast(z1, z2, Math.Operations.ADDITION)
        assert_array_almost_equal(result.A, z1.A + z2.A)
        assert_array_equal(result.B, np.minimum(z1.B, z2.B))

    def test_negative_subtraction(self):
        z1 = ZNUMS["neg1"]
        z2 = ZNUMS["neg2"]
        result = _fast(z1, z2, Math.Operations.SUBTRACTION)
        assert_array_equal(result.B, np.minimum(z1.B, z2.B))
        assert _is_monotonic(result.A)

    def test_two_negatives_addition(self):
        z1 = ZNUMS["neg1"]  # A=[-4,-3,-2,-1]
        z2 = ZNUMS["neg2"]  # A=[-2,-1,0,1]
        result = _fast(z1, z2, Math.Operations.ADDITION)
        assert_array_almost_equal(result.A, z1.A + z2.A)  # [-6,-4,-2,0]
        assert result.A[0] < 0  # at least the lower bound is negative


# --- Wide vs narrow ranges ---


class TestRangeVariety:
    """Test with Z-numbers of varying A spread."""

    def test_wide_plus_narrow(self):
        z_wide = ZNUMS["wide"]    # A=[1,5,8,12]
        z_narrow = ZNUMS["narrow"]  # A=[4,5,5,6]
        result = _fast(z_wide, z_narrow, Math.Operations.ADDITION)
        assert_array_almost_equal(result.A, z_wide.A + z_narrow.A)
        assert_array_equal(result.B, np.minimum(z_wide.B, z_narrow.B))

    def test_wide_times_narrow(self):
        z_wide = ZNUMS["wide"]
        z_narrow = ZNUMS["narrow"]
        result = _fast(z_wide, z_narrow, Math.Operations.MULTIPLICATION)
        assert _is_monotonic(result.A)
        assert_array_equal(result.B, np.minimum(z_wide.B, z_narrow.B))


# --- Self-operations ---


class TestSelfOperations:
    """Operating a Z-number with itself."""

    def test_self_addition_B(self):
        z = ZNUMS["z1"]
        result = _fast(z, z, Math.Operations.ADDITION)
        assert_array_equal(result.B, z.B)  # min(B, B) = B
        assert_array_almost_equal(result.A, z.A * 2)

    def test_self_subtraction_B(self):
        z = ZNUMS["z1"]
        result = _fast(z, z, Math.Operations.SUBTRACTION)
        assert_array_equal(result.B, z.B)

    def test_self_multiplication_B(self):
        z = ZNUMS["z1"]
        result = _fast(z, z, Math.Operations.MULTIPLICATION)
        assert_array_equal(result.B, z.B)

    def test_self_division_B(self):
        z = ZNUMS["z1"]
        result = _fast(z, z, Math.Operations.DIVISION)
        assert_array_equal(result.B, z.B)


# --- Context manager: Znum.fast() ---


class TestFastContextManager:
    """Test that Znum.fast() context manager enables fast_b for all operators."""

    def test_addition_inside_context(self):
        z1, z2 = ZNUMS["z1"], ZNUMS["z2"]
        with Znum.fast():
            result = z1 + z2
        assert_array_equal(result.B, np.minimum(z1.B, z2.B))

    def test_subtraction_inside_context(self):
        z1, z2 = ZNUMS["z1"], ZNUMS["z2"]
        with Znum.fast():
            result = z1 - z2
        assert_array_equal(result.B, np.minimum(z1.B, z2.B))

    def test_multiplication_inside_context(self):
        z1, z2 = ZNUMS["z1"], ZNUMS["z2"]
        with Znum.fast():
            result = z1 * z2
        assert_array_equal(result.B, np.minimum(z1.B, z2.B))

    def test_division_inside_context(self):
        z1, z2 = ZNUMS["z1"], ZNUMS["z2"]
        with Znum.fast():
            result = z1 / z2
        assert_array_equal(result.B, np.minimum(z1.B, z2.B))

    def test_lp_outside_context(self):
        """After exiting context, arithmetic goes back to LP."""
        z1, z2 = ZNUMS["z1"], ZNUMS["z2"]
        with Znum.fast():
            fast_result = z1 + z2
        lp_result = z1 + z2
        # LP produces different (higher) B values
        assert not np.array_equal(fast_result.B, lp_result.B)

    def test_A_matches_lp_inside_context(self):
        z1, z2 = ZNUMS["z1"], ZNUMS["z2"]
        lp_result = z1 + z2
        with Znum.fast():
            fast_result = z1 + z2
        assert_array_equal(fast_result.A, lp_result.A)

    def test_chained_ops_inside_context(self):
        z1, z2, z3 = ZNUMS["z1"], ZNUMS["z2"], ZNUMS["frac2"]
        with Znum.fast():
            result = z1 + z2 + z3
        expected_B = np.minimum(np.minimum(z1.B, z2.B), z3.B)
        assert_array_equal(result.B, expected_B)

    def test_nested_context(self):
        z1, z2 = ZNUMS["z1"], ZNUMS["z2"]
        with Znum.fast():
            with Znum.fast():
                result = z1 + z2
            # still fast after inner exits
            result2 = z1 - z2
        assert_array_equal(result.B, np.minimum(z1.B, z2.B))
        assert_array_equal(result2.B, np.minimum(z1.B, z2.B))

    def test_context_restores_after_exception(self):
        """fast_b resets even if an exception occurs inside the block."""
        z1, z2 = ZNUMS["z1"], ZNUMS["z2"]
        try:
            with Znum.fast():
                raise ValueError("test error")
        except ValueError:
            pass
        # Should be back to LP
        result = z1 + z2
        lp_result = _lp(z1, z2, Math.Operations.ADDITION)
        assert_array_equal(result.B, lp_result.B)

    def test_scalar_mul_unaffected(self):
        """Scalar multiplication doesn't go through z_solver_main, should still work."""
        z = ZNUMS["z1"]
        with Znum.fast():
            result = z * 3
        assert_array_almost_equal(result.A, z.A * 3)
        assert_array_equal(result.B, z.B)

    def test_context_matches_direct_fast_b(self):
        """Context manager should produce same results as fast_b=True kwarg."""
        z1, z2 = ZNUMS["z1"], ZNUMS["z2"]
        direct = _fast(z1, z2, Math.Operations.ADDITION)
        with Znum.fast():
            context = z1 + z2
        assert_array_equal(direct.A, context.A)
        assert_array_equal(direct.B, context.B)
