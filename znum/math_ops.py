"""Fuzzy arithmetic operations for Z-numbers.

The arithmetic pipeline:

1. **A computation** (LP-free): Cross-product of A intermediates, apply the
   arithmetic operation, merge duplicates, extract trapezoidal corners.

2. **B computation**: Two modes:
   - **LP mode** (default): Linear programming via HiGHS to build probability
     distributions, then derive result B via probability-possibility transform.
   - **Analytical mode** (Znum.fast(), triangular only): Li et al. 2023 extended
     triangular distribution — no LP, no B inflation.
"""
from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING

import numpy as np
from highspy import Highs

_state = threading.local()

if TYPE_CHECKING:
    from znum.core import Znum

_PRECISION = 6
_PENALTY_COEFFICIENT = 10_000


class Math:
    """Core arithmetic engine for Z-number operations."""

    class Operations:
        ADDITION = 1
        SUBTRACTION = 2
        DIVISION = 3
        MULTIPLICATION = 4

    class QIntermediate:
        VALUE = "value"
        MEMBERSHIP = "memb"

    _operation_functions = {
        Operations.ADDITION: lambda x, y: x + y,
        Operations.SUBTRACTION: lambda x, y: x - y,
        Operations.MULTIPLICATION: lambda x, y: x * y,
        Operations.DIVISION: lambda x, y: x / y,
    }

    def __init__(self, root: Znum) -> None:
        self.root = root

    # ---- Intermediate representations (used by core.py and dist.py) ----

    @staticmethod
    def get_default_membership(size: int) -> list[float]:
        """Generate a symmetric triangular membership function of given size."""
        half = math.ceil(size / 2)
        arr = [i * (1 / (half - 1)) for i in range(half)]
        return (arr if size % 2 == 0 else arr[:-1]) + list(reversed(arr))

    def get_intermediate(self, Q: np.ndarray, mu: np.ndarray) -> dict[str, np.ndarray]:
        """Compute intermediate value/membership representation."""
        Q_int_value = np.concatenate([
            np.linspace(Q[0], Q[1], self.root.left + 1),
            np.linspace(Q[2], Q[3], self.root.right + 1),
        ])
        Q_int_memb = np.concatenate([
            np.linspace(mu[0], mu[1], self.root.left + 1),
            np.linspace(mu[2], mu[3], self.root.right + 1),
        ])
        return {"value": Q_int_value, "memb": Q_int_memb}

    # ---- Probability distribution matrix ----

    def get_matrix(self) -> np.ndarray:
        """Build probability distribution matrix.

        With Znum.fast() active and triangular input: analytical (Li et al. 2023).
        Otherwise: LP via HiGHS.
        """
        if getattr(_state, 'fast', False) and self.root.is_triangular:
            from znum.tri_math import analytical_matrix
            return analytical_matrix(self.root)
        return self._get_matrix_lp()

    def _get_matrix_lp(self) -> np.ndarray:
        """Build optimization matrix via linear programming for each B intermediate value.

        Builds the HiGHS model once, then re-solves for each B intermediate
        value by changing only the first constraint's RHS (b_eq[0]).
        """
        a_vals = np.asarray(self.root.A_int["value"])
        b_vals = np.asarray(self.root.B_int["value"])
        size = len(a_vals)
        n_cols = len(b_vals)

        d = _PENALTY_COEFFICIENT
        i37 = self._weighted_centroid(self.root.A_int)
        n_vars = size + 2

        # Build model once
        h = Highs()
        h.silent()
        h.addVars(n_vars, np.zeros(n_vars), np.ones(n_vars))
        h.changeColsCost(
            2,
            np.array([size, size + 1], dtype=np.int32),
            np.array([d, d], dtype=np.float64),
        )

        # Row 0: memb . x[:size] - d*x[size] + d*x[size+1] = b20
        row0_idx = np.arange(n_vars, dtype=np.int32)
        row0_val = np.concatenate([self.root.A_int["memb"], [-d, d]])
        h.addRow(b_vals[0], b_vals[0], n_vars, row0_idx, row0_val)

        # Row 1: sum(x[:size]) = 1
        row1_idx = np.arange(size, dtype=np.int32)
        h.addRow(1.0, 1.0, size, row1_idx, np.ones(size))

        # Row 2: a_vals . x[:size] = i37
        h.addRow(i37, i37, size, row1_idx, a_vals)

        # When all intermediate values are constant (e.g. ideal Z-numbers),
        # all LPs are identical -- solve once and replicate.
        if np.all(a_vals == a_vals[0]) and np.all(b_vals == b_vals[0]):
            h.run()
            col = np.array(h.allVariableValues())[:size]
            return np.tile(col, (n_cols, 1)).T

        # Solve each column, reusing the model structure
        results = []
        for b20 in b_vals:
            h.changeRowBounds(0, b20, b20)
            h.clearSolver()
            h.run()
            results.append(np.array(h.allVariableValues())[:size])

        return np.array(results).T

    @staticmethod
    def _weighted_centroid(Q_int: dict[str, np.ndarray]) -> float:
        """Compute the membership-weighted centroid of intermediate values."""
        return np.dot(Q_int["value"], Q_int["memb"]) / np.sum(Q_int["memb"])

    # ---- A computation (no LP) ----

    @staticmethod
    def _compute_a_pairs(z1: Znum, z2: Znum, operation: int) -> list[list[float]]:
        """Cross-product of A intermediates: apply operation + fuzzy-min membership.

        Returns list of [value, membership] rows. No LP involved.
        """
        op_fn = Math._operation_functions[operation]
        pairs = []
        for a1_val, a1_memb in zip(z1.A_int["value"], z1.A_int["memb"]):
            for a2_val, a2_memb in zip(z2.A_int["value"], z2.A_int["memb"]):
                pairs.append([op_fn(a1_val, a2_val), min(a1_memb, a2_memb)])
        return pairs

    @staticmethod
    def _merge_rows(rows: list[list[float]]) -> list[list[float]]:
        """Merge rows sharing the same value: max membership, sum extra columns."""
        merged: dict = {}
        for row in rows:
            key = row[0]
            if key in merged:
                merged[key][0] = max(merged[key][0], row[1])
                for i, n in enumerate(row[2:]):
                    merged[key][i + 1] += n
            else:
                merged[key] = row[1:]
        return [[key] + merged[key] for key in merged]

    @staticmethod
    def _extract_trapezoid(rows: list[list[float]]) -> list[float]:
        """Extract [a, b, c, d] trapezoidal corners from (value, membership) rows."""
        Q = np.empty(4)

        Q[0] = min(rows, key=lambda x: x[0])[0]
        Q[3] = max(rows, key=lambda x: x[0])[0]

        core = list(filter(lambda x: round(x[1], _PRECISION) == 1, rows))

        Q[1] = min(core, key=lambda x: x[0])[0]
        Q[2] = max(core, key=lambda x: x[0])[0]

        return [round(i, _PRECISION) for i in Q]

    # ---- B computation: LP path ----

    @staticmethod
    def _compute_b_columns(z1: Znum, z2: Znum) -> list[list[float]]:
        """Build LP outer-product columns for all (i,j) A-intermediate pairs.

        Calls get_matrix() (LP) on each Znum, then computes
        outer(matrix1[i], matrix2[j]) for each pair.
        """
        matrix1 = z1.math.get_matrix()
        matrix2 = z2.math.get_matrix()
        columns = []
        for i in range(len(z1.A_int["value"])):
            for j in range(len(z2.A_int["value"])):
                columns.append(np.outer(matrix1[i], matrix2[j]).flatten().tolist())
        return columns

    @staticmethod
    def _compute_prob_pos(merged_rows: list[list[float]], z1: Znum, z2: Znum) -> list[list[float]]:
        """Compute probability-possibility distribution for result B.

        Uses col 1 (A memberships) as weights, cols 2+ as LP data.
        """
        cols_by_column = list(zip(*merged_rows))
        memberships = cols_by_column[1]
        b_columns = cols_by_column[2:]

        size1 = len(z1.B_int["memb"])
        size2 = len(z2.B_int["memb"])

        result = []
        for i, column in enumerate(b_columns):
            row = [
                sum(val * col for val, col in zip(memberships, column)),
                min(z1.B_int["memb"][i // size1], z2.B_int["memb"][i % size2]),
            ]
            result.append(row)
        return result

    @staticmethod
    def _compute_result_B_lp(a_pairs: list[list[float]], z1: Znum, z2: Znum) -> list[float]:
        """Full LP-based B pipeline: LP matrices -> outer products -> prob_pos -> trapezoid."""
        b_columns = Math._compute_b_columns(z1, z2)
        combined = [pair + cols for pair, cols in zip(a_pairs, b_columns)]
        merged = Math._merge_rows(combined)
        prob_pos = Math._compute_prob_pos(merged, z1, z2)
        return Math._extract_trapezoid(prob_pos)

    # ---- Public entry point ----

    @staticmethod
    def z_solver_main(z1: Znum, z2: Znum, operation: int) -> Znum:
        """Perform fuzzy arithmetic on two Z-numbers.

        Args:
            z1: First Z-number operand.
            z2: Second Z-number operand.
            operation: The arithmetic operation (from Math.Operations).
        """
        # Runtime import to avoid circular dependency: core.py -> math_ops.py -> core.py
        from znum.core import Znum

        fast = getattr(_state, 'fast', False)

        # Fast analytical path for triangular Z-numbers (Li et al. 2023)
        if fast and z1.is_triangular and z2.is_triangular:
            from znum.tri_math import tri_solve
            return tri_solve(z1, z2, operation)

        # LP path (default, or non-triangular fallback)
        a_pairs = Math._compute_a_pairs(z1, z2, operation)
        merged_a = Math._merge_rows(a_pairs)
        A = Math._extract_trapezoid(merged_a)
        B = Math._compute_result_B_lp(a_pairs, z1, z2)

        return Znum(A, B)
