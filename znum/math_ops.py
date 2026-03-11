"""Fuzzy arithmetic operations for Z-numbers."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from highspy import Highs

if TYPE_CHECKING:
    from znum.core import Znum

_PRECISION = 6
_PENALTY_COEFFICIENT = 10_000


class Math:
    """Core arithmetic engine for Z-number operations.

    Uses linear programming (HiGHS solver) to compute fuzzy
    arithmetic results while preserving membership constraints.
    """

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

    @staticmethod
    def get_default_membership(size: int) -> list[float]:
        """Generate a symmetric triangular membership function of given size."""
        half = math.ceil(size / 2)
        arr = [i * (1 / (half - 1)) for i in range(half)]
        return (arr if size % 2 == 0 else arr[:-1]) + list(reversed(arr))

    def get_intermediate(self, Q: np.ndarray, mu: np.ndarray) -> dict[str, np.ndarray]:
        """Compute intermediate value/membership representation for LP solving."""
        Q_int_value = np.concatenate([
            np.linspace(Q[0], Q[1], self.root.left + 1),
            np.linspace(Q[2], Q[3], self.root.right + 1),
        ])
        Q_int_memb = np.concatenate([
            np.linspace(mu[0], mu[1], self.root.left + 1),
            np.linspace(mu[2], mu[3], self.root.right + 1),
        ])
        return {"value": Q_int_value, "memb": Q_int_memb}

    def get_matrix(self) -> np.ndarray:
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

        # Row 0: memb · x[:size] - d*x[size] + d*x[size+1] = b20
        row0_idx = np.arange(n_vars, dtype=np.int32)
        row0_val = np.concatenate([self.root.A_int["memb"], [-d, d]])
        h.addRow(b_vals[0], b_vals[0], n_vars, row0_idx, row0_val)

        # Row 1: sum(x[:size]) = 1
        row1_idx = np.arange(size, dtype=np.int32)
        h.addRow(1.0, 1.0, size, row1_idx, np.ones(size))

        # Row 2: a_vals · x[:size] = i37
        h.addRow(i37, i37, size, row1_idx, a_vals)

        # When all intermediate values are constant (e.g. ideal Z-numbers),
        # all LPs are identical — solve once and replicate.
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

    @staticmethod
    def get_Q_from_matrix(matrix: list[list[float]]) -> list[float]:
        """Extract the 4-corner trapezoidal Q values from an optimization matrix."""
        Q = np.empty(4)

        Q[0] = min(matrix, key=lambda x: x[0])[0]
        Q[3] = max(matrix, key=lambda x: x[0])[0]

        matrix = list(filter(lambda x: round(x[1], _PRECISION) == 1, matrix))

        Q[1] = min(matrix, key=lambda x: x[0])[0]
        Q[2] = max(matrix, key=lambda x: x[0])[0]

        return [round(i, _PRECISION) for i in Q]

    @staticmethod
    def get_matrix_main(number_z1: Znum, number_z2: Znum, operation: int) -> list:
        """Build the combined operation matrix for two Z-numbers."""
        matrix: list = []
        matrix1 = number_z1.math.get_matrix()
        matrix2 = number_z2.math.get_matrix()

        for i, (a1_val, a1_memb) in enumerate(
            zip(number_z1.A_int["value"], number_z1.A_int["memb"])
        ):
            for j, (a2_val, a2_memb) in enumerate(
                zip(number_z2.A_int["value"], number_z2.A_int["memb"])
            ):
                row = [
                    Math._operation_functions[operation](a1_val, a2_val),
                    min(a1_memb, a2_memb),
                ]
                matrix.append(row + np.outer(matrix1[i], matrix2[j]).flatten().tolist())
        return matrix

    @staticmethod
    def get_minimized_matrix(matrix: list) -> list:
        """Merge rows with identical first values, keeping max membership."""
        minimized: dict = {}
        for row in matrix:
            if row[0] in minimized:
                minimized[row[0]][0] = max(minimized[row[0]][0], row[1])
                for i, n in enumerate(row[2:]):
                    minimized[row[0]][i + 1] += n
            else:
                minimized[row[0]] = row[1:]
        return [[key] + minimized[key] for key in minimized]

    @staticmethod
    def get_prob_pos(matrix: list, number_z1: Znum, number_z2: Znum) -> list:
        """Compute probability-possibility distribution for the result."""
        matrix_by_column = list(zip(*matrix))
        column1 = matrix_by_column[1]
        matrix_by_column = matrix_by_column[2:]

        final_matrix = []
        size1 = len(number_z1.B_int["memb"])
        size2 = len(number_z2.B_int["memb"])

        for i, column in enumerate(matrix_by_column):
            row = [
                sum([val * col for val, col in zip(column1, column)]),
                min(
                    number_z1.B_int["memb"][i // size1],
                    number_z2.B_int["memb"][i % size2],
                ),
            ]
            final_matrix.append(row)
        return final_matrix

    @staticmethod
    def z_solver_main(number_z1: Znum, number_z2: Znum, operation: int) -> Znum:
        """Perform a fuzzy arithmetic operation and return the resulting Z-number."""
        # Runtime import to avoid circular dependency: core.py -> math_ops.py -> core.py
        from znum.core import Znum

        matrix = Math.get_matrix_main(number_z1, number_z2, operation)
        matrix = Math.get_minimized_matrix(matrix)
        A = Math.get_Q_from_matrix(matrix)
        matrix = Math.get_prob_pos(matrix, number_z1, number_z2)
        B = Math.get_Q_from_matrix(matrix)

        return Znum(A, B)
