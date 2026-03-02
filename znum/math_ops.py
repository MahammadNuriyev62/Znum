"""Fuzzy arithmetic operations for Z-numbers using linear programming."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy import array
from scipy import optimize

if TYPE_CHECKING:
    from znum.core import Znum

_LP_METHOD = "highs-ds"
_PRECISION = 6
_PENALTY_COEFFICIENT = 10_000


class Math:
    """Core arithmetic engine for Z-number operations.

    Uses linear programming (scipy.optimize.linprog) to compute fuzzy
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

    def get_membership(self, Q: np.ndarray, n: float) -> float:
        """Get the membership value at point n by linear interpolation on Q."""
        return self._interpolate(n, Q, self.root.C)

    def _interpolate(self, x: float, xs: np.ndarray, ys: np.ndarray) -> float:
        """Piecewise linear interpolation: find y for a given x."""
        segments = [
            [x1, x2, y1, y2]
            for x1, x2, y1, y2 in zip(xs[1:], xs[:-1], ys[1:], ys[:-1])
        ]
        for x1, x2, y1, y2 in segments:
            if x1 <= x <= x2 or x1 >= x >= x2:
                if y1 == y2:
                    return y1
                if x1 == x2:
                    return max(y1, y2)
                k = (y2 - y1) / (x2 - x1)
                b = y1 - k * x1
                return k * x + b
        return 0

    def get_intermediate(self, Q: np.ndarray) -> dict[str, np.ndarray]:
        """Compute intermediate value/membership representation for LP solving."""
        left_part = (Q[1] - Q[0]) / self.root.left
        right_part = (Q[3] - Q[2]) / self.root.right

        Q_int_value = np.concatenate(
            (
                [round(Q[0] + i * left_part, 13) for i in range(self.root.left + 1)],
                [round(Q[2] + i * right_part, 13) for i in range(self.root.right + 1)],
            )
        )
        Q_int_memb = np.array([self.get_membership(Q, i) for i in Q_int_value])
        return {"value": Q_int_value, "memb": Q_int_memb}

    def get_matrix(self) -> np.ndarray:
        """Build optimization matrix via linear programming for each B intermediate value."""
        a_vals = np.asarray(self.root.A_int["value"])
        b_vals = np.asarray(self.root.B_int["value"])
        size = len(a_vals)
        n_cols = len(b_vals)

        d = _PENALTY_COEFFICIENT
        i37 = self._weighted_centroid(self.root.A_int)
        c = np.concatenate([np.zeros(size), (d, d)], axis=0)
        bounds = np.full((size + 2, 2), (0, 1))
        A_eq = array(
            [
                np.concatenate((self.root.A_int["memb"], (-d, d))),
                np.concatenate(([1] * size, (0, 0))),
                np.concatenate((a_vals, (0, 0))),
            ]
        )

        # When all intermediate values are constant (e.g. ideal Z-numbers),
        # all LPs are identical — solve once and replicate.
        if np.all(a_vals == a_vals[0]) and np.all(b_vals == b_vals[0]):
            col = optimize.linprog(
                c, A_eq=A_eq, b_eq=array((b_vals[0], 1, i37)),
                bounds=bounds, method=_LP_METHOD,
            ).x[:-2]
            return np.tile(col, (n_cols, 1)).T

        return np.array(
            [
                optimize.linprog(
                    c, A_eq=A_eq, b_eq=array((b20, 1, i37)),
                    bounds=bounds, method=_LP_METHOD,
                ).x[:-2]
                for b20 in b_vals
            ]
        ).T

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
