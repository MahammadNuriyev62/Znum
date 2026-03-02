from __future__ import annotations

from typing import TYPE_CHECKING

from .utils import MCDMUtils
from .valid import Valid

if TYPE_CHECKING:
    from znum.core import Znum


class Dist:
    """Distance metrics for comparing Z-numbers to ideal solutions."""

    class Simple:
        """Simple (Manhattan-style) distance metric."""

        _COEF = 0.5

        @staticmethod
        def calculate(znum: Znum, n: float) -> float:
            """Compute the simple distance between a Z-number and a crisp value n."""
            return sum([abs(n - p) for p in znum.A + znum.B]) * Dist.Simple._COEF

    class Hellinger:
        """Hellinger-based distance metric with A, B, and H components.

        The total distance is a weighted combination:
            distance = A * _COEF_A + B * _COEF_B + H * _COEF_H
        """

        _COEF_A = 0.5
        _COEF_B = 0.25
        _COEF_H = 0.25

        @staticmethod
        @Valid.Decorator.check_if_znums_are_even
        @Valid.Decorator.check_if_znums_are_in_same_dimension
        def calculate(znum1: Znum, znum2: Znum) -> float:
            """Compute the Hellinger distance between two Z-numbers."""
            H = Dist.Hellinger._calculate_h(znum1, znum2)
            results = Dist.Hellinger._calculate_ab(znum1, znum2)
            A, B = results["A"], results["B"]
            return (
                A * Dist.Hellinger._COEF_A
                + B * Dist.Hellinger._COEF_B
                + H * Dist.Hellinger._COEF_H
            )

        @staticmethod
        def _calculate_h(znum1: Znum, znum2: Znum) -> float:
            """Compute the H-component via optimization matrices and Hellinger formula."""
            matrix1, matrix2 = znum1.math.get_matrix(), znum2.math.get_matrix()
            transpose1 = MCDMUtils.transpose_matrix(matrix1)
            transpose2 = MCDMUtils.transpose_matrix(matrix2)
            return min(
                Dist.Hellinger._formula_hellinger(col1, col2)
                for col1, col2 in zip(transpose1, transpose2)
            )

        @staticmethod
        def _calculate_ab(znum1: Znum, znum2: Znum) -> dict[str, float]:
            """Compute the A and B distance components."""
            dimension = znum1.dimension
            half_dimension = dimension // 2
            znums = {"A": [znum1.A, znum2.A], "B": [znum1.B, znum2.B]}
            results: dict[str, list[float]] = {"A": [], "B": []}

            for key, (q1, q2) in znums.items():
                z1_left, z1_right = q1[:half_dimension], reversed(q1[half_dimension:])
                z2_left, z2_right = q2[:half_dimension], reversed(q2[half_dimension:])
                for z1l, z1r, z2l, z2r in zip(z1_left, z1_right, z2_left, z2_right):
                    results[key].append(Dist.Hellinger._formula_q(z1l, z2l, z1r, z2r))

            return {key: max(vals) for key, vals in results.items()}

        @staticmethod
        def _formula_hellinger(P: tuple | list, Q: tuple | list) -> float:
            """Compute Hellinger distance between two probability distributions."""
            return (
                (sum(((p**0.5) - (q**0.5)) ** 2 for p, q in zip(P, Q))) ** 0.5
            ) / (2**0.5)

        @staticmethod
        def _formula_q(
            z1_left: float, z2_left: float, z1_right: float, z2_right: float,
        ) -> float:
            """Compute the Q-distance between midpoints of two Z-number halves."""
            return abs(
                (z1_left + z1_right) / 2 - (z2_left + z2_right) / 2
            )

        @staticmethod
        def get_ideal_from_znum(znum: Znum, value: int = 0) -> Znum:
            """Create an ideal Z-number with uniform values for distance comparison."""
            # Runtime import to avoid circular dependency: core.py -> dist.py -> core.py
            from .core import Znum as ZnumCls
            from .math_ops import Math

            size = len(znum.A_int[Math.QIntermediate.VALUE])

            A_int = {
                Math.QIntermediate.VALUE: [value] * size,
                Math.QIntermediate.MEMBERSHIP: Math.get_default_membership(size),
            }
            B_int = {
                Math.QIntermediate.VALUE: A_int[Math.QIntermediate.VALUE].copy(),
                Math.QIntermediate.MEMBERSHIP: A_int[Math.QIntermediate.MEMBERSHIP].copy(),
            }

            return ZnumCls(
                [value] * znum.dimension,
                [value] * znum.dimension,
                A_int=A_int,
                B_int=B_int,
            )
