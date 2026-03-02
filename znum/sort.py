"""Fuzzy number comparison using the NxF (Necessity for Fuzzy) framework.

Compares two Z-numbers by computing dominance degrees through normalized
intermediate representations and possibility measures over three fuzzy
linguistic categories: better (nbF), equal (neF), and worse (nwF).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from znum.core import Znum

# NxF linguistic categories and their trapezoidal membership boundaries.
# Each tuple defines (left, inner-left, inner-right, right) of the trapezoid.
_NXF_OPTIONS = dict(nbF="nbF", neF="neF", nwF="nwF")

_NXF = {
    _NXF_OPTIONS["nbF"]: (-1, -1, -0.3, -0.1),   # "better" region
    _NXF_OPTIONS["neF"]: (-0.3, -0.1, 0.1, 0.3),  # "equal" region
    _NXF_OPTIONS["nwF"]: (0.1, 0.3, 1, 1),         # "worse" region
}


class Sort:
    """Fuzzy dominance-based comparison between two Z-numbers."""

    @staticmethod
    def solver_main(znum1: Znum, znum2: Znum) -> tuple[float, float]:
        """Compare znum1 against znum2 and return (d, do) dominance scores.

        Returns:
            A tuple (d, do) where do = 1 - d. Higher do means znum1 dominates znum2 more.
        """
        (norm_a1, norm_a2) = Sort._normalize(znum1.A, znum2.A)

        intermediate_a = Sort._get_intermediate(norm_a1, norm_a2)
        intermediate_b = Sort._get_intermediate(znum1.B, znum2.B)

        intermediates = {"A": intermediate_a, "B": intermediate_b}
        nxf_possibilities = {
            q: {
                option: Sort._nxf_possibility(intermediates[q], option)
                for option in _NXF_OPTIONS
            }
            for q in intermediates
        }
        nxf_values = {
            q: {
                option: Sort._nxf_value(nxf_possibilities[q], option)
                for option in _NXF_OPTIONS
            }
            for q in intermediates
        }

        d = Sort._final_score(nxf_values)
        do = 1 - d

        return d, do

    @staticmethod
    def _normalize(
        q1: list[float] | object,
        q2: list[float] | object,
    ) -> tuple[list[float], list[float]]:
        """Min-max normalize both sequences together into [0, 1]."""
        qs = [*q1, *q2]
        min_q, max_q = min(qs), max(qs)

        if min_q == max_q:
            return [0] * len(q1), [0] * len(q2)

        normalized = [(q - min_q) / (max_q - min_q) for q in qs]
        return normalized[: len(q1)], normalized[len(q1):]

    @staticmethod
    def _get_intermediate(norm_q1: list[float], norm_q2: list[float]) -> list[float]:
        """Compute element-wise difference between q1 and reversed q2."""
        return [
            q1 - norm_q2[len(norm_q2) - index - 1]
            for (index, q1) in enumerate(norm_q1)
        ]

    @staticmethod
    def _nxf_possibility(
        intermediate: list[float],
        option: str,
    ) -> float:
        """Compute the possibility measure of intermediate falling in the given NxF category."""
        a1, a2, a3, a4 = intermediate
        alpha_l, a1, a2, alpha_r = [a2 - a1, a2, a3, a4 - a3]

        b1, b2, b3, b4 = _NXF[option]
        beta_l, b1, b2, beta_r = [b2 - b1, b2, b3, b4 - b3]

        return Sort._formula_nxf_possibility(
            alpha_l, a1, a2, alpha_r, beta_l, b1, b2, beta_r
        )

    @staticmethod
    def _formula_nxf_possibility(
        alpha_l: float, a1: float, a2: float, alpha_r: float,
        beta_l: float, b1: float, b2: float, beta_r: float,
    ) -> float:
        """Compute possibility of overlap between two trapezoidal shapes."""
        if 0 < a1 - b2 < alpha_l + beta_r:
            denominator = alpha_l + beta_r
            if denominator == 0:
                return 0.0
            return 1 - (a1 - b2) / denominator
        elif max(a1, b1) <= min(a2, b2):
            return 1
        elif 0 < b1 - a2 < alpha_r + beta_l:
            denominator = alpha_r + beta_l
            if denominator == 0:
                return 0.0
            return 1 - (b1 - a2) / denominator
        else:
            return 0

    @staticmethod
    def _nxf_value(nxf_possibilities: dict[str, float], option: str) -> float:
        """Normalize a possibility value against the sum of all possibilities."""
        others_sum = sum(
            nxf_possibilities[opt]
            for opt in nxf_possibilities
            if opt != option
        )
        possibility = nxf_possibilities[option]
        denominator = possibility + others_sum
        if denominator == 0:
            return 0.0
        return possibility / denominator

    @staticmethod
    def _final_score(nxf_values: dict[str, dict[str, float]]) -> float:
        """Combine A and B NxF values into a single dominance score."""
        nxf_sum = tuple(
            (a + b) for a, b in zip(*(q.values() for q in nxf_values.values()))
        )
        nb, ne = nxf_sum[:2]
        if nb == 0 or (2 - ne) / 2 >= nb:
            return 0.0
        return (2 * nb + ne - 2) / nb
