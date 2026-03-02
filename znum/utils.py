from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from znum.core import Znum


class MCDMUtils:
    """Utility functions shared across MCDM solvers (normalization, matrix ops)."""

    class CriteriaType:
        COST = "C"
        BENEFIT = "B"

    @staticmethod
    def subtract_matrix(o1: list, o2: list) -> list:
        """Element-wise subtraction of two lists of Z-numbers."""
        return [z1 - z2 for z1, z2 in zip(o1, o2)]

    @staticmethod
    def normalize(znums_of_criteria: tuple | list, criteria_type: str) -> None:
        """Normalize a column of Z-numbers in-place based on criteria type."""
        normalizers = {
            MCDMUtils.CriteriaType.COST: MCDMUtils.normalize_cost,
            MCDMUtils.CriteriaType.BENEFIT: MCDMUtils.normalize_benefit,
        }
        normalizers.get(
            criteria_type, normalizers[MCDMUtils.CriteriaType.COST]
        )(znums_of_criteria)

    @staticmethod
    def normalize_benefit(znums_of_criteria: tuple | list) -> None:
        """Normalize for benefit criteria: divide all A values by max(A)."""
        all_a = [a for znum in znums_of_criteria for a in znum.A]
        max_a = max(all_a)
        if max_a == 0:
            raise ValueError("Cannot normalize benefit criteria: all A values are zero")
        for znum in znums_of_criteria:
            znum.A = [a / max_a for a in znum.A]

    @staticmethod
    def normalize_cost(znums_of_criteria: tuple | list) -> None:
        """Normalize for cost criteria: divide min(A) by each A value (reversed)."""
        all_a = [a for znum in znums_of_criteria for a in znum.A]
        min_a = min(all_a)
        for znum in znums_of_criteria:
            if any(a == 0 for a in znum.A):
                raise ValueError("Cannot normalize cost criteria: A contains zero values")
            znum.A = list(reversed([min_a / a for a in znum.A]))

    @staticmethod
    def normalize_weight(weights: list[Znum]) -> None:
        """Normalize weight vector so weights sum to 1."""
        znum_sum = weights[0]
        for weight in weights[1:]:
            znum_sum += weight
        for i, znum in enumerate(weights):
            weights[i] = znum / znum_sum

    @staticmethod
    def parse_table(table: list[list]) -> list:
        """Extract weights, alternatives, and criteria types from a decision table."""
        weights: list[Znum] = table[0]
        table_main_part: list[list[Znum]] = table[1:-1]
        criteria_types: list[str] = table[-1]
        return [weights, table_main_part, criteria_types]

    @staticmethod
    def numerate(single_column_table: list[Znum]) -> list[tuple[int, Znum]]:
        """Pair each element with a 1-based index."""
        return list(enumerate(single_column_table, 1))

    @staticmethod
    def sort_numerated_single_column_table(
        single_column_table: List[List[Znum]],
    ) -> tuple:
        """Sort a numerated list by value in descending order."""
        return tuple(
            sorted(single_column_table, reverse=True, key=lambda x: x[1])
        )

    @staticmethod
    def transpose_matrix(matrix: list[list] | tuple[tuple]) -> zip:
        """Transpose a 2D matrix."""
        return zip(*matrix)

    @staticmethod
    def accurate_sum(znums: list[Znum]) -> Znum:
        """Sum a list of Znum objects sequentially to maintain precision."""
        result = znums[0]
        for znum in znums[1:]:
            result = result + znum
        return result
