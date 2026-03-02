from __future__ import annotations

from typing import TYPE_CHECKING

from .sort import Sort
from .utils import MCDMUtils

if TYPE_CHECKING:
    from znum.core import Znum


class Promethee:
    """PROMETHEE (Preference Ranking Organization METHod for Enrichment Evaluations).

    Ranks alternatives using pairwise preference comparisons and net flow scores.
    Higher net flow means a better alternative.

    Args:
        table: Decision matrix where table[0] is weights, table[1:-1] are
            alternatives, and table[-1] is criteria types.
        normalize_weights: Whether to normalize the weight vector.
    """

    def __init__(self, table: list[list], normalize_weights: bool = False) -> None:
        self.weights: list[Znum] = table[0]
        self.table_main_part: list[list[Znum]] = table[1:-1]
        self.criteria_types: list[str] = table[-1]
        self._result: tuple | None = None

        if normalize_weights:
            MCDMUtils.normalize_weight(self.weights)

    def solve(self) -> tuple:
        """Run PROMETHEE and return a sorted table of (index, net_flow) tuples."""
        table_main_part_transpose = tuple(zip(*self.table_main_part))
        for column_number, column in enumerate(table_main_part_transpose):
            MCDMUtils.normalize(column, self.criteria_types[column_number])

        preference_table = Promethee._calculate_preference_table(self.table_main_part)

        Promethee._apply_weights(preference_table, self.weights)
        Promethee._sum_pairwise_preferences(preference_table)

        vertical_sum = Promethee._column_sums(preference_table)
        horizontal_sum = Promethee._row_sums(preference_table)

        net_flows = MCDMUtils.subtract_matrix(horizontal_sum, vertical_sum)

        numerated = list(enumerate(net_flows, 0))
        sorted_table = tuple(sorted(numerated, reverse=True, key=lambda x: x[1]))

        self._result = sorted_table
        return sorted_table

    @property
    def result(self) -> tuple:
        """The sorted (index, net_flow) tuples (must call solve() first)."""
        if self._result is None:
            raise ValueError("Must call solve() before accessing result")
        return self._result

    @property
    def ordered_indices(self) -> list[int]:
        """Alternative indices sorted by net flow (best first)."""
        if self._result is None:
            raise ValueError("Must call solve() before accessing ordered_indices")
        return [r[0] for r in self._result]

    @property
    def index_of_best_alternative(self) -> int:
        """Index of the best alternative."""
        return self.ordered_indices[0]

    @property
    def index_of_worst_alternative(self) -> int:
        """Index of the worst alternative."""
        return self.ordered_indices[-1]

    @staticmethod
    def _calculate_preference_table(table_main_part: list[list[Znum]]) -> list[list]:
        """Build the pairwise preference matrix using fuzzy dominance comparisons."""
        preference_table = []
        for i_alt, alternative in enumerate(table_main_part):
            alt_row = []
            for j_alt, other_alt in enumerate(table_main_part):
                if i_alt != j_alt:
                    pairwise_prefs = []
                    for criterion, other_criterion in zip(alternative, other_alt):
                        (_, do1) = Sort.solver_main(criterion, other_criterion)
                        (_, do2) = Sort.solver_main(other_criterion, criterion)
                        d = max(do1 - do2, 0)
                        pairwise_prefs.append(d)
                    alt_row.append(pairwise_prefs)
                else:
                    alt_row.append([])
            preference_table.append(alt_row)
        return preference_table

    @staticmethod
    def _apply_weights(preference_table: list[list], weights: list[Znum]) -> None:
        """Multiply each preference value by its criterion weight."""
        for prefs_by_alt in preference_table:
            for prefs_by_criteria in prefs_by_alt:
                for index, (pref, weight) in enumerate(
                    zip(prefs_by_criteria, weights)
                ):
                    prefs_by_criteria[index] = weight * pref  # Znum * Number

    @staticmethod
    def _sum_pairwise_preferences(preference_table: list[list]) -> None:
        """Collapse per-criterion preferences into a single value per pair."""
        for prefs_by_alt in preference_table:
            for index, prefs_by_criteria in enumerate(prefs_by_alt):
                prefs_by_alt[index] = sum(prefs_by_criteria)

    @staticmethod
    def _column_sums(preference_table: list[list]) -> list:
        """Sum columns (entering flow)."""
        return [sum(column) for column in zip(*preference_table)]

    @staticmethod
    def _row_sums(preference_table: list[list]) -> list:
        """Sum rows (leaving flow)."""
        return [sum(row) for row in preference_table]
