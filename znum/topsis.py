from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .dist import Dist
from .utils import MCDMUtils

if TYPE_CHECKING:
    from znum.core import Znum


class Topsis:
    """TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

    Ranks alternatives by their closeness to an ideal solution. Higher closeness
    coefficient means a better alternative.

    Args:
        table: Decision matrix where table[0] is weights, table[1:-1] are
            alternatives, and table[-1] is criteria types.
        normalize_weights: Whether to normalize the weight vector.
        distance_type: Distance method to use (DistanceMethod.SIMPLE or
            DistanceMethod.HELLINGER). Defaults to HELLINGER.
    """

    class DataType:
        ALTERNATIVE = "A"
        CRITERIA = "C"
        TYPE = "TYPE"

    class DistanceMethod:
        SIMPLE = 1
        HELLINGER = 2

    def __init__(
        self,
        table: list[list],
        normalize_weights: bool = False,
        distance_type: int | None = None,
    ) -> None:
        self.weights: list[Znum] = table[0]
        self.table_main_part: list[list[Znum]] = table[1:-1]
        self.criteria_types: list[str] = table[-1]
        self.normalize_weights = normalize_weights
        self.distance_type = distance_type if distance_type is not None else Topsis.DistanceMethod.HELLINGER
        self._result: list[float] | None = None

    def solve(self) -> list[float]:
        """Run TOPSIS and return closeness coefficients for each alternative."""
        main_table_part_transpose = tuple(zip(*self.table_main_part))

        for column_number, column in enumerate(main_table_part_transpose):
            MCDMUtils.normalize(column, self.criteria_types[column_number])

        if self.normalize_weights:
            MCDMUtils.normalize_weight(self.weights)

        Topsis._apply_weights(self.table_main_part, self.weights)

        if self.distance_type == Topsis.DistanceMethod.SIMPLE:
            table_best = Topsis._compute_distances(self.table_main_part, lambda z: Dist.Simple.calculate(z, 1))
            table_worst = Topsis._compute_distances(self.table_main_part, lambda z: Dist.Simple.calculate(z, 0))
        else:
            table_best = Topsis._compute_distances(
                self.table_main_part,
                lambda z: Dist.Hellinger.calculate(z, Dist.Hellinger.get_ideal_from_znum(z, 1)),
            )
            table_worst = Topsis._compute_distances(
                self.table_main_part,
                lambda z: Dist.Hellinger.calculate(z, Dist.Hellinger.get_ideal_from_znum(z, 0)),
            )

        s_best = Topsis._sum_rows(table_best)
        s_worst = Topsis._sum_rows(table_worst)
        self._result = Topsis._closeness_coefficient(s_best, s_worst)

        return self._result

    @property
    def result(self) -> list[float]:
        """Closeness coefficients (must call solve() first)."""
        if self._result is None:
            raise ValueError("Must call solve() before accessing result")
        return self._result

    @property
    def ordered_indices(self) -> list[int]:
        """Alternative indices sorted by closeness coefficient (best first)."""
        if self._result is None:
            raise ValueError("Must call solve() before accessing ordered_indices")
        return sorted(range(len(self._result)), key=lambda i: self._result[i], reverse=True)

    @property
    def index_of_best_alternative(self) -> int:
        """Index of the best alternative."""
        return self.ordered_indices[0]

    @property
    def index_of_worst_alternative(self) -> int:
        """Index of the worst alternative."""
        return self.ordered_indices[-1]

    @staticmethod
    def solver_main(
        table: list[list],
        normalize_weights: bool = False,
        distance_type: int | None = None,
    ) -> list[float]:
        """Static convenience method. Prefer Topsis(table).solve() instead."""
        if distance_type is None:
            distance_type = Topsis.DistanceMethod.HELLINGER
        topsis = Topsis(table, normalize_weights, distance_type)
        return topsis.solve()

    @staticmethod
    def _apply_weights(table_main_part: list[list[Znum]], weights: list[Znum]) -> None:
        """Multiply each cell by its corresponding criterion weight."""
        for row in table_main_part:
            for i, (znum, weight) in enumerate(zip(row, weights)):
                row[i] = znum * weight

    @staticmethod
    def _compute_distances(
        table_main_part: list[list[Znum]],
        distance_fn: Callable[[Znum], float],
    ) -> list[list[float]]:
        """Compute distance from each cell to an ideal using distance_fn."""
        return [[distance_fn(znum) for znum in row] for row in table_main_part]

    @staticmethod
    def _sum_rows(table: list[list[float]]) -> list[float]:
        """Sum each row to get aggregate distances per alternative."""
        return [sum(row) for row in table]

    @staticmethod
    def _closeness_coefficient(s_best: list[float], s_worst: list[float]) -> list[float]:
        """Compute closeness coefficient: D_worst / (D_best + D_worst)."""
        return [worst / (best + worst) for best, worst in zip(s_best, s_worst)]
