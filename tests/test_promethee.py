from znum.Znum import Znum
from znum.Beast import Beast
from znum.Promethee import Promethee


def test_same_alternatives():
    weights = [
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
    ]
    alternative1 = [
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
    ]
    alternative2 = [
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
    ]
    alternative3 = [
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
    ]
    weight_types = [
        Beast.CriteriaType.COST,
        Beast.CriteriaType.BENEFIT,
        Beast.CriteriaType.BENEFIT,
    ]
    promethee = Promethee(
        [weights, alternative1, alternative2, alternative3, weight_types],
        shouldNormalizeWeight=True,
    )
    promethee.solve()
    indices = promethee.ordered_indices
    assert indices == [0, 1, 2]  # order doesn't change


def test_obvious_best_alternative():
    weights = [
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
    ]
    alternative1 = [
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
    ]
    alternative2 = [
        Znum([2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4]),
        Znum([2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4]),
        Znum([2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4]),
    ]
    alternative3 = [
        Znum([3, 4, 5, 6], [0.1, 0.2, 0.3, 0.4]),
        Znum([3, 4, 5, 6], [0.1, 0.2, 0.3, 0.4]),
        Znum([3, 4, 5, 6], [0.1, 0.2, 0.3, 0.4]),
    ]
    weight_types = [
        Beast.CriteriaType.COST,
        Beast.CriteriaType.BENEFIT,
        Beast.CriteriaType.BENEFIT,
    ]
    promethee = Promethee(
        [weights, alternative1, alternative2, alternative3, weight_types],
        shouldNormalizeWeight=True,
    )
    promethee.solve()
    indices = promethee.ordered_indices
    assert indices == [2, 1, 0]
    assert promethee.index_of_best_alternative == 2
    assert promethee.index_of_worst_alternative == 0


def test_3():
    weights = [
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([10, 20, 30, 40], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
    ]
    alternative1 = [
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([10, 20, 30, 40], [0.1, 0.2, 0.3, 0.4]),
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
    ]
    alternative2 = [
        Znum([2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4]),
        Znum([2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4]),
        Znum([2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4]),
    ]
    alternative3 = [
        Znum([3, 4, 5, 6], [0.1, 0.2, 0.3, 0.4]),
        Znum([3, 4, 5, 6], [0.1, 0.2, 0.3, 0.4]),
        Znum([3, 4, 5, 6], [0.1, 0.2, 0.3, 0.4]),
    ]
    weight_types = [
        Beast.CriteriaType.COST,
        Beast.CriteriaType.BENEFIT,
        Beast.CriteriaType.BENEFIT,
    ]
    promethee = Promethee(
        [weights, alternative1, alternative2, alternative3, weight_types],
        shouldNormalizeWeight=True,
    )
    promethee.solve()
    indices = promethee.ordered_indices
    assert indices == [0, 2, 1]
    assert promethee.index_of_best_alternative == 0
    assert promethee.index_of_worst_alternative == 1


def test_4():
    weights = [
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([10, 20, 30, 40], [0.1, 0.2, 0.3, 0.4]),
        Znum([100, 200, 300, 400], [0.1, 0.2, 0.3, 0.4]),
    ]
    alternative1 = [
        Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),
        Znum([10, 20, 30, 40], [0.1, 0.2, 0.3, 0.4]),
        Znum([10, 20, 30, 40], [0.1, 0.2, 0.3, 0.4]),
    ]
    alternative2 = [
        Znum([2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4]),
        Znum([2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4]),
        Znum([2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4]),
    ]
    alternative3 = [
        Znum([3, 4, 5, 6], [0.1, 0.2, 0.3, 0.4]),
        Znum([3, 4, 5, 6], [0.1, 0.2, 0.3, 0.4]),
        Znum([3, 4, 5, 6], [0.1, 0.2, 0.3, 0.4]),
    ]
    weight_types = [
        Beast.CriteriaType.BENEFIT,
        Beast.CriteriaType.BENEFIT,
        Beast.CriteriaType.COST,
    ]
    promethee = Promethee(
        [weights, alternative1, alternative2, alternative3, weight_types],
        shouldNormalizeWeight=True,
    )
    promethee.solve()
    indices = promethee.ordered_indices
    assert indices == [1, 2, 0]
    assert promethee.index_of_best_alternative == 1
    assert promethee.index_of_worst_alternative == 0
