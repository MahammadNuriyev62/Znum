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
