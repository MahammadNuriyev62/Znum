from znum.Znum import Znum
from helper.Beast import Beast


if __name__ == "__main__":
    table = Beast.read_znums_from_xlsx(Beast.Methods.PROMETHEE)
    problem = Znum.Promethee(table)
    problem.solve()
    print(problem.ordered_indices)
