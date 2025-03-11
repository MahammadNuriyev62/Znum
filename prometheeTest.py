from znum.Znum import Znum
from helper.Beast import Beast


if __name__ == "__main__":
    table = Beast.read_znums_from_xlsx(Beast.Methods.PROMETHEE)
    p = Znum.Promethee(table).solve()
    #
    print(p)
