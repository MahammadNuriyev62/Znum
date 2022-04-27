from znum.Znum import Znum
from helper.Beast import Beast
from pprint import pprint


if __name__ == '__main__':

    table = Beast.read_znums_from_xlsx(Beast.Methods.PROMETHEE)
    p = Znum.Promethee.solver_main(table)
    #
    # print(p)

