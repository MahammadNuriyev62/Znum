from znum.Znum import Znum
from helper.Beast import Beast
from pprint import pprint


if __name__ == '__main__':

    table = Beast.read_znums_from_xlsx()
    pprint(table, width=1000)
    # p = Znum().beast.solver_main(table)
    #
    # print(p)

