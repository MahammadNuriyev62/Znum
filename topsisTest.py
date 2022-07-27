from znum.Znum import Znum
from helper.Beast import Beast

if __name__ == '__main__':

    table = Beast.read_znums_from_xlsx()

    p = Znum.Topsis.solver_main(table)
    print(p)

