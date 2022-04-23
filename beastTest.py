from znum.Znum import Znum
from helper.Beast import Beast
from pprint import pprint

if __name__ == '__main__':

    def z():
        return Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

    table = Beast.read_znums_from_xlsx()
    p = Znum().beast.solver_main(table)

    print(p)

