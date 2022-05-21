from znum.Znum import Znum
from pprint import pprint

if __name__ == '__main__':

    z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])

    print(z1 + z2)