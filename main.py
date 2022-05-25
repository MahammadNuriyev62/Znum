from helper.Beast import Beast
from znum.Znum import Znum
from pprint import pprint

def main():
    z1 = Znum([6, 7, 8, 9], [0.8, 0.9, 0.95, 0.97])
    z2 = Znum([1, 2, 3, 4], [0.8, 0.9, 0.95, 0.97])
    print(z1 + z2)


    # z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    # z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    # print(z1 + z2)
    # # z1 = Znum([1, 2, 3, 4], [0.95, 0.96, 0.97, 0.98])
    # # z2 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    # #
    # # z = ...
    # # for i in range(1000):
    # #     z = z1 + z2
    # # print(z)
    # # ...

if __name__ == '__main__':
    main()

