from znum.Znum import Znum
from pprint import pprint

if __name__ == '__main__':
    znum1 = Znum([0.235, 0.355, 0.507, 0.662], [0.0000001, 0.0000029, 0.0000662, 0.0010787])
    znum2 = Znum([-1.206, -0.96, -0.73, -0.568], [0.0005868, 0.0023007, 0.0082631, 0.0271535])
    znum3 = Znum([-0.34, -0.139, 0.054, 0.249], [0.0001737, 0.0004808, 0.0007893, 0.0027116])
    znum4 = Znum([0.152, 0.297, 0.492, 0.714], [8.04e-05, 0.0003813, 0.0017818, 0.0052415])
    znum5 = Znum([0.04, 0.173, 0.326, 0.476], [1.61e-05, 0.0001472, 0.0011994, 0.007292])
    # d2 = znum1.sort.main_solver(znum2, znum1)
    # d3 = znum1.sort.main_solver(znum2, znum2)
    # d4 = znum1.sort.main_solver(znum1, znum1)
    array = [znum1, znum2, znum3, znum4, znum5]

    # print(d1, d2, d3, d4)
    # print(znum2 > znum1)
    print(znum3 > znum2)
    pprint(tuple(reversed(sorted(array))))