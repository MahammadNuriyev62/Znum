from znum.Znum import Znum

z1 = Znum(A=[1, 2, 3, 4], B=[0.1, 0.2, 0.3, 0.4])
z2 = Znum(A=[2, 3, 4, 5], B=[0.1, 0.2, 0.3, 0.4])
z3 = Znum([0, 1, 2, 3], [0.0, 0.1, 0.2, 0.3])
z4 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.4, 0.6])
print(z1 * 2)
print(z1 + z1)
print(z1 + z2)
print(z3 + z4)
