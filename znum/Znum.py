from scipy import optimize
from numpy import linalg, array
# from Replacer import replacer_reverse, round_sig
from znum.Math import Math


class Znum:
    def __init__(self, A, B, left=3, right=2, precision=3):
        self.A = A
        self.B = B
        self.precision = precision
        self.left, self.right = left, right
        self.A_int = None  # self.get_intermediate(A)
        self.B_int = None  # self.get_intermediate(B)
        self.math = Math(self)

    def __str__(self):
        return f"Znum(A={self.A}, B={self.B})"

    def __repr__(self) -> str:
        return f"Znum(A={self.A}, B={self.B})"

    def __add__(self, other):
        return self.math.z_solver_main(self, other, '+')

    def __mul__(self, other):
        return self.math.z_solver_main(self, other, '*')

    def __sub__(self, other):
        return self.math.z_solver_main(self, other, '-')

    def __truediv__(self, other):
        return self.math.z_solver_main(self, other, '/')

    def __pow__(self, power, modulo=None):
        return Znum(A=[a**power for a in self.A], B=self.B.copy())