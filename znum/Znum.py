from scipy import optimize
from numpy import linalg, array
# from Replacer import replacer_reverse, round_sig
from znum.Math import Math
from znum.Sort import Sort
from znum.Topsis import Topsis
from znum.Promethee import Promethee
from znum.Beast import Beast


class Znum:

    Topsis = Topsis
    Sort = Sort
    Promethee = Promethee
    Beast = Beast

    def __init__(self, A=None, B=None, left=3, right=2, precision=3):
        self.A = A or [1, 2, 3, 4]
        self.B = B or [0.1, 0.2, 0.3, 0.4]
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
        if type(other) is Znum:
            return self.math.z_solver_main(self, other, '*')
        if type(other) is float or type(other) is int:
            return Znum([a*other for a in self.A], self.B.copy())
        else:
            raise Exception(f'Znum cannot multiplied by a data type {type(other)}')

    def __sub__(self, other):
        return self.math.z_solver_main(self, other, '-')

    def __truediv__(self, other):
        return self.math.z_solver_main(self, other, '/')

    def __pow__(self, power, modulo=None):
        return Znum(A=[a**power for a in self.A], B=self.B.copy())

    def __gt__(self, o):
        o: Znum
        d, do = Znum.Sort.solver_main(self, o)
        _d, _do = Znum.Sort.solver_main(o, self)
        return do > _do

    def __lt__(self, o):
        o: Znum
        d, do = Znum.Sort.solver_main(self, o)
        _d, _do = Znum.Sort.solver_main(o, self)
        return do < _do

    def __eq__(self, o):
        o: Znum
        d, do = Znum.Sort.solver_main(self, o)
        _d, _do = Znum.Sort.solver_main(o, self)
        return do == 1 and _do == 1

    def __ge__(self, o):
        o: Znum
        d, do = Znum.Sort.solver_main(self, o)
        _d, _do = Znum.Sort.solver_main(o, self)
        return do >= _do

    def __le__(self, o):
        o: Znum
        d, do = Znum.Sort.solver_main(self, o)
        _d, _do = Znum.Sort.solver_main(o, self)
        return do <= _do

    def copy(self):
        return Znum(A=self.A.copy(), B=self.B.copy())

