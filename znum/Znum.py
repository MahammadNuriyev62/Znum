from .Math import Math
from .Sort import Sort
from .Topsis import Topsis
from .Promethee import Promethee
from .Beast import Beast
from .Vikor import Vikor
from .Valid import Valid
from .Type import Type
from .Dist import Dist


class Znum:
    Vikor = Vikor
    Topsis = Topsis
    Sort = Sort
    Promethee = Promethee
    Beast = Beast
    Math = Math
    Dist = Dist

    def __init__(self, A=None, B=None, left=4, right=4, C=None, A_int=None, B_int=None):
        self._A = A or Znum.get_default_A()
        self._B = B or Znum.get_default_B()
        self._C = C or Znum.get_default_C()
        self._dimension = len(self._A)
        self.left, self.right = left, right
        self.math = Math(self)
        self.valid = Valid(self)
        self.type = Type(self)
        self.A_int = A_int or self.math.get_intermediate(A)
        self.B_int = B_int or self.math.get_intermediate(B)

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, A: list):
        self._A = A
        self.A_int = self.math.get_intermediate(A)

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, B: list):
        self._B = B
        self.B_int = self.math.get_intermediate(B)

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, C: list):
        self._C = C

    @property
    def dimension(self):
        return len(self._A)

    @staticmethod
    def get_default_A():
        return [1, 2, 3, 4]

    @staticmethod
    def get_default_B():
        return [0.1, 0.2, 0.3, 0.4]

    @staticmethod
    def get_default_C():
        return [0, 1, 1, 0]

    # @staticmethod
    # def create1(A_value: list, A_memb: list, B_value: list, B_memb: list, left: int = 3, right: int = 2):
    #     znum = Znum()
    #     znum.A, znum.B, znum.A_int, znum.B_int, znum.left, znum.right = \
    #         None, None, {"value": A_value, "memb": A_memb}, {"value": B_value, "memb": B_memb}, left, right
    #     return znum
    #
    # @staticmethod
    # def create2(A: list, B: list, left: int = 3, right: int = 2):
    #     znum = Znum(A, B, left, right)
    #     return znum

    def __str__(self):
        return "Znum(A="+str(self.A)+", B="+str(self.B)+")"

    def __repr__(self) -> str:
        return "Znum(A="+str(self.A)+", B="+str(self.B)+")"

    def __add__(self, other):
        return self.math.z_solver_main(self, other, Math.Operations.ADDITION)

    def __mul__(self, other):
        """
        :type other: Union[Znum, int]
        """
        if type(other) is Znum:
            return self.math.z_solver_main(self, other, Math.Operations.MULTIPLICATION)
        if type(other) is float or type(other) is int:
            return Znum([a * other for a in self.A], self.B.copy())
        else:
            raise Exception(f'Znum cannot multiplied by a data type {type(other)}')

    def __sub__(self, other):
        return self.math.z_solver_main(self, other, Math.Operations.SUBTRACTION)

    def __truediv__(self, other):
        return self.math.z_solver_main(self, other, Math.Operations.DIVISION)

    def __pow__(self, power, modulo=None):
        return Znum(A=[a ** power for a in self.A], B=self.B.copy())

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

    def to_json(self):
        return {"A": self.A, "B": self.B}

    def to_array(self):
        return self._A + self._B
