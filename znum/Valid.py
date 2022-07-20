import znum.Znum as xusun
from znum.exception import *


class Valid:
    def __init__(self, root):
        self.root: xusun.Znum = root
        self.validate_A()
        self.validate_B()

    def validate_A(self):
        A = self.root.A
        if list(sorted(A)) != A:
            raise InvalidAPartOfZnumException()

    def validate_B(self):
        B = self.root.B
        if list(sorted(B)) != B or B[-1] > 1 or B[0] < 0:
            raise InvalidBPartOfZnumException()

    def validate(self):
        A, B, C = self.root.A, self.root.B, self.root.C

        if len(A) != len(B):
            raise InvalidZnumDimensionException()

        if len(A) == len(B) != len(C):
            raise InvalidZnumCPartDimensionException()

