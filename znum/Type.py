import znum.Znum as xusun

class Type:
    TRIANGLE = 1
    TRAPEZOID = 2
    ANY = 3

    def __init__(self, root):
        self.root: xusun.Znum = root
        self.value = self.get_type()

    def get_type(self):
        A, B = self.root.A, self.root.B

        if len(A) == 4:
            if A[1] == A[2] and B[1] == B[2]:
                return Type.TRIANGLE
            else:
                return Type.TRAPEZOID
        else:
            return Type.ANY

    @property
    def isTrapezoid(self):
        return self.value == Type.TRAPEZOID

    @property
    def isTriangle(self):
        return self.value == Type.TRIANGLE