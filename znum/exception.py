class InvalidAPartOfZnumException(Exception):
    def __init__(self, message="A part of Znum is not valid"):
        self.message = message
        super().__init__(self.message)

class InvalidBPartOfZnumException(Exception):
    def __init__(self, message="B part of Znum is not valid"):
        self.message = message
        super().__init__(self.message)

class InvalidZnumDimensionException(Exception):
    def __init__(self, message="Dimensions of A and B parts should be the same"):
        self.message = message
        super().__init__(self.message)

class InvalidZnumCPartDimensionException(Exception):
    def __init__(self, message="In case A, B are not trapezoid, C must be specified"):
        self.message = message
        super().__init__(self.message)

class IncompatibleABPartsException(Exception):
    def __init__(self, message="Specified A and B are not compatible"):
        self.message = message
        super().__init__(self.message)