"""Custom exceptions for Z-number validation errors."""


class InvalidAPartOfZnumException(Exception):
    """Raised when the A part of a Z-number is not non-decreasing."""

    def __init__(self, message="A part of Znum is not valid"):
        super().__init__(message)


class InvalidBPartOfZnumException(Exception):
    """Raised when the B part of a Z-number is not non-decreasing or outside [0, 1]."""

    def __init__(self, message="B part of Znum is not valid"):
        super().__init__(message)


class InvalidZnumDimensionException(Exception):
    """Raised when A and B parts have different dimensions."""

    def __init__(self, message="Dimensions of A and B parts should be the same"):
        super().__init__(message)


class InvalidZnumCPartDimensionException(Exception):
    """Raised when a non-trapezoid Z-number is missing a custom C part."""

    def __init__(self, message="In case A, B are not trapezoid, C must be specified"):
        super().__init__(message)


class ZnumMustBeEvenException(Exception):
    """Raised when a Z-number has an odd number of elements in A and B."""

    def __init__(self, message="Znum() must have even number of values in A and B parts"):
        super().__init__(message)


class ZnumsMustBeInSameDimensionException(Exception):
    """Raised when Z-numbers being compared have different dimensions."""

    def __init__(self, message="Znum()s must have the same dimensions (len(A1) == len(A2) == ..."):
        super().__init__(message)
