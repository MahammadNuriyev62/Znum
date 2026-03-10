"""Znum - Z-number arithmetic and multi-criteria decision making.

A Z-number Z = (A, B) is a fuzzy number where:
- A: The restriction on values (trapezoidal fuzzy number)
- B: The reliability/confidence of A

Example:
    >>> from znum import Znum
    >>> z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    >>> z2 = Znum([2, 4, 8, 10], [0.5, 0.6, 0.7, 0.8])
    >>> z3 = z1 + z2
    >>> print(z3)
"""

from .core import Znum
from .topsis import Topsis
from .promethee import Promethee
from .utils import MCDMUtils
from .exceptions import (
    InvalidAPartOfZnumException,
    InvalidBPartOfZnumException,
    InvalidZnumDimensionException,
    InvalidZnumCPartDimensionException,
    ZnumMustBeEvenException,
    ZnumsMustBeInSameDimensionException,
)

__version__ = "1.1.0"

__all__ = [
    "Znum",
    "Topsis",
    "Promethee",
    "MCDMUtils",
    "InvalidAPartOfZnumException",
    "InvalidBPartOfZnumException",
    "InvalidZnumDimensionException",
    "InvalidZnumCPartDimensionException",
    "ZnumMustBeEvenException",
    "ZnumsMustBeInSameDimensionException",
]
