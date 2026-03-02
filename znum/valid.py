from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from znum.core import Znum

from .exceptions import (
    InvalidAPartOfZnumException,
    InvalidBPartOfZnumException,
    ZnumMustBeEvenException,
    ZnumsMustBeInSameDimensionException,
)


class Valid:
    """Validation logic for Z-number construction and operations."""

    class Decorator:
        """Decorators for validating Z-number arguments to functions."""

        @staticmethod
        def _filter_znums(args: tuple, callback, exception: type) -> None:
            """Check all Znum arguments against a callback; raise exception on failure."""
            from znum.core import Znum

            for arg in args:
                if isinstance(arg, Znum):
                    if callback(arg):
                        raise exception

        @staticmethod
        def check_if_znums_are_even(func):
            """Ensure all Znum arguments have an even number of elements."""
            @functools.wraps(func)
            def wrapper(*args):
                Valid.Decorator._filter_znums(
                    args, lambda znum: not znum.is_even, ZnumMustBeEvenException
                )
                return func(*args)
            return wrapper

        @staticmethod
        def check_if_znums_are_in_same_dimension(func):
            """Ensure all Znum arguments have the same dimension."""
            @functools.wraps(func)
            def wrapper(*args):
                from znum.core import Znum

                dimension = None
                for arg in args:
                    if isinstance(arg, Znum):
                        if dimension is None:
                            dimension = arg.dimension
                        elif dimension != arg.dimension:
                            raise ZnumsMustBeInSameDimensionException()
                return func(*args)
            return wrapper

    def __init__(self, root: Znum) -> None:
        self.root = root
        self._validate_a()
        self._validate_b()

    def _validate_a(self) -> None:
        """Ensure A values are non-decreasing."""
        A = self.root.A
        if not np.all(A[:-1] <= A[1:]):
            raise InvalidAPartOfZnumException()

    def _validate_b(self) -> None:
        """Ensure B values are non-decreasing and within [0, 1]."""
        B = self.root.B
        if not np.all(B[:-1] <= B[1:]) or B[-1] > 1 or B[0] < 0:
            raise InvalidBPartOfZnumException()
