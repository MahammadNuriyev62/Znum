from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .math_ops import Math, _state
from .sort import Sort
from .valid import Valid

_B_MIN_THRESHOLD = 1e-3
_B_EPSILON = 1e-6


class Znum:
    """A Z-number: a fuzzy number with restriction (A) and reliability (B) components.

    A Z-number Z = (A, B) where:
    - A is a fuzzy restriction on the values of a variable
    - B is a measure of reliability (confidence) of A

    Both A and B are represented as trapezoidal fuzzy numbers, each with
    its own membership function (mu_A and mu_B).

    Args:
        A: Fuzzy restriction values (trapezoidal), e.g. [1, 2, 3, 4].
        B: Reliability values (trapezoidal), e.g. [0.1, 0.2, 0.3, 0.4].
        left: Number of intermediate points on the left slope.
        right: Number of intermediate points on the right slope.
        A_int: Pre-computed intermediate representation of A.
        B_int: Pre-computed intermediate representation of B.
    """

    _DEFAULT_MU = np.array([0, 1, 1, 0], dtype=float)

    def __init__(
        self,
        A: ArrayLike | None = None,
        B: ArrayLike | None = None,
        left: int = 4,
        right: int = 4,
        A_int: dict | None = None,
        B_int: dict | None = None,
    ) -> None:
        self._A = np.array(A if A is not None else Znum.get_default_A(), dtype=float)
        self._B = np.array(B if B is not None else Znum.get_default_B(), dtype=float)

        # Prevent degenerate LP solutions when B values are near-zero
        if self._B[-1] < _B_MIN_THRESHOLD:
            warnings.warn(
                f"B[-1] < {_B_MIN_THRESHOLD}: small epsilon added to B values "
                "to avoid degenerate linear programming solutions.",
                stacklevel=2,
            )
            for i in range(len(self._B)):
                self._B[i] += _B_EPSILON * (i + 1)

        # Each component gets its own membership function:
        # [1,1,1,1] for crisp (all values equal), [0,1,1,0] for standard trapezoid
        self._mu_A = Znum._compute_mu(self._A)
        self._mu_B = Znum._compute_mu(self._B)

        self._dimension = len(self._A)
        self.left, self.right = left, right
        self.math = Math(self)
        self.valid = Valid(self)
        self._A_int = A_int
        self._B_int = B_int

    @property
    def A_int(self) -> dict:
        """Intermediate representation of A (computed lazily)."""
        if self._A_int is None:
            self._A_int = self.math.get_intermediate(self._A, self._mu_A)
        return self._A_int

    @A_int.setter
    def A_int(self, value: dict) -> None:
        self._A_int = value

    @property
    def B_int(self) -> dict:
        """Intermediate representation of B (computed lazily)."""
        if self._B_int is None:
            self._B_int = self.math.get_intermediate(self._B, self._mu_B)
        return self._B_int

    @B_int.setter
    def B_int(self, value: dict) -> None:
        self._B_int = value

    @property
    def A(self) -> NDArray[np.float64]:
        """The fuzzy restriction values."""
        return self._A

    @A.setter
    def A(self, A: ArrayLike) -> None:
        self._A = np.array(A, dtype=float)
        self._mu_A = Znum._compute_mu(self._A)
        self.A_int = self.math.get_intermediate(self._A, self._mu_A)

    @property
    def B(self) -> NDArray[np.float64]:
        """The reliability/confidence values."""
        return self._B

    @B.setter
    def B(self, B: ArrayLike) -> None:
        self._B = np.array(B, dtype=float)
        self._mu_B = Znum._compute_mu(self._B)
        self.B_int = self.math.get_intermediate(self._B, self._mu_B)

    @property
    def mu_A(self) -> NDArray[np.float64]:
        """Membership function of the fuzzy restriction (A)."""
        return self._mu_A

    @property
    def mu_B(self) -> NDArray[np.float64]:
        """Membership function of the reliability (B)."""
        return self._mu_B

    @property
    def dimension(self) -> int:
        """Number of corner points in the fuzzy number."""
        return len(self._A)

    @property
    def is_trapezoid(self) -> bool:
        """Whether this Z-number has a trapezoidal shape (4 corner points)."""
        return len(self._A) == 4

    @property
    def is_triangle(self) -> bool:
        """Whether this Z-number has a triangular shape (e.g. [1, 2, 2, 3])."""
        return len(self._A) >= 3 and self._A[1] == self._A[-2]

    @property
    def is_even(self) -> bool:
        """Whether this Z-number has an even number of corner points."""
        return len(self._A) % 2 == 0

    @classmethod
    def crisp(cls, value: int | float) -> Znum:
        """Create a crisp (exact) Z-number with full reliability.

        A crisp Z-number has no fuzziness and 100% reliability:
        A = (value, value, value, value), B = (1, 1, 1, 1).

        Args:
            value: The exact numeric value.

        Returns:
            A Z-number representing a crisp value with full reliability.

        Example:
            >>> z = Znum.crisp(5)
            >>> print(z)
            Znum(A=[5.0, 5.0, 5.0, 5.0], B=[1.0, 1.0, 1.0, 1.0])
        """
        return cls(A=[value] * 4, B=[1, 1, 1, 1])

    @property
    def is_triangular(self) -> bool:
        """Whether both A and B are triangular (inner points equal)."""
        return (len(self._A) == 4 and self._A[1] == self._A[2]
                and len(self._B) == 4 and self._B[1] == self._B[2])

    @property
    def A_tri(self) -> NDArray[np.float64] | None:
        """A as triangular (a, m, b), or None if not triangular."""
        if len(self._A) == 4 and self._A[1] == self._A[2]:
            return np.array([self._A[0], self._A[1], self._A[3]])
        return None

    @property
    def B_tri(self) -> NDArray[np.float64] | None:
        """B as triangular (a, m, b), or None if not triangular."""
        if len(self._B) == 4 and self._B[1] == self._B[2]:
            return np.array([self._B[0], self._B[1], self._B[3]])
        return None

    @classmethod
    @contextmanager
    def fast(cls):
        """Context manager for fast analytical computation (Li et al. 2023).

        When active, triangular Z-numbers use analytical extended triangular
        distribution instead of LP. Non-triangular Z-numbers fall back to LP.

        Example:
            >>> with Znum.fast():
            ...     result = z1 + z2  # analytical if both triangular
        """
        prev = getattr(_state, 'fast', False)
        _state.fast = True
        try:
            yield
        finally:
            _state.fast = prev

    @staticmethod
    def get_default_A() -> NDArray[np.float64]:
        return np.array([1, 2, 3, 4], dtype=float)

    @staticmethod
    def get_default_B() -> NDArray[np.float64]:
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=float)

    @staticmethod
    def _compute_mu(Q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the membership function for a trapezoidal fuzzy number.

        Degenerate slopes (where endpoints are equal) get membership 1
        instead of 0, since all points collapse onto the flat top.
        """
        mu = Znum._DEFAULT_MU.copy()
        if Q[0] == Q[1]:
            mu[0] = 1
        if Q[2] == Q[3]:
            mu[3] = 1
        return mu

    def __str__(self) -> str:
        return "Znum(A=" + str(self.A.tolist()) + ", B=" + str(self.B.tolist()) + ")"

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: Znum | int | float) -> Znum:
        if isinstance(other, (int, float)) and other == 0:
            return self
        return self.math.z_solver_main(self, other, Math.Operations.ADDITION)

    def __radd__(self, other: Znum | int | float) -> Znum:
        if isinstance(other, (int, float)) and other == 0:
            return self
        return self + other

    def __sub__(self, other: Znum) -> Znum:
        return self.math.z_solver_main(self, other, Math.Operations.SUBTRACTION)

    def __mul__(self, other: Znum | int | float) -> Znum:
        if isinstance(other, Znum):
            return self.math.z_solver_main(self, other, Math.Operations.MULTIPLICATION)
        if isinstance(other, (float, int)):
            return Znum(A=self.A * other, B=self.B.copy())
        raise TypeError(f"Znum cannot be multiplied by type {type(other).__name__}")

    def __truediv__(self, other: Znum) -> Znum:
        return self.math.z_solver_main(self, other, Math.Operations.DIVISION)

    def __pow__(self, power: int | float, modulo: int | None = None) -> Znum:
        return Znum(A=self.A**power, B=self.B.copy())

    def __gt__(self, other: Znum) -> bool:
        d, do = Sort.solver_main(self, other)
        _d, _do = Sort.solver_main(other, self)
        return do > _do

    def __lt__(self, other: Znum) -> bool:
        d, do = Sort.solver_main(self, other)
        _d, _do = Sort.solver_main(other, self)
        return do < _do

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Znum):
            return False
        d, do = Sort.solver_main(self, other)
        _d, _do = Sort.solver_main(other, self)
        return do == 1 and _do == 1

    def __ge__(self, other: Znum) -> bool:
        d, do = Sort.solver_main(self, other)
        _d, _do = Sort.solver_main(other, self)
        return do >= _do

    def __le__(self, other: Znum) -> bool:
        d, do = Sort.solver_main(self, other)
        _d, _do = Sort.solver_main(other, self)
        return do <= _do

    def copy(self) -> Znum:
        """Return a deep copy of this Z-number."""
        return Znum(A=self.A.copy(), B=self.B.copy())

    def to_json(self) -> dict[str, list[float]]:
        """Serialize to a JSON-compatible dictionary."""
        return {"A": self.A.tolist(), "B": self.B.tolist()}

    def to_array(self) -> NDArray[np.float64]:
        """Concatenate A and B parts into a single array."""
        return np.concatenate([self._A, self._B])
