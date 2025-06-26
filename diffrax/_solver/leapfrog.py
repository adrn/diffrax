from collections.abc import Callable
from typing import ClassVar, TypeAlias

from equinox.internal import ω
from jaxtyping import ArrayLike, Float, PyTree

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractSolver


_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None

Ya: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]
Yb: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]


class Leapfrog(AbstractSolver):
    """Leapfrog symplectic integrator.

    2nd order symplectic method. Does not support adaptive step sizing.
    Uses 1st order local linear interpolation for dense/ts output.

    Standard leapfrog scheme for symplectic integration. Assuming that:

        v0, w0 = y0

    and:

        f, g = terms

    This method computes the next step as:

        w_half = w0 + h/2 * g(t0, v0)
        v1 = v0 + h * f(t0, w_half)
        w1 = w_half + h/2 * g(t1, v1)
    """

    term_structure: ClassVar = (AbstractTerm, AbstractTerm)
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def order(self, terms):
        return 2

    def init(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[tuple[Ya, Yb], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        f, g = terms
        v0, w0 = y0
        h = t1 - t0

        w_half = (w0**ω + 0.5 * h * g.vf(t0, v0, args) ** ω).ω
        v1 = (v0**ω + h * f.vf(t0, w_half, args) ** ω).ω
        w1 = (w_half**ω + 0.5 * h * g.vf(t1, v1, args) ** ω).ω

        y1 = (v1, w1)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> VF:
        f, g = terms
        v0, w0 = y0
        fv = f.vf(t0, w0, args)
        gw = g.vf(t0, v0, args)
        return fv, gw


Leapfrog.__init__.__doc__ = """**Arguments:** None"""
