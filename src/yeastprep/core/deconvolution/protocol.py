"""The LinearOperator protocol and its sole combinator, Compose.

Vendored (trimmed) from resolvde/protocol.py: dropped ``apply_normal`` and
the "operator may optionally expose ``normal``" fast path, since this
package's NLCG solver carries no regularizer to need it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

__all__ = ["LinearOperator", "Compose"]


@runtime_checkable
class LinearOperator(Protocol):
    """A linear operator: an immutable bag of precomputed tensors plus
    two pure functions.

    Acts on the trailing ``len(in_shape)`` axes of its input and
    broadcasts over any leading axes.
    """

    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    operator_norm_sq: float

    def forward(self, x: Tensor) -> Tensor: ...

    def adjoint(self, y: Tensor) -> Tensor: ...


class Compose:
    """Composition of one or more linear operators, in mathematical order.

    ``Compose(A, B, C).forward(x) == A.forward(B.forward(C.forward(x)))``
    -- ``C`` (innermost) applies first, forward; adjoint reverses the order.
    ``operator_norm_sq`` is the product of the component bounds, a valid
    (if not always tight) upper bound.

    Shape adjacency between every consecutive pair is checked at
    construction and raises ``ValueError`` immediately on a mismatch.
    """

    def __init__(self, *ops: LinearOperator):
        if not ops:
            raise ValueError("Compose requires at least one operator")

        for outer, inner in zip(ops[:-1], ops[1:]):
            outer_in = tuple(outer.in_shape)
            inner_out = tuple(inner.out_shape)
            if outer_in != inner_out:
                raise ValueError(
                    f"Cannot compose {type(outer).__name__} after "
                    f"{type(inner).__name__}: {type(outer).__name__} expects "
                    f"input of shape {outer_in} but {type(inner).__name__} "
                    f"produces {inner_out}."
                )

        self.ops: tuple[LinearOperator, ...] = ops
        self.in_shape = tuple(ops[-1].in_shape)
        self.out_shape = tuple(ops[0].out_shape)

        norm_sq = 1.0
        for op in ops:
            norm_sq *= op.operator_norm_sq
        self.operator_norm_sq = float(norm_sq)

    def forward(self, x: Tensor) -> Tensor:
        for op in reversed(self.ops):
            x = op.forward(x)
        return x

    def adjoint(self, y: Tensor) -> Tensor:
        for op in self.ops:
            y = op.adjoint(y)
        return y
