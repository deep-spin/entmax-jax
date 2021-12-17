from typing import Union
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from entmax_jax.utils import reshape_to_broadcast, tile_to_broadcast


# FIXME: make alpha differentiable
@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
@partial(jax.jit, static_argnums=(2, 3))
def _entmax(x, alpha, axis, n_iter):
    alpha = jnp.asarray(alpha)
    # if alpha is a scalar, we tile it to N_dim directly
    # else if alpha is N_dim-1 we just need to expand
    # in the right dim
    alpha = (
        tile_to_broadcast(alpha, x.shape, axis)
        if alpha.ndim == 0
        else jnp.expand_dims(alpha, axis)
    )
    d = float(x.shape[axis])  # TODO: open issue regarding ** operator bug
    x = (alpha - 1) * x
    thres_l = jnp.max(x, axis, keepdims=True) - 1
    delta = 1 - d ** (1 - alpha)

    # define bisection loop body
    two_float = jnp.array(2, dtype=jnp.float32)
    def loop_body(i, thres_l):
        threshold = thres_l + delta / (two_float ** i)
        p = jnp.maximum(x - threshold, 0) ** (1 / (alpha - 1))
        z = jnp.sum(p, axis=axis, keepdims=True)
        thres_l = jnp.where(z >= 1, threshold, thres_l)
        return thres_l

    threshold = lax.fori_loop(1, n_iter + 1, loop_body, thres_l)
    #threshold = thres_l
    #for i in range(1, n_iter + 1):
    #    threshold = loop_body(i, threshold)
    p = jnp.maximum(x - threshold, 0) ** (1 / (alpha - 1))
    return p / jnp.sum(p, axis=axis, keepdims=True)


@_entmax.defjvp
@partial(jax.jit, static_argnums=(1, 2))
def _entmax_jvp(alpha, axis, n_iter, primals, tangents):
    # unpack arguments
    x = primals[0]
    dx = tangents[0]

    # calculate entmax p and auxiliary s
    p = _entmax(x, alpha, axis, n_iter)
    s = jnp.where(p > 0, p ** (2 - alpha), 0)

    # jvp as simplified product with jacobian
    dy = dx * s
    g = jnp.sum(dy, axis=axis) / jnp.sum(s, axis=axis)
    dy = dy - jnp.expand_dims(g, axis) * s
    return p, dy


def entmax(
    x: jnp.array, alpha: Union[float, jnp.array], axis: int = -1, n_iter: int = 50
):
    """
    Implements entmax, a generalization of sparsemax and softmax

    Solves the optimization problem:

        max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.

    where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,
    using a bisection (root finding, binary search) algorithm.

    For further details, see
    https://arxiv.org/abs/1905.05702 &
    https://arxiv.org/abs/1909.00015

    Args:
        x: input array
        alpha: entropic index. If scalar or python float, the same value is used for all
            rows, otherwise use alpha is expected to be a (N-1) tensor, where N is
            number of dimensions of x
        axis: axis to compute entmax along
        n_iter: number of iterations used in the bisection algorithm
    """
    return _entmax(x, alpha, axis, n_iter)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
@partial(jax.jit, static_argnums=(1,))
def _sparsemax(x, axis):
    # get indices of elements in the right axis
    # and reshape to allow broadcasting to other dimensions
    idxs = jnp.arange(x.shape[axis]) + 1
    idxs = reshape_to_broadcast(idxs, x.shape, axis)

    # calculate number of elements that belong to the support
    sorted_x = jnp.flip(lax.sort(x, dimension=axis), axis=axis)
    cum = jnp.cumsum(sorted_x, axis=axis)
    k = jnp.sum(jnp.where(1 + sorted_x * idxs > cum, 1, 0), axis=axis, keepdims=True)

    # calculate threshold and project to simplex
    threshold = (jnp.take_along_axis(cum, k - 1, axis=axis) - 1) / k
    return jnp.maximum(x - threshold, 0)


@_sparsemax.defjvp
@partial(jax.jit, static_argnums=(0,))
def _sparsemax_jvp(axis, primals, tangents):
    # unpack arguments
    x = primals[0]
    dx = tangents[0]

    # calculate entmax p and auxiliary s
    p = _sparsemax(x, axis)
    s = jnp.where(p > 0, 1, 0)

    # jvp as simplified product with jacobian
    dy = dx * s
    g = jnp.sum(dy, axis=axis) / jnp.sum(s, axis=axis)
    dy = dy - jnp.expand_dims(g, axis) * s
    return p, dy


def sparsemax(x: jnp.array, axis: int = -1):
    """
    Implements sparsemax, a function that maps arbitrary vectors onto the
    probability simplex, with possibly non-dense support

    Solves the projection:

        min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.

    This is a special case of the more general entmax with alpha=2

    For further details, see
    https://arxiv.org/abs/1602.02068

    Args:
        x: the input array
        axis: axis to compute sparsemax along
    """
    return _sparsemax(x, axis)


# TODO: check if automatic jvp is efficient
# if not might be useful to manually define the JVP
@partial(jax.custom_jvp, nondiff_argnums=(1,))
@partial(jax.jit, static_argnums=(1,))
def _entmax15(x, axis):
    x = x / 2

    # get indices of elements in the right axis
    # and reshape to allow broadcasting to other dimensions
    idxs = jnp.arange(x.shape[axis]) + 1
    idxs = reshape_to_broadcast(idxs, x.shape, axis)

    # calculate number of elements that belong to the support
    sorted_x = jnp.flip(lax.sort(x, dimension=axis), axis=axis)
    cum_x = jnp.cumsum(sorted_x, axis=axis)
    cum_x_sq = jnp.cumsum(sorted_x ** 2, axis=axis)
    mean = cum_x / idxs
    var = cum_x_sq - (mean ** 2) * idxs
    clamped_delta = jnp.maximum((1 - var) / idxs, 0)
    thresholds = mean - jnp.sqrt(clamped_delta)
    k = jnp.sum(jnp.where(thresholds <= sorted_x, 1, 0), axis=axis, keepdims=True)

    # calculate threshold and project to simplex
    threshold = jnp.take_along_axis(thresholds, k - 1, axis=axis)
    return jnp.maximum(x - threshold, 0) ** 2


@_entmax15.defjvp
@partial(jax.jit, static_argnums=(0,))
def _entmax15_vjp(axis, primals, tangents):
    # unpack arguments
    x = primals[0]
    dx = tangents[0]

    # calculate entmax p and auxiliary s
    p = _entmax15(x, axis)
    s = jnp.sqrt(p)

    # jvp as simplified product with jacobian
    dy = dx * s
    g = jnp.sum(dy, axis=axis) / jnp.sum(s, axis=axis)
    dy = dy - jnp.expand_dims(g, axis) * s
    return p, dy


def entmax15(x: jnp.array, axis: int = -1):
    """
    Implements entmax15, a function that maps arbitrary vectors onto the
    probability simplex, with possibly non-dense support

    Solves the optimization problem:

        max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.

    where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.

    Args:
      x: the input array
      axis: axis to compute entmax15 along
    """
    return _entmax15(x, axis)
