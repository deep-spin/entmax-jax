import pytest

import jax
import jax.numpy as jnp

from entmax_jax import entmax, entmax15, sparsemax


@pytest.mark.parametrize(
    "x,y,alpha",
    [
        ([0, 1, 1.5], [0, 0.25, 0.75], 2),
        ([[0.5, 1], [1.5, 2.0]], [[0.25, 0.75], [0.32600737, 0.67399263]], [2, 1.5]),
    ],
)
@pytest.mark.parametrize("n_iter", [20, 50])
def test_entmax(x, y, alpha, n_iter):
    """ tests if entmax gives out the right output predefined input/output pair """
    x, y = jnp.asarray(x), jnp.asarray(y)
    y_pred = entmax(x, alpha=alpha, n_iter=n_iter, axis=-1)
    assert jnp.all(jnp.isclose(y, y_pred))


@pytest.mark.parametrize("x,axis", [([[0.5, 1], [1.5, 2.0]], 0)])
def test_entmax_axis(x, axis, alpha=1.5, n_iter=50):
    """ tests of entmax gives out the right when computed on axis other than default """
    x = jnp.asarray(x)
    transp = list(range(x.ndim))
    transp[-1] = axis
    transp[axis] = x.ndim - 1
    x_t = jnp.transpose(x, transp)
    y = entmax(x, alpha=alpha, n_iter=n_iter)
    y_t = entmax(x_t, alpha=alpha, axis=axis, n_iter=n_iter)
    assert jnp.all(jnp.isclose(y, jnp.transpose(y_t, transp)))


@pytest.mark.parametrize(
    "x,jac,alpha", [([0, 1, 1.5], [[0, 0, 0], [0, 0.5, -0.5], [0, -0.5, 0.5]], 2)],
)
@pytest.mark.parametrize("n_iter", [20, 50])
def test_entmax_jacobian(x, jac, alpha, n_iter):
    """ tests if entmax jacobian matches the expected value """
    x, jac = jnp.asarray(x), jnp.asarray(jac)
    jac_pred = jax.jacobian(entmax)(x, alpha, n_iter=n_iter)
    assert jnp.all(jnp.isclose(jac, jac_pred))


@pytest.mark.parametrize("x", [[0, 1, 1.5], [0, 0.5, 1]])
def test_sparsemax(x):
    """ test if sparsemax output matches entmax with alpha=2 """
    x = jnp.asarray(x)
    y_sp = sparsemax(x)
    y_en = entmax(x, alpha=2)
    assert jnp.all(jnp.isclose(y_sp, y_en))


@pytest.mark.parametrize("x", [[0, 1, 1.5], [0, 1.5, 2]])
def test_entmax15(x):
    """ test if entmax15 output matches entmax with alpha=1.5 """
    x = jnp.asarray(x)
    y_en15 = entmax15(x)
    y_en = entmax(x, alpha=1.5)
    assert jnp.all(jnp.isclose(y_en15, y_en))
