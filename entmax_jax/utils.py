import jax.numpy as jnp

LOWER_CONST = 1e-7
UPPER_CONST = 1 - LOWER_CONST


def logprobs(probs):
    probs = jnp.maximum(probs, LOWER_CONST)
    probs = jnp.minimum(probs, UPPER_CONST)
    return jnp.log(probs)


def multiply_no_nan(x, y):
    dtype = jnp.result_type(x, y)
    return jnp.where(jnp.equal(x, 0.0), jnp.zeros((), dtype=dtype), jnp.multiply(x, y))


def reshape_to_broadcast(array: jnp.array, shape: tuple, axis: int):
    """ reshapes the `array` to be broadcastable to `shape`"""
    new_shape = [1 for _ in shape]
    new_shape[axis] = shape[axis]
    return jnp.reshape(array, new_shape)


def tile_to_broadcast(array: jnp.array, shape: tuple, axis: int):
    tile_shape = list(shape)
    tile_shape[axis] = 1
    return jnp.tile(array, tile_shape)
