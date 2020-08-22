import jax.numpy as jnp


def reshape_to_broadcast(array: jnp.array, shape: tuple, axis: int):
    """ reshapes the `array` to be broadcastable to `shape`"""
    new_shape = [1 for _ in shape]
    new_shape[axis] = shape[axis]
    return jnp.reshape(array, new_shape)


def tile_to_broadcast(array: jnp.array, shape: tuple, axis: int):
    tile_shape = list(shape)
    tile_shape[axis] = 1
    return jnp.tile(array, tile_shape)
