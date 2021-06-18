import jax
import jax.numpy as jnp

from entmax_jax.utils import logprobs, multiply_no_nan


@jax.jit
def entmax_loss(predicted_probs, true_probs, z, alpha=1.5, reduction=jnp.mean):
    omega = (1 - (predicted_probs ** alpha).sum(axis=1)) / (alpha * (alpha - 1))
    loss = jnp.einsum("ij,ij->i", predicted_probs - true_probs, z)
    return reduction(loss + omega)


@jax.jit
def sparsemax_loss(predicted_probs, true_probs, reduction=jnp.mean):
    return reduction(jnp.sum((predicted_probs - true_probs) ** 2, axis=1))


@jax.jit
def softmax_loss(predicted_probs, true_probs, reduction=jnp.mean):
    loss = jnp.sum(
        multiply_no_nan(true_probs, logprobs(true_probs) - logprobs(predicted_probs)),
        axis=1,
    )
    return reduction(loss)
