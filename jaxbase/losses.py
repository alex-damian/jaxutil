import jax
from jax import numpy as jnp

# MSE Loss
def MSELoss(f, y):
    return jnp.sum((f - y) ** 2) / 2

# Cross Entropy Loss
def CrossEntropyLoss(f, y):
    logits = jax.nn.log_softmax(f)
    return -jnp.sum(logits * y)