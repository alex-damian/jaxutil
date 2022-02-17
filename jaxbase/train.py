import jax
from jax import numpy as jnp
from jaxbase.util import fold


def GD(loss, p0, lr, steps, return_loss=True, **kwargs):
    def step(p, _):
        if return_loss:
            step_loss, grads = jax.value_and_grad(loss)(p)
        else:
            step_loss = None
            grads = jax.grad(loss)(p)
        p = jax.tree_multimap(lambda pi, gi: pi - lr * gi, p, grads)
        return p, step_loss, None

    fold_output = fold(step, p0, jnp.arange(steps), **kwargs)
    if return_loss:
        return fold_output[:2]
    else:
        return fold_output[0]
