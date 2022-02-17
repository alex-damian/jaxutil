from jax import numpy as jnp, lax
from jax.numpy import linalg as jla


def ridge(x, y, reg):
    U, S, V = jla.svd(x, full_matrices=False)
    UTy = U.T @ y

    ridge_problem = lambda reg: V.T @ ((S / (S * S + reg)) * UTy)
    if jnp.ndim(reg) == 0:
        return ridge_problem(reg)
    else:
        return lax.map(ridge_problem, reg)
