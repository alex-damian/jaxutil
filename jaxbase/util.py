import jax
from jax import numpy as jnp
import numpy as np
from jax import lax


def clean_dict(d):
    return {key: val.item() if isinstance(val,(np.ndarray,jnp.ndarray)) else val for key, val in d.items()}


def Identity(*args, **kwargs):
    return lambda x: x


def batch_split(batch, n_batch=None, batch_size=None, strict=True):
    n = len(jax.tree_leaves(batch)[0])
    if isinstance(n_batch, list):
        n_batch = len(devices)
        batch_size = n//n_batch
    elif isinstance(n_batch, int):
        batch_size = n//n_batch
    elif isinstance(batch_size, int):
        n_batch = n//batch_size
    else:
        raise Exception("Need to specify either n_batch or batch_size")

    if strict:
        assert n_batch*batch_size == n
    else:
        batch = jax.tree_map(lambda x: x[:n_batch*batch_size], batch)
    batches = jax.tree_map(lambda x: x.reshape((n_batch, batch_size, *x.shape[1:])), batch)
    return batches


def laxmean(f, x, init):
    n = len(jax.tree_leaves(x)[0])

    def _step(s, xi):
        s = jax.tree_multimap(lambda si, fi: si + fi / n, s, f(xi))
        return s, None

    return lax.scan(_step, init, x)[0]
