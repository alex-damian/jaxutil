import jax
from jax import numpy as jnp
import numpy as np
from jax import lax


def clean_dict(d):
    return {key: val.item() if isinstance(val,(np.ndarray,jnp.ndarray)) else val for key, val in d.items()}


def Identity(*args, **kwargs):
    return lambda x: x


def batch_split(x, devices):
    if isinstance(devices, list):
        n_devices = len(devices)
    elif isinstance(devices, int):
        n_devices = devices
    else:
        raise Exception("Unknown device list")
    return jax.tree_map(lambda x: x.reshape((n_devices, -1, *x.shape[1:])), x)


def laxmean(f, x, init):
    n = len(jax.tree_leaves(x)[0])

    def _step(s, xi):
        s = jax.tree_multimap(lambda si, fi: si + fi / n, s, f(xi))
        return s, None

    return lax.scan(_step, init, x)[0]
