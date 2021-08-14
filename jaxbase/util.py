import jax
from jax import numpy as jnp
from jax import lax


def clean_dict(d):
    return {key: val.item() for key, val in d.items()}


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


def laxmean(f, x, out_shape):
    n = len(jax.tree_leaves(x)[0])
    def _step(s, xi):
        s += f(xi)/n
        return s, None

    return lax.scan(_step, jnp.zeros(out_shape), x)[0]
