import os
from collections import namedtuple
from functools import partial

import GPUtil
import jax
import numpy as np
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree


def flat_init(model, *args, **kwargs):
    params = model.init(*args, **kwargs)
    params, unravel = ravel_pytree(params)
    f = lambda p, *args, **kwargs: model.apply(unravel(p), *args, **kwargs)
    return f, params, unravel


qt = lambda **kwargs: namedtuple("tuple", kwargs.keys())(**kwargs)


class RNG:
    def __init__(self, seed=None, key=None):
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        elif key is not None:
            self.key = key
        else:
            raise Exception("RNG expects either a seed or random key.")

    def next(self, n_keys=1):
        self.key, *keys = jax.random.split(self.key, n_keys + 1)
        return keys[0] if len(keys) == 1 else keys

    def __getattr__(self, name):
        return partial(getattr(jax.random, name), self.next())


def auto_cpu(x64=True):
    jax.config.update("jax_platform_name", "cpu")
    if x64:
        jax.config.update("jax_enable_x64", True)


def auto_gpu():
    deviceID = GPUtil.getFirstAvailable(verbose=True)[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceID)


def clean_dict(d):
    return {
        key: val.item() if isinstance(val, (np.ndarray, jnp.ndarray)) else val
        for key, val in d.items()
    }
