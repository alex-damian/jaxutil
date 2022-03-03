import jax
from jax import numpy as jnp
import numpy as np
from functools import partial
import GPUtil
import os


class RNG:
    def __init__(self, seed=None, rng=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
        elif rng is not None:
            self.rng = rng
        else:
            raise Exception("RNG expects either a seed or an rng key.")

    def next_rng(self):
        self.rng, rng = jax.random.split(self.rng)
        return rng

    def __getattr__(self, name):
        return partial(getattr(jax.random, name), self.next_rng())


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
