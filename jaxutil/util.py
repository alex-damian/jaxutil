from functools import partial

import jax
from jax.tree_util import register_pytree_node


class ddict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


register_pytree_node(
    ddict,
    lambda x: (tuple(x.values()), tuple(x.keys())),
    lambda keys, values: ddict(zip(keys, values)),
)


class RNG:
    def __init__(self, seed=None, key=None):
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        elif key is not None:
            self.key = key
        else:
            raise Exception("RNG expects either a seed or random key.")

    def next(self, n_keys=1):
        if n_keys > 1:
            return jax.random.split(self.next(), n_keys)
        else:
            self.key, key = jax.random.split(self.key)
            return key

    def __getattr__(self, name):
        return partial(getattr(jax.random, name), self.next())


register_pytree_node(
    RNG,
    lambda rng: ((rng.key,), None),
    lambda _, c: RNG(key=c[0]),
)
