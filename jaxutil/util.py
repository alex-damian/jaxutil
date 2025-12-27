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
    def __init__(self, *, seed=None, key=None):
        if seed is not None:
            assert key is None
            self.key = jax.random.PRNGKey(seed)
        else:
            assert key is not None
            self.key = key

    def __call__(self, n_keys=1):
        if n_keys > 1:
            return jax.random.split(self(), n_keys)
        else:
            key, self.key = jax.random.split(self.key)
            return key

    def fork(self, n_forks=1):
        return RNG(key=self(n_forks))

    def __getattr__(self, name):
        return partial(getattr(jax.random, name), self())


register_pytree_node(
    RNG,
    lambda rng: ((rng.key,), None),
    lambda _, c: RNG(key=c[0]),
)


def tree_to_dict(pytree):
    return {
        jax.tree_util.keystr(k, simple=True, separator="."): v
        for k, v in jax.tree.leaves_with_path(pytree)
    }
