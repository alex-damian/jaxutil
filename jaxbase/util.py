import jax
from jax import numpy as jnp
import numpy as np
from jax import lax
from functools import reduce

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

def auto_batch(f,max_batch_size,strict=False,reduce_mean=False,output_size=None):
    # copied from https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
    def factors(n): return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

    def batched_f(*args,rng=None):
        batch_size = len(jax.tree_flatten(args)[0][0])
        if batch_size <= max_batch_size:
            if rng is not None: return f(*args,rng=rng)
            else: return f(*args)
        else:
            if strict:
                minibatch_size = max(factor for factor in factors(batch_size) if factor <= max_batch_size)
                n_batch = batch_size//minibatch_size
            else:
                n_batch = batch_size//max_batch_size
                minibatch_size = batch_size//n_batch
            
            mapped_args = batch_split(args,batch_size=minibatch_size,strict=strict)

            def _split(x):
                x = x[:n_batch*minibatch_size]
                x = x.reshape((n_batch,minibatch_size,*x.shape[1:]))
                return x
            
            if rng is None: _f = lambda args: f(*args)
            else:
                rngs = random.split(rng,n_batch)
                mapped_args = (mapped_args,rngs)
                def _f(carry):
                    *args,rng = carry
                    return f(*args,rng=rng)
            if reduce_mean:
                return laxmean(_f,mapped_args,jnp.zeros(output_size))
            else:
                out = lax.map(_f,mapped_args)
                out = jax.tree_map(lambda x: x.reshape((n_batch*minibatch_size,*x.shape[2:])),out)
                return out
    return batched_f
