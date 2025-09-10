from typing import Callable, Optional, TypeVar

import jax
from jax import eval_shape, lax, vmap
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_unflatten
from jax_tqdm.scan_pbar import scan_tqdm
from jaxopt.tree_util import tree_add, tree_scalar_mul
from tqdm.auto import trange

Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")

tree_len = lambda tree: len(tree_leaves(tree)[0])


def tree_stack(trees):
    _, treedef = tree_flatten(trees[0])
    leaf_list = [tree_flatten(tree)[0] for tree in trees]
    leaf_stacked = [jnp.stack(leaves) for leaves in zip(*leaf_list)]
    return tree_unflatten(treedef, leaf_stacked)


def batch_split(
    batch,
    n_batch: Optional[int] = None,
    batch_size: Optional[int] = None,
    devices: Optional[tuple] = None,
    strict=True,
):
    if devices is not None:
        batches = batch_split(batch, n_batch=len(devices))
        batches = tree_map(lambda x: jax.device_put_sharded(list(x), devices), batches)
        return batches

    n = tree_len(batch)

    if n_batch is not None:
        batch_size = n // n_batch
    elif batch_size is not None:
        n_batch = n // batch_size

    assert isinstance(n_batch, int) and isinstance(batch_size, int)

    if strict:
        assert n_batch * batch_size == n
    else:
        batch = tree_map(lambda x: x[: n_batch * batch_size], batch)

    batches = tree_map(lambda x: x.reshape((n_batch, batch_size, *x.shape[1:])), batch)
    return batches


def fold(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    state,
    xs=None,
    length=None,
    backend="python",
    jit=False,
    show_progress=False,
    save_every=1,
) -> tuple[Carry, Y]:
    fast_step = lambda state, x: (f(state, x)[0], None)

    assert xs is not None or length is not None
    if xs is None:
        xs = jnp.arange(length)

    if jit:
        f = jax.jit(f)
        fast_step = jax.jit(fast_step)

    n = tree_len(xs)

    if backend == "python":
        idx = trange(n) if show_progress else range(n)
        save_list = []
        for i in idx:
            batch = tree_map(lambda x: x[i], xs)
            if i % save_every == 0:
                state, save = f(state, batch)
                save_list.append(save)
            else:
                state, _ = fast_step(state, batch)
        save_list = tree_stack(save_list)
        return state, save_list

    elif backend == "lax":
        if save_every > 1:
            xs = batch_split(xs, batch_size=save_every, strict=False)

            def epoch_fn(state: Carry, batch: X) -> tuple[Carry, Y]:
                x0 = tree_map(lambda x: x[0], batch)
                state, save = f(state, x0)
                sub_batch = tree_map(lambda x: x[1:], batch)
                state = lax.scan(fast_step, state, sub_batch)[0]
                return state, save

            if show_progress:
                epoch_fn = scan_tqdm(tree_len(xs))(epoch_fn)

            return lax.scan(epoch_fn, state, xs)
        else:
            step_fn = f
            if show_progress:
                step_fn = scan_tqdm(n)(step_fn)
            return lax.scan(step_fn, state, xs)  # type: ignore

    raise ValueError(f"Unknown backend: {backend}")


def laxmap(f, xs, batch_size=None, **kwargs):
    if batch_size is None:
        return fold(lambda _, x: (None, f(x)), None, xs, **kwargs)[1]
    else:
        batches = batch_split(xs, batch_size=batch_size)
        batched_out = fold(
            lambda _, batch: (None, vmap(f)(batch)), None, batches, **kwargs
        )[1]
        flat_out = tree_map(lambda x: x.reshape(-1, *x.shape[2:]), batched_out)
        return flat_out


def laxsum(f, data, batch_size=None, **kwargs):
    x0 = tree_map(lambda x: x[0], data)
    sum_init = tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), eval_shape(f, x0))
    if batch_size is None:
        return fold(
            lambda avg, x: (tree_add(avg, f(x)), None), sum_init, data, **kwargs
        )[0]
    else:

        def batched_f(batch):
            out_tree = vmap(f)(batch)
            return tree_map(lambda x: x.sum(0), out_tree)

        batches = batch_split(data, batch_size=batch_size)
        return fold(
            lambda avg, batch: (tree_add(avg, batched_f(batch)), None),
            sum_init,
            batches,
            **kwargs,
        )[0]


def laxmean(f, data, *args, **kwargs):
    n = tree_len(data)
    _f = lambda *args: tree_scalar_mul(1 / n, f(*args))
    return laxsum(_f, data, *args, **kwargs)
