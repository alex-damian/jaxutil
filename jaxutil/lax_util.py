import jax
from jax import eval_shape, lax, vmap
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_unflatten
from tqdm.auto import tqdm, trange

tree_len = lambda tree: len(tree_leaves(tree)[0])
tree_add = lambda ta, tb: tree_map(lambda a, b: a + b, ta, tb)


def tree_stack(trees):
    _, treedef = tree_flatten(trees[0])
    leaf_list = [tree_flatten(tree)[0] for tree in trees]
    leaf_stacked = [jnp.stack(leaves) for leaves in zip(*leaf_list)]
    return tree_unflatten(treedef, leaf_stacked)


def batch_split(batch, n_batch=None, batch_size=None, strict=True):
    n = tree_len(batch)

    if batch_size is not None:
        assert n_batch is None
        n_batch = n // batch_size
    else:
        assert n_batch is not None
        batch_size = n // n_batch

    if strict:
        assert n_batch * batch_size == n

    batch = tree_map(lambda x: x[: n_batch * batch_size], batch)
    return tree_map(lambda x: x.reshape((n_batch, batch_size, *x.shape[1:])), batch)


def fold(
    f,
    state,
    xs=None,
    length=None,
    backend="lax",
    jit=True,
    show_progress=False,
    save_every=1,
):
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
            n_steps = tree_len(xs)

            def scan_fn(state, aux):
                _, batch = aux
                x0 = tree_map(lambda x: x[0], batch)
                state, save = f(state, x0)
                state = lax.scan(fast_step, state, tree_map(lambda x: x[1:], batch))[0]
                return state, save
        else:
            n_steps = n
            scan_fn = lambda state, aux: f(state, aux[1])

        if show_progress:
            with tqdm(total=n_steps) as pbar:

                def update_pbar():
                    pbar.update(1)

                def pbar_scan_fn(state, aux):
                    result = scan_fn(state, aux)
                    jax.debug.callback(update_pbar)
                    return result

                result = lax.scan(pbar_scan_fn, state, (jnp.arange(n_steps), xs))
        else:
            result = lax.scan(scan_fn, state, (jnp.arange(n_steps), xs))

        return result

    raise ValueError(f"Unknown backend: {backend}")


def laxmap(f, xs, batch_size=None, **kwargs):
    if batch_size is None:
        return fold(lambda _, x: (None, f(x)), None, xs, **kwargs)[1]

    batches = batch_split(xs, batch_size=batch_size)
    batched_out = fold(
        lambda _, batch: (None, vmap(f)(batch)), None, batches, **kwargs
    )[1]
    return tree_map(lambda x: x.reshape(-1, *x.shape[2:]), batched_out)


def laxsum(f, data, batch_size=None, **kwargs):
    x0 = tree_map(lambda x: x[0], data)
    sum_init = tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), eval_shape(f, x0))
    if batch_size is None:
        return fold(
            lambda avg, x: (tree_add(avg, f(x)), None), sum_init, data, **kwargs
        )[0]

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
    _f = lambda *args: tree_map(lambda x: x / n, f(*args))
    return laxsum(_f, data, *args, **kwargs)
