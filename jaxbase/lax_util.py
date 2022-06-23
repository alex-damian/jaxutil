import jax
from jax import numpy as jnp, vmap, tree_map, eval_shape
from jax import lax
from jax.experimental.host_callback import id_tap
from tqdm.auto import tqdm, trange
from jaxbase.tree_util import *


def unpack(f):
    return lambda args: f(*args)


def batch_split(
    batch,
    n_batch: int = None,
    batch_size: int = None,
    devices: tuple = None,
    strict=True,
):
    if devices is not None:
        batches = batch_split(batch, n_batch=len(devices))
        batches = tree_map(lambda x: jax.device_put_sharded(list(x), devices), batches)
        return batches

    n = len(jax.tree_leaves(batch)[0])

    if n_batch is not None:
        batch_size = n // n_batch
    elif batch_size is not None:
        n_batch = n // batch_size

    if strict:
        assert n_batch * batch_size == n
    else:
        batch = tree_map(lambda x: x[: n_batch * batch_size], batch)

    batches = tree_map(lambda x: x.reshape((n_batch, batch_size, *x.shape[1:])), batch)
    return batches


def fold_clean(f, save=True):
    def clean_f(state, batch):
        fout = f(state, batch)
        for key in ["state", "save", "avg", "add"]:
            if key not in fout:
                fout[key] = None
        if save == False:
            fout["save"] = None
        return fout

    return clean_f


def fold(
    f,
    state,
    data=None,
    steps=None,
    backend="python",
    jit=False,
    show_progress=False,
    save_every=1,
):
    n = tree_len(data) if data is not None else steps
    x0 = tree_idx(data, 0) if data is not None else None
    f_save = (lambda state, _: f(state)) if data is None else f
    f_nosave = fold_clean(f_save, save=False)
    f_save = fold_clean(f_save, save=True)
    if jit:
        f_save = jax.jit(f_save)
        f_nosave = jax.jit(f_nosave)

    if backend == "lax":
        if save_every != 1:
            raise Exception("save_every is not compatible with backend lax")
        out_tree = jax.eval_shape(lambda args: f_save(*args), (state, x0))
        avg_init = tree_zeros(out_tree["avg"])
        avg_init = tree_map(lambda x: x * 1.0, avg_init)
        add_init = tree_zeros(out_tree["add"])
        if show_progress:
            pbar = tqdm(total=n)

        def step(carry, batch):
            state, avg, add = carry
            fout = f_save(state, batch)
            batch_state = fout["state"]
            batch_save = fout["save"]
            avg = tree_map(lambda si, fi: si + fi / n, avg, fout["avg"])
            add = tree_map(lambda si, fi: si + fi, add, fout["add"])
            if show_progress:
                id_tap(lambda *_: pbar.update(), None)
            return (batch_state, avg, add), batch_save

        (state, avg, add), save = lax.scan(
            step, (state, avg_init, add_init), xs=data, length=steps
        )
        if show_progress:
            pbar.close()
        return dict(state=state, save=save, avg=avg, add=add)
    elif backend == "python":
        iterator = trange(n) if show_progress else range(n)
        avg = None
        add = None
        save = []
        for i in iterator:
            batch = tree_idx(data, i) if data is not None else None
            if i % save_every == 0:
                fout = f_save(state, batch)
                save.append(fout["save"])
            else:
                fout = f_nosave(state, batch)
            state = fout["state"]
            if avg is None:
                avg = tree_map(lambda si: si / n, fout["avg"])
            else:
                avg = tree_map(lambda si, fi: si + fi / n, avg, fout["avg"])
            if add is None:
                add = fout["add"]
            else:
                add = tree_map(lambda si, fi: si + fi, add, fout["add"])
        save = tree_stack(save)
        return dict(state=state, save=save, avg=avg, add=add)


def laxmap(f, data, batch_size=None, **kwargs):
    if batch_size == None:
        return fold(lambda _, x: dict(save=f(x)), None, data, **kwargs)["save"]
    else:
        batches = batch_split(data, batch_size=batch_size)
        batched_out = fold(
            lambda _, batch: dict(save=vmap(f)(batch)), None, batches, **kwargs
        )["save"]
        flat_out = tree_map(lambda x: x.reshape(-1, *x.shape[2:]), batched_out)
        return flat_out


def laxmean(f, data, batch_size=None, **kwargs):
    if batch_size == None:
        return fold(lambda _, x: dict(avg=f(x)), None, data, **kwargs)["avg"]
    else:

        def batched_f(batch):
            out_tree = vmap(f)(batch)
            reduced_tree = tree_map(lambda x: x.mean(0), out_tree)
            return dict(avg=reduced_tree)

        batches = batch_split(data, batch_size=batch_size)
        return fold(lambda _, batch: batched_f(batch), None, batches, **kwargs)["avg"]


def laxsum(f, data, batch_size=None, **kwargs):
    if batch_size == None:
        return fold(lambda _, x: dict(add=f(x)), None, data, **kwargs)["avg"]
    else:

        def batched_f(batch):
            out_tree = vmap(f)(batch)
            reduced_tree = tree_map(lambda x: x.sum(0), out_tree)
            return dict(add=reduced_tree)

        batches = batch_split(data, batch_size=batch_size)
        return fold(lambda _, batch: batched_f(batch), None, batches, **kwargs)["avg"]
