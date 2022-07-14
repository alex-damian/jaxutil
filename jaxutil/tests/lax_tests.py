import pytest
from jax import numpy as jnp
from jaxutil.lax import *


@pytest.mark.parametrize("backend", ["python", "lax"])
@pytest.mark.parametrize("show_progress", [True, False])
@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("save_every", [1, 2, 3])
def fold_nodata(backend, show_progress=True, save_every=0):
    f = lambda x: (x, x + 1)
    state, save = fold(
        f,
        0.0,
        steps=10,
        backend=backend,
        show_progress=show_progress,
        save_every=save_every,
    )
    assert state == save_every * (10 // save_every)
    assert jnp.allclose(save, jnp.arange(0, 10, save_every))


@pytest.mark.parametrize("backend", ["python", "lax"])
@pytest.mark.parametrize("show_progress", [True, False])
@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("save_every", [1, 2, 3])
def fold_nodata(backend, show_progress, save_every):
    f = lambda _, x: (x, x + 1)
    state, save = fold(
        f,
        0.0,
        jnp.arange(10),
        backend=backend,
        show_progress=show_progress,
        save_every=save_every,
    )
    assert state == save_every * (10 // save_every)
    assert jnp.allclose(save, jnp.arange(0, 10, save_every))
