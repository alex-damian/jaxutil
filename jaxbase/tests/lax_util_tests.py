import pytest
from jax import numpy as jnp
from jaxbase.lax_util import *


@pytest.mark.parametrize("backend", ["python", "lax"])
def test_nodata(backend):
    f = lambda x: dict(state=x + 1, save=x, avg=x, add=x)
    fold_out = fold(f, 0.0, steps=10, backend=backend)
    assert fold_out["state"] == 10
    assert jnp.allclose(fold_out["save"], jnp.arange(10))
    assert fold_out["avg"] == 4.5
    assert fold_out["add"] == 45


@pytest.mark.parametrize("backend", ["python", "lax"])
def test_data(backend):
    data = jnp.arange(10)
    f = lambda _, x: dict(state=x + 1, save=x, avg=x, add=x)
    fold_out = fold(f, 0, data, backend=backend)
    assert fold_out["state"] == 10
    assert jnp.allclose(fold_out["save"], data)
    assert fold_out["avg"] == 4.5
    assert fold_out["add"] == 45


@pytest.mark.parametrize("backend", ["python"])
def test_save_every(backend):
    f = lambda x: dict(state=x + 1, save=x, avg=x, add=x)
    fold_out = fold(f, 0, steps=10, backend=backend, save_every=2)
    assert fold_out["state"] == 10
    assert jnp.allclose(fold_out["save"], jnp.arange(0, 10, 2))
    assert fold_out["avg"] == 4.5
    assert fold_out["add"] == 45
