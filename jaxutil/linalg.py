from jax import lax
from jax import numpy as jnp
from jax import vmap
from jax.numpy import linalg as jla

cos_dist = lambda x, y: (x @ y) / (jla.norm(x) * jla.norm(y))


def get_ridge_fn(x, y):
    dtype = x.dtype
    m, n = x.shape
    rcond = jnp.finfo(dtype).eps * max(n, m)
    u, s, vt = jla.svd(x, full_matrices=False)
    uTy = jnp.matmul(u.conj().T, y, precision=lax.Precision.HIGHEST)

    def ridge_problem(reg=None, rel_reg=None):
        if (reg is None) * (rel_reg is None):
            raise Exception("Exactly one of reg, rel_reg can be provided.")
        if rel_reg is not None:
            reg = rel_reg * s[0]
        mask = s[0] * (s * s + reg) >= rcond * s * (s[0] * s[0] + reg)
        safe_s = jnp.where(mask, s, 1)
        s_ridge = jnp.where(mask, safe_s / (safe_s**2 + reg), 0)
        return jnp.matmul(vt.conj().T, s_ridge * uTy, precision=lax.Precision.HIGHEST)

    return ridge_problem


def orthogonalize(X, axis="col"):
    assert axis in ["row", "col"]
    if axis == "col":
        Q, R = jla.qr(X)
        Q = Q * jnp.sign(jnp.diag(R))
        return Q
    elif axis == "row":
        return orthogonalize(X.T).T
