from jax import numpy as jnp, lax, vmap
from jax.numpy import linalg as jla
from scipy.sparse.linalg import LinearOperator, eigsh

cos_dist = lambda x, y: (x @ y) / (jla.norm(x) * jla.norm(y))


def ridge(x, y, reg=0, rel_reg=None):
    dtype = x.dtype
    m, n = x.shape
    rcond = jnp.finfo(dtype).eps * max(n, m)
    u, s, vt = jla.svd(x, full_matrices=False)
    uTy = jnp.matmul(u.conj().T, y, precision=lax.Precision.HIGHEST)
    if rel_reg is not None:
        reg = rel_reg * s[0]

    def ridge_problem(reg):
        mask = s[0] * (s * s + reg) >= rcond * s * (s[0] * s[0] + reg)
        safe_s = jnp.where(mask, s, 1)
        s_ridge = jnp.where(mask, safe_s / (safe_s**2 + reg), 0)
        return jnp.matmul(vt.conj().T, s_ridge * uTy, precision=lax.Precision.HIGHEST)

    if jnp.ndim(reg) == 0:
        return ridge_problem(reg)
    else:
        return lax.map(ridge_problem, reg)


def eigsh(A, dim, *args):
    operator = LinearOperator((dim, dim), A)
    return eigsh(operator, *args)


def gram_schmidt(*args):
    subspace = jnp.stack(args, 1)
    P = jla.qr(subspace)[0].T
    P = P * jnp.sign(vmap(jnp.dot)(P, subspace.T))[:, None]
    return P
