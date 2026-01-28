from functools import partial
from typing import Callable, Tuple

import jax.numpy.linalg as jla
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array

LinearOperator = Callable[[Array], Array]

_mm = partial(jnp.matmul, precision=lax.Precision.HIGHEST)


def eigh(A: Array, B: Array | None = None) -> Tuple[Array, Array]:
    A = 0.5 * (A + A.T)
    if B is None:
        evals, evecs = jla.eigh(A)
        return evals[::-1], evecs[:, ::-1]

    eps = jnp.finfo(B.dtype).eps
    B = 0.5 * (B + B.T)
    B_S, B_U = jla.eigh(B)
    limit = 10 * eps * jnp.maximum(B_S.max(), 1.0)
    s_rsqrt = jnp.where(B_S > limit, lax.rsqrt(B_S), 0.0)
    W = B_U * s_rsqrt[None, :]
    C = W.T @ A @ W
    evals, Z = jla.eigh(C)
    evals = evals[::-1]
    evecs = (W @ Z)[:, ::-1]
    return evals, evecs


def lanczos(
    A: LinearOperator,
    v0: Array,
    m: int,
    n_reortho: int = 2,
    B: LinearOperator | None = None,
    Binv: LinearOperator | None = None,
    construct_basis: bool = True,
) -> Tuple[Array, Array | None]:
    if B is None and Binv is None:
        B = Binv = lambda x: x
    else:
        assert callable(B) and callable(Binv)

    v0 /= jnp.sqrt(_mm(v0, B(v0)))

    if construct_basis:
        V = jnp.zeros((len(v0), m)).at[:, 0].set(v0)  # full basis
    else:
        V = jnp.zeros((len(v0), 2)).at[:, 1].set(v0)  # three term recurrence
        n_reortho = 1

    def step(V, i):
        v = V[:, i if construct_basis else 1]
        Av = A(v)
        w = Binv(Av)
        alpha = _mm(v, Av)
        orthogonalize = lambda _, w: w - _mm(V, _mm(V.T, B(w)))
        w = lax.fori_loop(0, n_reortho, orthogonalize, w)
        beta = jnp.sqrt(_mm(w, B(w)))
        v_next = jnp.where(beta > 0, w / beta, jnp.zeros_like(w))
        if construct_basis:
            V = jnp.where(i < m - 1, V.at[:, i + 1].set(v_next), V)
        else:
            V = jnp.stack([v, v_next], axis=1)
        return V, (alpha, beta)

    V, (a, b) = lax.scan(step, V, jnp.arange(m))
    T = jnp.diag(a) + jnp.diag(b[:-1], 1) + jnp.diag(b[:-1], -1)
    return T, V if construct_basis else None


def lobpcg(A: LinearOperator, X: Array, maxiters=100, tol=1e-6):
    k = X.shape[1]
    X, _ = jla.qr(X)
    P = jnp.zeros_like(X)

    def cond(state):
        i, _, _, is_converged, _ = state
        return jnp.logical_and(i < maxiters, ~is_converged)

    def body(state):
        i, X, P, _, _ = state
        AX = A(X)
        mu = jnp.sum(X * AX, axis=0)
        R = AX - mu * X
        V = jnp.concatenate((X, R, P), axis=1)
        V = jla.qr(V)[0]
        AV = A(V)
        evals, C = eigh(V.T @ AV)
        evals, C = evals[:k], C[:, :k]
        X_new = V @ C
        res_norm = jla.norm(R, axis=0)
        is_converged = jnp.all(res_norm < tol)
        return i + 1, X_new, X_new - X, is_converged, evals

    init_state = (0, X, P, False, jnp.zeros(k))
    _, X, _, _, evals = lax.while_loop(cond, body, init_state)
    return evals, X
