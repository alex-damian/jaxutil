from jax import numpy as jnp
from jax.numpy import linalg as jla
from jax import random
from scipy.sparse.linalg import eigsh, LinearOperator, ArpackNoConvergence


def trace(A, dim, rng, num_iter=200):
    tr = 0
    var = 0
    for i in range(num_iter):
        rng, subrng = random.split(rng)
        v = random.rademacher(subrng, (dim,), dtype=jnp.float32)
        estimate = v @ A(v)
        tr += estimate / num_iter
        var += estimate**2 / num_iter
    var = var - (tr**2)
    stderr = jnp.sqrt(var / num_iter)
    return tr, stderr


def frobenius_norm(A, dim, rng, num_iter=200):
    A2 = lambda v: A(A(v))
    return trace(A2, dim, rng, num_iter)


def power_iter(A, dim, rng, num_iter=100):
    v = random.normal(rng, (dim,))
    v = v / jla.norm(v)
    for _ in range(num_iter):
        Av = A(v)
        v = Av / jla.norm(Av)
    return v @ A(v), v


def scipy_iter(A, dim, rng, tol=1e-2):
    A_scipy = LinearOperator((dim, dim), A)
    v = random.normal(rng, (dim,))
    eig = eigsh(A_scipy, k=1, return_eigenvectors=False, tol=tol, v0=v)[0]
    return eig


def sharpness(A, dim, rng, method="scipy"):
    assert method in ["scipy", "power_iter"]
    if method == "scipy":
        try:
            return scipy_iter(A, dim, rng)
        except ArpackNoConvergence:
            print("Arpack did not converge, reverting to power iteration")
    return power_iter(A, dim, rng)


def extreme_eigs(A, dim, rng, method="scipy"):
    eig1 = sharpness(A, dim, rng, method="scipy")
    A_shift = lambda v: A(v) - eig1 * v
    eig2 = sharpness(A_shift, dim, rng, method="scipy") + eig1
    return sorted((eig1, eig2))
