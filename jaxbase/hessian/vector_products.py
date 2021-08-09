from jax import numpy as jnp
from jax import jit, grad, vmap, vjp, jvp


def Hvp(f, criterion, params, data, v):
    def loss(params, data):
        x, y = data
        return jnp.mean(vmap(criterion)(f(params, x), y))

    return jvp(grad(lambda params: loss(params, data)), [params], [v])[1]


def Gvp(f, criterion, params, data, v):
    x, y = data
    fi = lambda params: f(params, x)
    outputs, Jv = jvp(fi, [params], [v])
    HJv = grad(lambda f: jnp.vdot(vmap(grad(criterion))(f, y), Jv))(outputs) / len(x)
    JtHJv = vjp(fi, params)[1](HJv)[0]
    return JtHJv


def Evp(f, criterion, params, data, v):
    return Hvp(f, criterion, params, data, v) - Gvp(f, criterion, params, data, v)
