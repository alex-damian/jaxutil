import jax
from jax import jacfwd, jacrev, jvp
from jax import numpy as jnp
from jax.tree_util import tree_leaves


def smart_jacobian(f, has_aux=False):
    def jacfun(*args, **kwargs):
        inputs = args[0]
        out_shape = jax.eval_shape(f, *args, **kwargs)
        out_shape = out_shape[0] if has_aux else out_shape

        if isinstance(out_shape, jax.ShapeDtypeStruct) and out_shape.shape == ():
            return jax.grad(f, has_aux=has_aux)(*args, **kwargs)
        else:
            in_dim = sum(jnp.size(leaf) for leaf in tree_leaves(inputs))
            out_dim = sum(jnp.size(leaf) for leaf in tree_leaves(out_shape))
            if in_dim >= out_dim:
                return jacrev(f, has_aux=has_aux)(*args, **kwargs)
            else:
                return jacfwd(f, has_aux=has_aux)(*args, **kwargs)

    return jacfun


def D(f, x, order=1, *vs, return_all=False):
    if return_all:

        def _f(x):
            out = f(x)
            return out, (out,)

        return _D(_f, x, order, *vs, return_all=return_all)[1]
    else:
        _f = f
        return _D(_f, x, order, *vs, return_all=return_all)


def _D(f, x, order=1, *vs, return_all=False):
    assert len(vs) <= order
    if order == 0:
        return f(x)
    elif len(vs) == order:
        v, *vs = vs
    else:
        v = None

    def Df(x):
        if return_all:
            if v is None:
                jac, hist = smart_jacobian(f, has_aux=True)(x)
                return jac, (*hist, jac)
            else:
                _, jac, hist = jvp(f, [x], [v], has_aux=True)
                return jac, (*hist, jac)
        else:
            if v is None:
                return smart_jacobian(f)(x)
            else:
                return jvp(f, [x], [v])[1]

    return _D(Df, x, order - 1, *vs, return_all=return_all)
