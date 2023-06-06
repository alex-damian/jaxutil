import jax
from jax import numpy as jnp, jacrev, jacfwd, jvp
from jax.tree_util import tree_leaves


def smart_jacobian(f, argnums=0, has_aux=False, return_value=False):
    def jacfun(*args, **kwargs):
        if return_value:

            def _f(*args, **kwargs):
                if has_aux:
                    out, aux = f(*args, **kwargs)
                    return out, (out, aux)
                else:
                    out = f(*args, **kwargs)
                    return out, out

            _jacfun = smart_jacobian(
                _f, argnums=argnums, has_aux=True, return_value=False
            )
            jac, aux = _jacfun(*args, **kwargs)
            if has_aux:
                out, *aux = aux
                return out, jac, aux
            else:
                out = aux
                return out, jac

        inputs = (
            args[argnums] if isinstance(argnums, int) else [args[i] for i in argnums]
        )
        if has_aux:
            out_shape = jax.eval_shape(f, *args, **kwargs)[0]
        else:
            out_shape = jax.eval_shape(f, *args, **kwargs)

        if isinstance(out_shape, jax.ShapeDtypeStruct) and out_shape.shape == ():
            return jax.grad(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
        else:
            in_dim = sum(jnp.size(leaf) for leaf in tree_leaves(inputs))
            out_dim = sum(jnp.size(leaf) for leaf in tree_leaves(out_shape))
            if in_dim >= out_dim:
                return jacrev(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
            else:
                return jacfwd(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)

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
