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


def diff(f, x, order=1, *vs):
    assert len(vs) <= order
    if order == 0:
        return f(x)
    elif len(vs) == order:
        v, *vs = vs
        Df = lambda x: jax.jvp(f, [x], [v])[1]
    else:
        Df = jax.jacobian(f)
    return diff(Df, x, order - 1, *vs)
