from jax import jacobian, jvp


def D(f, x, order=1, *args):
    if order == 0:
        return f(x)
    elif len(args) < order:
        _f = jacobian(f)
        args = args
    else:
        v, *args = args
        _f = lambda x: jvp(f, (x,), (v,))[1]
    return D(_f, x, order - 1, *args)
