from jax.flatten_util import ravel_pytree

def flat_init(model, rng, x, with_state=False):
    variables = model.init(rng, x)
    if with_state:
        state, params = variables.pop("params")
        flatparams, unravel = ravel_pytree(params)
        def apply(params, state, x, **kwargs):
            return model.apply({"params": unravel(params), **state}, x, **kwargs)
        return flatparams, state, apply, unravel
    else:
        flatparams, unravel = ravel_pytree(variables)
        def apply(params, x, **kwargs):
            return model.apply(unravel(params), x, **kwargs)
        return flatparams, apply, unravel