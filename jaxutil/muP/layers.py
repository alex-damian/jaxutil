from jax import numpy as jnp
from flax import linen as nn


class muLinear(nn.Module):
    features: int
    head: bool = False
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        fan_out = self.features
        
        if not self.head:
            w_std = jnp.sqrt(1/fan_out)
            multiplier = jnp.sqrt(fan_out/fan_in)
        else:
            w_std = 1/jnp.sqrt(fan_in)
            multiplier = 1/jnp.sqrt(fan_in)
        w_init = nn.initializers.normal(stddev=w_std)
        w = self.param("w", w_init, (x.shape[-1], self.features))
        out = (x@w)*multiplier
        if self.use_bias:
            b = self.param("b", nn.initializers.zeros, (self.features,))
            out += b*jnp.sqrt(fan_out/fan_in)
        return out
