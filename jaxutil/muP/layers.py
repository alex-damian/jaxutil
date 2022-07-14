from jax import numpy as jnp
from flax import linen as nn


class muLinear(nn.Module):
    features: int
    head: bool = False
    gain: float = 2

    @nn.compact
    def __call__(self, x):
        if self.head:
            scales = (1 / x.shape[-1], 1 / jnp.sqrt(self.features))
        else:
            scales = (jnp.sqrt(self.gain / x.shape[-1]), 1 / jnp.sqrt(self.features))
        w_init = nn.initializers.normal(stddev=jnp.sqrt(scales[0] * scales[1]))
        w = self.param("w", w_init, (x.shape[-1], self.features))
        b = self.param("b", nn.initializers.zeros, (self.features,))
        return (x @ w) * jnp.sqrt(scales[0] / scales[1]) + b
