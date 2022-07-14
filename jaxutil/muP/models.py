from .layers import muLinear
from typing import Callable, List
from flax import linen as nn


class muMLP(nn.Module):
    widths: List
    sigma: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for width in self.widths[:-1]:
            x = muLinear(width)(x)
            x = self.sigma(x)
        x = muLinear(self.widths[-1], head=True)(x)
        return x
