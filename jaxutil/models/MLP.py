from flax import linen as nn
from .torch_layers import *
from typing import Callable, List
from flax.linen import initializers as jinit


class MLP(nn.Module):
    widths: List
    gain: float = 2
    sigma: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        init = jinit.variance_scaling(self.gain, "fan_in", "normal")
        linear = partial(nn.Dense, kernel_init=init, dtype=None)
        widths, n_classes = self.widths
        for width in widths:
            x = linear(width)(x)
            x = self.sigma(x)
        x = linear(n_classes)(x)
        return x


class CNN(nn.Module):
    widths: List
    gain: float = 2
    sigma: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        init = jinit.variance_scaling(self.gain, "fan_in", "normal")
        conv = partial(nn.Conv, kernel_init=init, dtype=None)
        linear = partial(nn.Dense, kernel_init=init, dtype=None)
        widths, n_classes = self.widths
        for width in widths:
            x = conv(width, (3, 3))(x)
            x = self.sigma(x)
        x = linear(n_classes)(x)
        return x
