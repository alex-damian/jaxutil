from typing import Callable, Sequence

import jax
from flax import nnx
from jax import numpy as jnp
from jax import vmap
from jax.nn import *
from jaxtyping import Array

is_param = lambda x: isinstance(x, nnx.Param)


def scale_lr(tree, scale):
    def scale_param(param):
        lr = param.get_metadata("lr", 1.0)
        return param.replace(lr=lr * scale)

    return jax.tree.map(scale_param, tree, is_leaf=is_param)


def scale_by_lr(x, power: float = 1.0):
    return jax.tree.map(lambda x: x.copy(x * x.lr**power), x, is_leaf=is_param)


class Linear(nnx.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        head: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        w_std = 1 / d_in if head else 1 / jnp.sqrt(d_in)
        self.w = nnx.Param(rngs.normal((d_in, d_out)) * w_std, lr=d_out / d_in)
        if bias:
            self.b = nnx.Param(jnp.zeros((d_out,)), lr=d_out)

    def __call__(self, x: Array):
        y = x @ self.w[...]
        if hasattr(self, "b"):
            y = y + self.b[...]
        return y


class MLP(nnx.Sequential):
    def __init__(self, widths: Sequence[int], activation: Callable, *, rngs: nnx.Rngs):
        layers = []
        for d_in, d_out in zip(widths[:-2], widths[1:-1]):
            layers.append(Linear(d_in, d_out, rngs=rngs))
            layers.append(activation)
        layers.append(Linear(widths[-2], widths[-1], head=True, rngs=rngs))
        super().__init__(*layers)


class Residual(nnx.Module):
    def __init__(
        self,
        block: Callable[[nnx.Rngs], nnx.Module],
        depth: int,
        unroll=True,
        *,
        rngs: nnx.Rngs,
    ):
        self.blocks = vmap(block)(rngs.fork(split=depth))
        self.blocks = scale_lr(self.blocks, scale=depth)
        self.depth = depth
        self.unroll = unroll

    def __call__(self, x: Array):
        def step_fn(x, block):
            return x + block(x) / self.depth, None

        return jax.lax.scan(step_fn, x, self.blocks, unroll=self.unroll)[0]
