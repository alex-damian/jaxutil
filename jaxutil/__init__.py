import jax
from jax import numpy as jnp, vmap, jit, lax, random
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange

from derivatives import D
from lax import (
    batch_split,
    fold,
    laxmap,
    laxsum,
    laxmean,
)
from linalg import (
    cos_dist,
    ridge,
    eigsh,
    orthogonalize,
)
from poly import (
    factorial,
    factorial2,
    He,
)
from tree import (
    tree_map,
    tree_idx,
    tree_len,
    tree_zeros,
    tree_add,
    tree_mul,
    tree_stack,
)
from util import (
    RNG,
    flat_init,
    qt,
    clean_dict,
    unpack,
)
from setup import jax_setup
