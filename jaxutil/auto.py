import jax
from jax import numpy as jnp, vmap, jit, lax, random
from jax.numpy import linalg as jla
from jax.tree_util import (
    tree_map,
    tree_leaves,
    Partial,
)
from jax.flatten_util import ravel_pytree
from flax import linen as nn
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange
from collections import namedtuple
from functools import partial
from .derivatives import smart_jacobian, D
from .lax_util import (
    batch_split,
    fold,
    laxmap,
    laxsum,
    laxmean,
)
from .linalg import (
    cos_dist,
    ridge,
    eigsh,
    orthogonalize,
)
from .poly import (
    factorial,
    factorial2,
    He,
)
from .tree import (
    tree_idx,
    tree_len,
    tree_zeros,
    tree_add,
    tree_mul,
    tree_stack,
)
from .util import (
    RNG,
    flat_init,
    qt,
    clean_dict,
    unpack,
    print_xla,
)
from .types import (
    i8,
    i16,
    i32,
    i64,
    ui8,
    ui16,
    ui32,
    ui64,
    bf16,
    f16,
    f32,
    f64,
    c64,
    c128,
)
from .setup import jax_setup
