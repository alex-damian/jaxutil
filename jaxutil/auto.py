import jax
from jax import numpy as jnp, vmap, jit, lax, random as jr, tree_util as jtu
from jax.numpy import linalg as jla
from flax import linen as nn
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from tqdm.auto import tqdm, trange
from collections import namedtuple
from functools import partial
from .derivatives import smart_jacobian, D
from jaxopt.tree_util import *
from typing import (
    Any,
    Callable,
    Union,
    List,   
)
from .lax_util import (
    tree_stack,
    batch_split,
    fold,
    laxmap,
    laxsum,
    laxmean,
)
from .linalg import (
    cos_dist,
    get_ridge_fn,
    orthogonalize,
)
from .poly import (
    factorial,
    factorial2,
    He,
)
from .util import (
    RNG,
    flat_init,
    ddict,
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
