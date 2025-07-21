from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Union

import jax
import jaxopt.tree_util as jtu
import numpy as np
import seaborn as sns
import treescope
from jax import jit, lax, nn
from jax import numpy as jnp
from jax import random as jr
from jax import vmap
from jax.flatten_util import ravel_pytree
from jax.numpy import linalg as jla
from matplotlib import pyplot as plt
from numpy import linalg as la
from scipy.stats import linregress
from tqdm.auto import tqdm, trange
from .derivatives import D, smart_jacobian
from .lax_util import batch_split, fold, laxmap, laxmean, laxsum, tree_stack
from .linalg import cos_dist, get_ridge_fn, orthogonalize
from .poly import He, factorial, factorial2
from .types import (
    bf16,
    c64,
    c128,
    f16,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    ui8,
    ui16,
    ui32,
    ui64,
)
from .util import RNG, ddict

treescope.basic_interactive_setup(autovisualize_arrays=True)