import os
import sys
from collections import namedtuple
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, List, Optional, Union

import equinox as eqx
import jax
import jax.numpy.linalg as jla
import numpy as np
import optax
from jax import jit, lax, nn, vmap
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
from jax.flatten_util import ravel_pytree
from jax.nn import initializers as nni
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from matplotlib import pyplot as plt
from numpy import linalg as la
from tqdm.auto import tqdm, trange

from jaxutil import *
