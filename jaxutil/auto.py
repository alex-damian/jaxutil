from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Union

import equinox as eqx
import jax
import jax.numpy.linalg as jla
import optax
from jax import jit, lax, vmap, nn
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
from jax.flatten_util import ravel_pytree
from matplotlib import pyplot as plt
from numpy import linalg as la
import numpy as np

from jaxutil import *