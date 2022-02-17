import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite_e import herme2poly

He = lambda k, x: jnp.polyval(herme2poly([0] * k + [1])[::-1], x)
