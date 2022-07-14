import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite_e import herme2poly

factorial = lambda k: jnp.prod(jnp.arange(k, 0, -1))
factorial2 = lambda k: jnp.prod(jnp.arange(k, 0, -2))
He = lambda k, x: jnp.polyval(herme2poly([0] * k + [1])[::-1], x)
