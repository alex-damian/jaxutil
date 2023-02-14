import jax
from jax import numpy as jnp, lax
import numpy as np
from numpy.polynomial.hermite_e import herme2poly


factorial = lambda n: jnp.prod(jnp.arange(n, 0, -1).astype(float))
factorial2 = lambda n: jnp.prod(jnp.arange(n, 0, -2).astype(float))
He = lambda k, x: jnp.polyval(herme2poly([0] * k + [1])[::-1], x)
