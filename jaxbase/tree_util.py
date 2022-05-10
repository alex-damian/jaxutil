import jax
from jax import numpy as jnp

tree_idx = lambda x, i: jax.tree_map(lambda x: x[i], x)
tree_len = lambda tree: len(jax.tree_leaves(tree)[0])

tree_zeros = lambda tree: jax.tree_map(
    lambda leaf: jnp.zeros(shape=leaf.shape, dtype=leaf.dtype), tree
)


def tree_stack(trees):
    _, treedef = jax.tree_flatten(trees[0])
    leaf_list = [jax.tree_flatten(tree)[0] for tree in trees]
    leaf_stacked = [jnp.stack(leaves) for leaves in zip(*leaf_list)]
    return jax.tree_unflatten(treedef, leaf_stacked)
