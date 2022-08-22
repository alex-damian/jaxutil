import jax
from jax import numpy as jnp, eval_shape
from jax.tree_util import tree_map, tree_leaves, tree_flatten, tree_unflatten
from jax.flatten_util import ravel_pytree

tree_idx = lambda x, i: tree_map(lambda x: x[i], x)
tree_len = lambda tree: len(tree_leaves(tree)[0])

tree_zeros = lambda tree: tree_map(
    lambda leaf: jnp.zeros(shape=leaf.shape, dtype=leaf.dtype), tree
)

tree_add = lambda x,y: tree_map(lambda x,y: x+y, x,y)
tree_mul = lambda x, c: tree_map(lambda xi: xi * c, x)


def tree_stack(trees):
    _, treedef = tree_flatten(trees[0])
    leaf_list = [tree_flatten(tree)[0] for tree in trees]
    leaf_stacked = [jnp.stack(leaves) for leaves in zip(*leaf_list)]
    return tree_unflatten(treedef, leaf_stacked)
