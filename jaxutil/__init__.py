from .derivatives import diff, smart_jacobian
from .lax_util import batch_split, fold, laxmap, laxmean, laxsum, tree_stack
from .linalg import cos_dist, get_ridge_fn, orthogonalize
from .poly import He, factorial, factorial2
from .util import RNG, ddict, enable_maxtext_gpu_flags, tree_to_dict