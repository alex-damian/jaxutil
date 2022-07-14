import torchvision
import torchvision.transforms as T
import numpy as np
from jax import numpy as jnp
from jax import random, lax, vmap

CIFAR10_MEAN = np.array([125.30691805, 122.95039414, 113.86538318])
CIFAR10_STD = np.array([62.99321928, 62.08870764, 66.70489964])


def OneHot(x):
    return np.eye(10)[x]


def CIFAR10_Normalize(x):
    x = np.asarray(x, dtype=np.float32)
    x = (x - CIFAR10_MEAN) / CIFAR10_STD
    return x


def numpy(data_dir, download=False, normalize=True, one_hot=False):
    traindata = torchvision.datasets.CIFAR10(data_dir, train=True, download=download)
    testdata = torchvision.datasets.CIFAR10(data_dir, train=False, download=download)
    train_x, train_y, test_x, test_y = (
        traindata.data,
        traindata.targets,
        testdata.data,
        testdata.targets,
    )
    if normalize:
        train_x, test_x = CIFAR10_Normalize(train_x), CIFAR10_Normalize(test_x)
    if one_hot:
        train_y, test_y = OneHot(train_y), OneHot(test_y)
    return train_x, train_y, test_x, test_y


def data_aug(batch, rng):
    x, y = batch

    def _augment(x, flip, crops):
        x = lax.cond(flip, lambda _: x, lambda _: jnp.fliplr(x), None)
        x = jnp.pad(x, pad_width=[(4, 4), (4, 4), (0, 0)])  # pad for shifting
        x = lax.dynamic_slice(x, (*crops, 0), (32, 32, 3))
        return x

    flip_rng, crop_rng = random.split(rng)
    flips = random.uniform(flip_rng, (len(x),)) > 0.5
    crops = random.randint(crop_rng, (len(x), 2), 0, 9)
    return vmap(_augment)(x, flips, crops), y
