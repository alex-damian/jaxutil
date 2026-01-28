import numpy as np
from datasets import load_dataset
from jax import lax
from jax import numpy as jnp
from jax import random, vmap


def OneHot(x, num_classes):
    return np.eye(num_classes)[x]


def CIFAR_Normalize(x):
    CIFAR10_MEAN = np.array([125.30691805, 122.95039414, 113.86538318])
    CIFAR10_STD = np.array([62.99321928, 62.08870764, 66.70489964])
    x = np.asarray(x, dtype=np.float32)
    x = (x - CIFAR10_MEAN) / CIFAR10_STD
    return x


def cifar10(normalize=True, one_hot=False):
    ds = load_dataset("cifar10")
    train_x, train_y = np.stack(ds["train"]["img"]), np.stack(ds["train"]["label"])  # type: ignore
    test_x, test_y = np.stack(ds["test"]["img"]), np.stack(ds["test"]["label"])  # type: ignore
    if normalize:
        train_x, test_x = CIFAR_Normalize(train_x), CIFAR_Normalize(test_x)
    if one_hot:
        train_y, test_y = OneHot(train_y, num_classes=10), OneHot(
            test_y, num_classes=10
        )
    return train_x, train_y, test_x, test_y


def cifar100(normalize=True, one_hot=False):
    ds = load_dataset("cifar100")
    train_x, train_y = np.stack(ds["train"]["img"]), np.stack(ds["train"]["label"])  # type: ignore
    test_x, test_y = np.stack(ds["test"]["img"]), np.stack(ds["test"]["label"])  # type: ignore
    if normalize:
        train_x, test_x = CIFAR_Normalize(train_x), CIFAR_Normalize(test_x)
    if one_hot:
        train_y, test_y = OneHot(train_y, num_classes=100), OneHot(
            test_y, num_classes=100
        )
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
