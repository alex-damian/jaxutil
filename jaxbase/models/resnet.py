import jax
from jax_resnet import (
    ResNet,
    ResNetStem,
    ResNetBlock,
    ConvBlock,
    ModuleDef,
    STAGE_SIZES,
)
from functools import partial
from flax import linen as nn

class CIFARStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock
    @nn.compact
    def __call__(self, x):
        return self.conv_block_cls(64, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)


def ResNet18(activation=jax.nn.relu, small_image=True, normalization="GN", **kwargs):
    config = {
        'stage_sizes': STAGE_SIZES[18],
        'block_cls': partial(ResNetBlock, activation=activation),
        'stem_cls': CIFARStem if small_image else ResNetStem
        }
    if normalization == "BN": config['norm_cls'] = partial(nn.BatchNorm, momentum=0.9)
    elif normalization == "GN": config['norm_cls'] = lambda *args, **kwargs: nn.GroupNorm(num_groups=32)
    elif normalization == None: config['norm_cls'] = lambda *args, **kwargs: lambda x: x
    else: raise Exception(f"Unknown normalization: {normalization}")
    if small_image: config['pool_fn'] = lambda x: x
    return ResNet(**config,**kwargs)
