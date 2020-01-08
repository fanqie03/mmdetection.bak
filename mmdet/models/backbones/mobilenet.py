from ..registry import BACKBONES
import torch.nn as nn
import torchvision as tv
from mmcv.cnn import constant_init, kaiming_init
import torch.utils.checkpoint as cp
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

import torch

@BACKBONES.register_module
class MobileNetV1(nn.Module):

    def __init__(self, arg1, arg2):
        super(MobileNetV1, self).__init__()
        pass

    def forward(self, x):  # should return a tuple
        pass

    def init_weights(self, pretrained=None):
        pass


@BACKBONES.register_module
class MobileNetV2(nn.Module):
    """
    >>> import torchvision
    >>> import torch
    >>> model = torchvision.models.mobilenet_v2().features
    >>> x = torch.empty(1, 3, 224, 224)
    >>> for i, module in enumerate(model):
    ...     x = module(x)
    ...     print(i, x.shape)
    ...
    0 torch.Size([1, 32, 112, 112])
    1 torch.Size([1, 16, 112, 112])
    2 torch.Size([1, 24, 56, 56])
    3 torch.Size([1, 24, 56, 56])
    4 torch.Size([1, 32, 28, 28])
    5 torch.Size([1, 32, 28, 28])
    6 torch.Size([1, 32, 28, 28])
    7 torch.Size([1, 64, 14, 14])
    8 torch.Size([1, 64, 14, 14])
    9 torch.Size([1, 64, 14, 14])
    10 torch.Size([1, 64, 14, 14])
    11 torch.Size([1, 96, 14, 14])
    12 torch.Size([1, 96, 14, 14])
    13 torch.Size([1, 96, 14, 14])
    14 torch.Size([1, 160, 7, 7])
    15 torch.Size([1, 160, 7, 7])
    16 torch.Size([1, 160, 7, 7])
    17 torch.Size([1, 320, 7, 7])
    18 torch.Size([1, 1280, 7, 7])
    """
    def __init__(self, out_indices=(3, 6, 13, 18)):
        super(MobileNetV2, self).__init__()
        self.features = tv.models.mobilenet_v2().features
        self.out_indices = out_indices

    def forward(self, x):  # should return a tuple

        outs = []
        for i, module in enumerate(self.features):
            x = module(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            from mmdet.apis import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')


@BACKBONES.register_module
class MobileNetV3(nn.Module):

    def __init__(self, arg1, arg2):
        super(MobileNetV3, self).__init__()
        pass

    def forward(self, x):  # should return a tuple
        pass

    def init_weights(self, pretrained=None):
        pass


if __name__ == '__main__':
    model = MobileNetV2()
    x = torch.empty(1, 3, 224, 224)
    for i, module in enumerate(model):
        x = module(x)
        print(i, x.shape)