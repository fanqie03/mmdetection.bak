from ..registry import BACKBONES
import torch.nn as nn
import torchvision as tv
from mmcv.cnn import constant_init, kaiming_init
import torch.utils.checkpoint as cp
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
import torch.functional as F

import torch


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1,
                      dilation=vision + 1, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2,
                      dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1,
                      groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4,
                      dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                  groups=in_channels, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


@BACKBONES.register_module
class Mb_Tiny_RFB(nn.Module):
    """
    (640, 480)
    0 torch.Size([1, 16, 320, 240])  # /2
    1 torch.Size([1, 32, 320, 240])
    2 torch.Size([1, 32, 160, 120])  # /4
    3 torch.Size([1, 32, 160, 120])
    4 torch.Size([1, 64, 80, 60])  # /8
    5 torch.Size([1, 64, 80, 60])
    6 torch.Size([1, 64, 80, 60])
    7 torch.Size([1, 64, 80, 60])
    8 torch.Size([1, 128, 40, 30])  # /16
    9 torch.Size([1, 128, 40, 30])
    10 torch.Size([1, 128, 40, 30])  # /16
    11 torch.Size([1, 256, 20, 15])
    12 torch.Size([1, 256, 20, 15])  # /32
    13 torch.Size([1, 256, 10, 8])  # /64
    """

    def __init__(self, out_indices=(3, 7, 10, 12)):
        super(Mb_Tiny_RFB, self).__init__()
        self.out_indices = out_indices
        self.base_channel = 8 * 2

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def extras(base_channel):
            return nn.Sequential(
                nn.Conv2d(in_channels=base_channel * 16, out_channels=base_channel * 4, kernel_size=1),
                nn.ReLU(),
                SeperableConv2d(in_channels=base_channel * 4, out_channels=base_channel * 16,
                                kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )

        self.features = nn.Sequential(
            conv_bn(3, self.base_channel, 2),  # 160*120
            conv_dw(self.base_channel, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 2, 2),  # 80*60
            conv_dw(self.base_channel * 2, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 4, 2),  # 40*30
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            BasicRFB(self.base_channel * 4, self.base_channel * 4, stride=1, scale=1.0),
            conv_dw(self.base_channel * 4, self.base_channel * 8, 2),  # 20*15
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 16, 2),  # 10*8
            conv_dw(self.base_channel * 16, self.base_channel * 16, 1),
            extras(self.base_channel)
        )

    def forward(self, x):
        outs = []
        for i, module in enumerate(self.features):
            x = module(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

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


if __name__ == '__main__':
    input_sizes = [(640, 480), (320, 240), (160, 120), (80, 60), (224, 224), (112, 112)]
    for input_size in input_sizes:
        data = torch.empty(1, 3, *input_size)
        model = Mb_Tiny_RFB(3, list(range(18)))
        r = model(data)
        print(r.shape)
        # flops, params = get_model_complexity_info(model, (3,) + input_size, print_per_layer_stat=True, as_strings=True)
        # print(flops, params)
