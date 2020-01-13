import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import AnchorGenerator, anchor_target, multi_apply
from ..losses import smooth_l1_loss
from ..registry import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module
class YoloV3Head(AnchorHead):
    pass
