import argparse

import torch
from mmcv import Config
from mmdet.models import build_detector
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(cfg.work_dir)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
                format(model.__class__.__name__))

    img = torch.ones((1,) + input_shape).cuda()
    writer.add_graph(model, img)
