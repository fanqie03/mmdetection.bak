import argparse
import os
import os.path as osp
import shutil
import tempfile
import time
import numpy as np
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint
import cv2
import pandas as pd


from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--out', help='output result file')

    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    # build the dataloader
    table = []

    for type in ['val', 'train']:
        dataset = build_dataset(cfg.data[type])
        print(len(dataset))
        for i in range(len(dataset)):

            d = dataset.img_infos[i]
            n = next(iter(dataset))
            # filename = n['img_meta'][0].data['filename']
            # ann = d['ann']
            ann = dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            filename = dataset.img_prefix + d['filename']



            for bbox, label in zip(bboxes, labels):
                width = bbox[2]-bbox[0]
                height = bbox[3]-bbox[1]
                row = [
                    type,
                    d['filename'],
                    d['width'],
                    d['height'],
                    label,

                    *bbox.tolist(),
                    width,
                    height
                ]

                table.append(row)

        table = np.asarray(table)
        table = pd.DataFrame(table)
        table.to_csv(args.out)
        pass


if __name__ == '__main__':
    main()
