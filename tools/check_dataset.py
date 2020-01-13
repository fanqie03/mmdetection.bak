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


from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--out', help='output result file')

    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')

    parser.add_argument('--clazz', nargs='+', type=int)

    args = parser.parse_args()
    return args


def draw_img(img_prefix, info, ann, args):

    bboxes = ann['bboxes']
    labels = ann['labels']
    filename = img_prefix + info['filename']

    table = []

    img = cv2.imread(filename)
    raw_img = img.copy()
    for bbox, label in zip(bboxes, labels):

        bbox = bbox.astype(np.int32)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), )
        cv2.putText(img, str(label), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), bottomLeftOrigin=True)

        sub = raw_img[bbox[1]:bbox[3], bbox[0]: bbox[2], :]

        # cv2.imshow('',sub)
        # cv2.waitKey()
        if args.out and min(bbox[3] - bbox[1], bbox[2] - bbox[0]) >= 24:
            parent = os.path.join(args.out, str(label))
            if not os.path.exists(parent):
                os.makedirs(parent)
            out_path = os.path.join(parent, str(time.time()) + '.jpg')
            cv2.imwrite(out_path, sub)

    bboxes = ann['bboxes_ignore']
    labels = ann['labels_ignore']
    filename = img_prefix + info['filename']

    table = []

    # img = cv2.imread(filename)
    # raw_img = img.copy()
    for bbox, label in zip(bboxes, labels):

        bbox = bbox.astype(np.int32)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), )
        cv2.putText(img, str(label), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), bottomLeftOrigin=True)

        sub = raw_img[bbox[1]:bbox[3], bbox[0]: bbox[2], :]

        # cv2.imshow('',sub)
        # cv2.waitKey()
        if args.out and min(bbox[3] - bbox[1], bbox[2] - bbox[0]) >= 24:
            parent = os.path.join(args.out, str(label))
            if not os.path.exists(parent):
                os.makedirs(parent)
            out_path = os.path.join(parent, str(time.time()) + '.jpg')
            cv2.imwrite(out_path, sub)


    if args.show:
        cv2.imshow('', img)
        cv2.waitKey()


def get_class_info(datasets, clazz):
    """
    获取特定的clazz的图片
    :param clazz:
    :return:
    """
    if not isinstance(clazz, list):
        clazz = [clazz]
    for i in range(datasets.__len__()):
        flag = False
        info = datasets.img_infos[i]
        ann = datasets.get_ann_info(i)
        labels = ann['labels']
        labels_ignore = ann['labels_ignore']
        for label in labels:
            if label in clazz:
                flag = True
                break
        for label in labels_ignore:
            if label in clazz:
                flag = True
                break
        if flag:
            yield info, ann


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    # build the dataloader

    for type in ['val', 'train']:
        dataset = build_dataset(cfg.data[type])
        print(len(dataset))

        if args.clazz is not None:
            for info, ann in get_class_info(dataset, args.clazz):
                draw_img(dataset.img_prefix, info, ann, args)
        else:
            for result in dataset:
                print(result)
            # for i in range(len(dataset)):
            #     info = dataset.img_infos[i]
            #     ann = dataset.get_ann_info(i)
            #     draw_img(dataset.img_prefix, info, ann, args)


if __name__ == '__main__':
    main()
