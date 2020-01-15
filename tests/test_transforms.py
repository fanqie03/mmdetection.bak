import torch
from mmdet.datasets.pipelines.transforms import Pad
from mmdet.datasets.pipelines.transforms import FilterBox
import numpy as np
import cv2


def test_pad():
    raw = dict(
        img=np.zeros((200, 401, 3), dtype=np.uint8)
    )
    cv2.imshow('raw', raw['img'])
    pad = Pad(square=True, pad_val=255)
    r = pad(raw)
    print(r['img'].shape)

    cv2.imshow('draw', r['img'])
    cv2.waitKey()

    raw = dict(
        img=np.zeros((402, 401, 3), dtype=np.uint8)
    )
    cv2.imshow('raw', raw['img'])
    pad = Pad(square=True, pad_val=255)
    r = pad(raw)
    print(r['img'].shape)

    cv2.imshow('draw', r['img'])
    cv2.waitKey()


def test_filter_box():
    bboxes = np.array([[0, 0, 10, 10],
                       [10, 10, 20, 20],
                       [10, 10, 19, 20],
                       [10, 10, 20, 19],
                       [10, 10, 19, 19]])
    gt_bboxes = np.array([[0, 0, 10, 9]])
    result = dict(gt_bboxes=bboxes)
    fb = FilterBox((10, 10))
    fb(result)


if __name__ == '__main__':
    # test_pad()
    test_filter_box()
