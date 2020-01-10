import torch
from mmdet.datasets.pipelines.transforms import Pad
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


if __name__ == '__main__':
    test_pad()
