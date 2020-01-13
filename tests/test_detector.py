from mmdet.models import Darknet
import cv2
import numpy
from pprint import pprint
import torch
import numpy as np


def get_test_input():
    img = cv2.imread('data/dog-cycle-car.png')
    img = cv2.resize(img, (416, 416))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    return img_


if __name__ == '__main__':
    model = Darknet('configs/yolo/yolov3.cfg')
    inp = get_test_input()
    pred = model(inp)
    print(pred, pred.shape)
    model = model.cuda()
    inp = inp.cuda()
    pred = model(inp)
    print(pred, pred.shape)

    model.load_weights('checkpoints/yolov3.weights')
    pred = model(inp)
    results = model.write_results(pred, 0.3, 80, )
    print(results)
