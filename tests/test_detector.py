from mmdet.models import Darknet
import cv2
import numpy
from pprint import pprint
import torch
import numpy as np


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def get_test_input():
    img = cv2.imread('data/dog-cycle-car.png')
    img_ = letterbox_image(img, (416, 416))
    img_ = cv2.resize(img_, (416, 416))
    img_ = img_[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    return img_, img


if __name__ == '__main__':
    model = Darknet('configs/yolo/yolov3.cfg')
    inp, img = get_test_input()
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
    model.draw_result(img, results)
