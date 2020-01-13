from mmdet.models import Darknet
import cv2
import numpy
from pprint import pprint
import torch
import numpy as np
from mmdet.models.builder import *
from mmdet.datasets.builder import *


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
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def get_test_input():
    img = cv2.imread('data/dog-cycle-car.png')
    img_ = letterbox_image(img, (416, 416))
    img_ = prep_image(img_, 416)
    return img_, img


if __name__ == '__main__':
    from mmcv import Config
    cfg = Config.fromfile('configs/yolo/yolov3_1x.py')
    model = build_detector(cfg.model)
    # dataset = build_dataset(cfg.data['val'])
    # model = Darknet('configs/yolo/yolov3.cfg')
    inp, img = get_test_input()
    pred = model(inp)
    print(pred, pred.shape)
    model = model.cuda()
    inp = inp.cuda()
    pred = model(inp)
    print(pred, pred.shape)

    model.load_weights(cfg.resume_from)
    pred = model(inp)
    results = model.write_results(pred, 0.3, 80, )
    print(results)
    model.draw_result(img, results)
