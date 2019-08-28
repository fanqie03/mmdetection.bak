from mmdet.apis import init_detector, inference_detector, show_result
import cv2
from time import time
from demo.util import save_result
import os
from pathlib import Path
import numpy as np

# 首先下载模型文件https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
checkpoint_file = 'checkpoints/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth'
# 初始化模型
model = init_detector(config_file, checkpoint_file)

interval = 0.2

p = Path('/home/cmf/datasets/helmet/raw_video_classifier')

for path in p.rglob('*/*.mp4'):
    try:
        cap = cv2.VideoCapture(str(path))

        start = time()

        while True:
            _, img = cap.read()
            img = np.rot90(img)

            tic = time()
            # if tic - start < interval:
            #     continue
            # else:
            #     start = tic

            result = inference_detector(model, img)
            # show_result(img, result, model.CLASSES, wait_time=1, score_thr=0.9)
            save_result(img, result, model.CLASSES, score_thr=0.9, out_dir='/home/cmf/datasets/helmet/extract_image', save_classes=[0])
    except Exception as e:
        print('raise Exception {}'.format(e))