from mmdet.apis import init_detector, inference_detector, show_result
import cv2
from time import time
from demo.util import save_result

# 首先下载模型文件https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
checkpoint_file = 'checkpoints/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth'
# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

print(model)

cap = cv2.VideoCapture('data/IPCamera/20190808/S-094506-0900.mp4')

start = time()

interval = 0.2

while True:
    _, img = cap.read()

    tic = time()
    if tic - start < interval:
        continue
    else:
        start = tic

    result = inference_detector(model, img)
    show_result(img, result, model.CLASSES, wait_time=1, score_thr=0.9)
    # save_result(img, result, model.CLASSES, score_thr=0.9, out_dir='/home/cmf/datasets/extract_data/gongdi')

# # 测试一系列图片
# imgs = ['test1.jpg', 'test2.jpg']
# for i, result in enumerate(inference_detector(model, imgs, device='cuda:0')):
#     show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))