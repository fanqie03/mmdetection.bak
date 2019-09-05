import argparse
from pathlib import Path
from time import time

import cv2
import numpy as np
from demo.util import save_result
from mmdet.apis import init_detector, inference_detector


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/oid/faster_rcnn_r50_fpn_1x.py')
    parser.add_argument('--checkpoint', default='work_dirs/faster_rcnn_r50_fpn_1x/epoch_12.pth')
    parser.add_argument('--interval', type=float, default=0.2)
    parser.add_argument('--out_dir', default='data/extract_video')
    parser.add_argument('--video_dir', default='data/raw_video')
    parser.add_argument('--score_thr', type=float, default=0.9)
    parser.add_argument('--min_size', type=int, default=30, help='小目标最小size')
    parser.add_argument('--expand_ratio', type=float, default=0.2, help='扩大倍数')
    parser.add_argument('--target_classes', default=[0], help='需要裁剪的类别')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rot90', default=0)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model = init_detector(args.config, args.checkpoint, device=args.device)
    p = Path(args.video_dir)
    cap = None

    for path in p.rglob('*/*.mp4'):
        try:
            print('open', path)
            cap = cv2.VideoCapture(str(path))

            start = time()

            while True:
                _, img = cap.read()
                # img = np.rot90(img, k=0)

                tic = time()
                if tic - start < args.interval:
                    continue
                else:
                    start = tic

                result = inference_detector(model, img)
                # show_result(img, result, model.CLASSES, wait_time=1, score_thr=0.9)
                save_result(img, result, model.CLASSES, score_thr=args.score_thr,
                            out_dir=args.out_dir,
                            save_classes=args.target_classes,
                            min_size=args.min_size,
                            expand_ratio=args.expand_ratio)

        except Exception as e:
            print('raise Exception {}'.format(e))
            cap.release()


if __name__ == '__main__':
    main()
