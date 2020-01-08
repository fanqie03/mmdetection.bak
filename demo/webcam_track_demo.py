import argparse
import time

import cv2
import torch
from threading import Thread
import os

from mmdet.apis import inference_detector, init_detector, show_result
from sort import *

import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--camera-id', default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=416, half=False):
        self.mode = 'images'
        self.img_size = img_size
        self.half = half  # half precision fp16 images

        # if os.path.isfile(sources):
        #     with open(sources, 'r') as f:
        #         sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        # else:
        #     sources = [sources]
        sources = [sources]
        self.count = 0
        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(0 if s == '0' else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 2:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()

        return img0

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def main():
    args = parse_args()

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    # camera = cv2.VideoCapture(args.camera_id)
    camera = LoadStreams(args.camera_id)

    # tracker
    tracker = Sort(10, 4)

    frame_rate = 0
    start_time = time.time()
    frame_count = 0

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        # ret_val, img = camera.read()
        img = camera.__next__()[0]
        result = inference_detector(model, img)

        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        bboxes = np.vstack(bbox_result)

        bboxes = bboxes[bboxes[:, 4] > args.score_thr]

        trackers = tracker.update(bboxes)
        trackers = trackers.astype(np.int32)

        for d in trackers:
            cv2.rectangle(img, (d[0], d[1]), (d[2], d[3]), (255, 255, 255))
            cv2.putText(img, '{}'.format(d[4]), (d[0], d[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        end_time = time.time()
        if (end_time - start_time) >= 1:
            frame_rate = int(frame_count / (end_time - start_time))
            start_time = time.time()
            frame_count = 0

        cv2.putText(img, str(frame_rate) + " fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    thickness=2, lineType=2)

        frame_count += 1

        show_result(
            img, result, model.CLASSES, score_thr=args.score_thr, wait_time=1)


if __name__ == '__main__':
    main()
