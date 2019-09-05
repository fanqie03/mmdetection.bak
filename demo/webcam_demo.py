import argparse
import time

import cv2
import torch

from mmdet.apis import inference_detector, init_detector, show_result


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


def main():
    args = parse_args()

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    camera = cv2.VideoCapture(args.camera_id)

    frame_rate = 0
    start_time = time.time()
    frame_count = 0

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        result = inference_detector(model, img)

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
