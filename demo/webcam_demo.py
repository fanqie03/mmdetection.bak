import argparse
import time
import os

import cv2
import torch

from mmdet.apis import inference_detector, init_detector, show_result
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--camera-id', default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default=None, help='detection result output directory'
    )
    parser.add_argument(
        '--out-video', type=str, default=None, help='detection result video output file'
    )
    parser.add_argument(
        '--out-file', type=str, default=None
    )
    parser.add_argument(
        '--show', type=int, default=1, help='show detection result or not'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    camera = cv2.VideoCapture(args.camera_id)

    if args.out_video:
        frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width, frame_height))
        args.show=False
        args.out_dir=None

    frame_rate = 0
    start_time = time.time()
    frame_count = 0

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        if ret_val == False or img is None:
            break

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

        if args.out_dir:
            args.out_file = os.path.join(args.out_dir, '{}.jpg'.format(time.time()))

        img = show_result(
            img, result, model.CLASSES, score_thr=args.score_thr, wait_time=1, show=args.show,
            out_file=args.out_file)

        if args.out_video:
            out.write(img)

    if args.out_video:
        out.release()


if __name__ == '__main__':
    main()
