from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import argparse
import time
import os.path as osp
import glob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('ckpt')
    parser.add_argument('--input', nargs='+')
    parser.add_argument('--output')
    parser.add_argument('--score_thr', type=float, default=0.3)
    return parser.parse_args()


def main():
    args = get_args()
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.ckpt, device='cuda:0')

    if len(args.input) == 1:
        args.input = glob.glob(args.input[0])
    for img_path in args.input:
        # test a single image and show the results
        # img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
        img = mmcv.imread(img_path)
        result = inference_detector(model, img)
        # visualize the results in a new window

        out_file = None if args.output is None else osp.join(args.output, str(time.time()) + '.jpg')

        show_result(img, result, model.CLASSES, score_thr=args.score_thr, out_file=out_file)


if __name__ == '__main__':
    main()
