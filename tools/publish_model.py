import argparse
import subprocess

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    parser.add_argument('--classes', help='change classes', nargs='+')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file, classes):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    if classes:
        checkpoint['meta']['CLASSES'] = classes
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    out_file = out_file.rstrip('.pth') + '-tmp.pth'
    torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    final_file = out_file.rstrip('-tmp.pth') + '-{}.pth'.format(sha[:8])
    subprocess.Popen(['mv', out_file, final_file])


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file, args.classes)


if __name__ == '__main__':
    main()
