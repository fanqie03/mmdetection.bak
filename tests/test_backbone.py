import torch
from mmdet.models.backbones import Mb_Tiny_RFB


def test_mb_tiny_rfb():
    input_sizes = [(640, 480), (320, 240), (160, 120), (80, 60), (224, 224), (112, 112)]
    for input_size in input_sizes:
        print(input_size)
        data = torch.empty(1, 3, *input_size)
        model = Mb_Tiny_RFB()
        r = model(data)
        for i, f in enumerate(r):
            print(i, f.shape)


if __name__ == '__main__':
    test_mb_tiny_rfb()
