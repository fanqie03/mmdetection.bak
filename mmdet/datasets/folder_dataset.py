import os.path as osp
import xml.etree.ElementTree as ET
import os

import mmcv
import numpy as np
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset

from .custom import CustomDataset
from .pipelines import Compose
from .registry import DATASETS

img_suffix = ('jpg', 'jpeg', 'png', 'bmp')


@DATASETS.register_module
class FolderDataset(Dataset):
    """
    read image from folder directly
        Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720
        },
        ...
    ]
    """

    def __init__(self,
                 pipeline,
                 data_root):
        self.data_root = data_root

        self.img_infos = self.listimg(self.data_root)
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def listimg(self, root):

        def filter(filename):
            if len(filename.split('.')) < 2:
                return False
            _, suffix = filename.split('.')
            if suffix.lower() in img_suffix:
                return True

        filenames = [osp.join(self.data_root, x) for x in os.listdir(root) if filter(x)]
        img_infos = []
        for filename in filenames:
            img = cv2.imread(filename)
            img_infos.append(dict(
                filename=osp.basename(filename),
                width=img.shape[1],
                height=img.shape[0]
            ))
        return img_infos

    def __len__(self):
        return len(self.img_infos)

    def pre_pipeline(self, results):
        results['img_prefix'] = self.data_root
        # results['ann_info'] = dict(bboxes=[], labels=[])
        # results['bbox_fields'] = []
        # results['mask_fields'] = []
        # results['seg_fields'] = []

    def prepare_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(
            img_info=img_info
        )
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __getitem__(self, idx):
        return self.prepare_img(idx)
