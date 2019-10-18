import os.path as osp
import xml.etree.ElementTree as ET
import os

import mmcv
import numpy as np
from tqdm import tqdm

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class FireDataset(CustomDataset):

    CLASSES = ('fire', )

    def __init__(self, min_size=None, **kwargs):
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size
        super(FireDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        """
        ann_file
        ├── img (10).xml
        ├── img (11).xml
        ├── img (12).xml
        ├── img (13).xml
        ├── img (14).xml
        ├── ...
        :param ann_file: it is a folder which has some .xml files
        :return:
        """

        xml_files = os.listdir(ann_file)
        # img_files = os.listdir(self.img_prefix)
        img_files = [x.split('.')[0] + '.jpg' for x in xml_files]

        img_infos = []

        for i, xml_file in enumerate(tqdm(xml_files)):
            file = osp.join(self.ann_file, xml_file)
            tree = ET.parse(file)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            bboxes = []
            labels = []
            bboxes_ignore = []
            labels_ignore = []

            for obj in root.findall('object'):

                name = obj.find('name').text
                label = self.cat2label[name]
                difficult = int(obj.find('difficult').text)
                bnd_box = obj.find('bndbox')
                bbox = [
                    int(bnd_box.find('xmin').text),
                    int(bnd_box.find('ymin').text),
                    int(bnd_box.find('xmax').text),
                    int(bnd_box.find('ymax').text)
                ]
                ignore = False
                if self.min_size:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.min_size or h < self.min_size:
                        ignore = True
                if difficult or ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)

            if not bboxes:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0,))
            else:
                bboxes = np.array(bboxes, ndmin=2) - 1
                labels = np.array(labels)
            if not bboxes_ignore:
                bboxes_ignore = np.zeros((0, 4))
                labels_ignore = np.zeros((0,))
            else:
                bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
                labels_ignore = np.array(labels_ignore)

            ann = dict(
                bboxes=bboxes.astype(np.float32),
                labels=labels.astype(np.int64),
                bboxes_ignore=bboxes_ignore.astype(np.float32),
                labels_ignore=labels_ignore.astype(np.int64))

            img_infos.append(
                dict(filename=img_files[i], width=width, height=height, ann=ann))

        return img_infos

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']
