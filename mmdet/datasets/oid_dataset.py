from .registry import DATASETS
from .custom import CustomDataset
import pandas as pd
import numpy as np
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time


@DATASETS.register_module
class OIDTDataset(CustomDataset):
    """
    self.img_infos = [{id, filename, width, height}]
    self.img_ids = [id]
    该数据集给予[OID_ToolKit](https://github.com/EscVM/OIDv4_ToolKit)项目
    """

    CLASSES = ('Human head',)

    def __init__(self, **kwargs):
        assert kwargs.get('class_description', None) is not None
        self.class_description = kwargs.pop('class_description')
        super(OIDTDataset, self).__init__(**kwargs)

    def split_label_bbox(self, line):
        words = line.split(' ')
        index = 0
        while True:
            try:
                t = float(words[index])
                break
            except ValueError:
                index += 1
        label = ' '.join(words[0: index])
        bbox = words[index: index+4]
        return label, bbox

    def load_annotations(self, ann_file):
        """
        读取self.img_prefix文件夹下的Label中的文件，以该文件夹为基础读取图片
        返回值给self.img_infos， self.img_info需要{filename, width, height}
        :param ann_file:
        :return:
        """
        # ann = pd.read_csv(ann_file)
        code_class_df = pd.read_csv(self.class_description, header=None, names=['code', 'class'])
        # print(code_class_df.head())

        code_class = dict(zip(code_class_df['code'], code_class_df['class']))
        class_code = dict(zip(code_class_df['class'], code_class_df['code']))

        obj_class = self.CLASSES[0]
        obj_code = class_code[obj_class]
        obj_path = Path(self.img_prefix, obj_class)
        label_path = os.path.join(self.img_prefix, obj_class, 'Label')
        # 以label中有的文件为基础的数据集
        label_files = list(obj_path.rglob('Label/*.txt'))
        # 数据集中的imgids
        img_ids = [x.name.rstrip('.txt') for x in label_files]
        # 若在Label文件中有的标识，则应该存在对应的图片
        filenames = [os.path.join(obj_class, '{}.jpg'.format(x)) for x in img_ids]
        # 确定width， height
        widths, heights = [], []
        for file in filenames:
            file_path = os.path.join(self.img_prefix, file)
            img = Image.open(file_path)
            widths.append(img.width)
            heights.append(img.height)

            img.close()

        # label格式：[class_name xmin ymin xmax ymax\n...]
        # Orange 0.0 0.0 793.7986559999999 765.0
        # 一次性读取ann，若内存不够可讲下面的模块搬到`get_ann_info`方法中
        bboxes = []
        labels = []
        for file in label_files:
            label = []
            bbox = []
            with open(str(file), 'r') as f:
                for line in f.readlines():
                    # s = line.split(' ')
                    _, b = self.split_label_bbox(line)
                    bbox.append(b)
                    label.append(1)  # 1为Orange的标号, 标签类从1开始，0为背景类
            bboxes.append(np.asarray(bbox).astype(np.float32))
            labels.append(np.asarray(label).astype(np.int64))
        # bboxes = np.asarray(bboxes, dtype=np.float32)
        # labels = np.asarray(labels, dtype=np.int64)

        img_infos = [{
            'filename': filename,
            'width': width,
            'height': height,
            'ann': {
                'bboxes': bbox,
                'labels': label,
                'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array([], dtype=np.int64)
            }
        } for filename, width, height, bbox, label
            in zip(filenames, widths, heights, bboxes, labels)]

        return img_infos

    def get_ann_info(self, idx):
        """
        获得对应idx的ann
        返回{
            'bboxes': <np.ndarray, float32> (n, 4),
            'labels': <np.ndarray, int64> (n, ),
            'bboxes_ignore': <np.ndarray, float32> (k, 4),
            'labels_ignore': <np.ndarray, int64> (k, ) (optional field)
        }
        :param idx:
        :return:
        """
        return self.img_infos[idx]['ann']


@DATASETS.register_module
class OIDTDatasetV2(CustomDataset):
    """
    self.img_infos = [{id, filename, width, height}]
    self.img_ids = [id]
    该数据集给予[OID_ToolKit](https://github.com/EscVM/OIDv4_ToolKit)项目
    直接读取图片,标签为配置中指定的类,bbox为(left, top, right, bottom)
    """

    CLASSES = ('Human head',)

    def __init__(self, **kwargs):
        assert kwargs.get('class_description', None) is not None
        self.class_description = kwargs.pop('class_description')
        classes = kwargs.pop('classes')
        # 要使用全部数据还是部分数据[0, 1]
        self.persentage = kwargs.pop('persentage') if kwargs.get('persentage') is not None else 1
        # 更新classes
        self.CLASSES = classes
        # classes = getattr(kwargs, 'classes')
        # delattr(kwargs, 'classes')
        super(OIDTDatasetV2, self).__init__(**kwargs)

        pass

    def load_annotations(self, ann_file):
        """
        读取self.img_prefix文件夹下的Label中的文件，以该文件夹为基础读取图片
        返回值给self.img_infos， self.img_info需要{filename, width, height}
        :param ann_file:
        :return:
        """
        # ann = pd.read_csv(ann_file)
        code_class_df = pd.read_csv(self.class_description, header=None, names=['code', 'class'])
        # print(code_class_df.head())

        code_class = dict(zip(code_class_df['code'], code_class_df['class']))
        class_code = dict(zip(code_class_df['class'], code_class_df['code']))

        ann_csv = pd.read_csv(self.ann_file)
        print(ann_csv.head())

        def add_class(df, code_class):
            df['class'] = [code_class[x] for x in df['LabelName'].to_list()]

        add_class(ann_csv, code_class)

        print(ann_csv.head())

        obj_classes = self.CLASSES

        obj_imgs = ann_csv[ann_csv['class'].isin(obj_classes)]

        filenames = [x for x in os.listdir(self.img_prefix) if x.endswith('jpg')]
        filenames = filenames[0:int(len(filenames) * self.persentage)]
        img_ids = [x.split('.')[0] for x in filenames]

        print('folder {} path have {} number imgs'.format(self.img_prefix, len(obj_imgs)))

        # 数据集中的imgids
        # img_ids = obj_imgs['ImageID']
        # img_ids =
        # 若在Label文件中有的标识，则应该存在对应的图片
        # filenames = ['{}.jpg'.format(x) for x in img_ids]

        # 确定width， height
        widths, heights = [], []
        for file in tqdm(filenames):
            file_path = os.path.join(self.img_prefix, file)
            img = Image.open(file_path)
            widths.append(img.width)
            heights.append(img.height)

            img.close()

        # label格式：[class_name xmin ymin xmax ymax\n...]
        # Orange 0.0 0.0 793.7986559999999 765.0
        # 一次性读取ann，若内存不够可讲下面的模块搬到`get_ann_info`方法中
        bboxes = []
        labels = []

        classes_id = dict([(x, i+1) for i, x in enumerate(self.CLASSES)])

        groups = obj_imgs.groupby('ImageID')

        print(classes_id)

        # sub_obj_imgs

        for i, img_id in enumerate(tqdm(img_ids)):
            label = []
            bbox = []
            width = widths[i]
            height = heights[i]
            # start = time.time()
            # sub_anns = obj_imgs[obj_imgs['ImageID'] == img_id]
            sub_anns = groups.get_group(img_id)
            # sub_anns = obj_imgs[(obj_imgs['ImageID'] == img_id) & (obj_imgs['class'].isin(self.CLASSES))]
            # print(time.time() - start)

            # bbox = sub_anns[['XMin', 'XMax', 'YMin', 'YMax']].values.tolist()
            # label = [classes_id[x] for x in sub_anns['class'].values.tolist()]

            for sub_ann in sub_anns.iterrows():
                content = sub_ann[1]
                c = content['class']
                if c not in self.CLASSES:
                    continue
                left = width * float(content['XMin'])
                top = height * float(content['YMin'])
                right = width * float(content['XMax'])
                bottom = height * float(content['YMax'])

                bbox.append([left, top, right, bottom])
                label.append(classes_id[content['class']])

            bboxes.append(np.asarray(bbox).astype(np.float32))
            labels.append(np.asarray(label).astype(np.int64))

        img_infos = [{
            'filename': filename,
            'width': width,
            'height': height,
            'ann': {
                'bboxes': bbox,
                'labels': label,
                'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array([], dtype=np.int64)
            }
        } for filename, width, height, bbox, label
            in zip(filenames, widths, heights, bboxes, labels)]

        return img_infos

    def get_ann_info(self, idx):
        """
        获得对应idx的ann
        返回{
            'bboxes': <np.ndarray, float32> (n, 4),
            'labels': <np.ndarray, int64> (n, ),
            'bboxes_ignore': <np.ndarray, float32> (k, 4),
            'labels_ignore': <np.ndarray, int64> (k, ) (optional field)
        }
        :param idx:
        :return:
        """
        return self.img_infos[idx]['ann']

