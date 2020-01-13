from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pprint import pprint
from ..registry import DETECTORS
from ..detectors import SingleStageDetector

import cv2


def draw_func(x, results, color):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(cls)
    # label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def bbox_iou(box1, box2):
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def unique(tensor):
    # tensor_np = tensor.detach().cpu().numpy()
    # unique_np = np.unique(tensor_np)
    # unique_tensor = torch.from_numpy(unique_np)
    #
    # tensor_res = tensor.new(unique_tensor.shape)
    # tensor_res.copy_(unique_tensor)
    # return tensor_res
    return torch.unique(tensor)


def predict_transform(prediction: torch.Tensor,
                      inp_dim: int, anchors: list, num_classes: int):
    CUDA = prediction.is_cuda
    # eg. inp_dim=416, prediction.size(2) = 13, stride=32, grid_size=13
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size ** 2)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size ** 2 * num_anchors, bbox_attrs)

    # 按照公式转换特征图的输出
    # 默认聚类算出来的anchor是以416x416的图像为基础的
    # 此时anchors适应feature map的大小
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid the center_x, center_y, object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    # 将网格偏移量添加到中心坐标预测中。
    # #####
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # 将锚应用与边界框的尺寸
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # 最后一件事是将detection map还原为图像大小
    prediction[:, :, :4] *= stride

    return prediction


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_cfg(cfgfile):
    with open(cfgfile, 'r') as file:
        lines = file.readlines()
        lines = [x.strip() for x in lines]
        lines = [x for x in lines if len(x) > 0]
        lines = [x for x in lines if x[0] != '#']

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks.append(block)  # 最后一行会退出循环，最后一层的block没有加入

    return blocks


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_fileters = filters = 3  # 前一层的通道数，默认通道数为三层
    output_filters = []  # 每一层的输出通道数目，用于创建module

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if (x['type'] == 'convolutional'):
            activation = x['activation']
            # 有batch norm没有bias，有bias没有batch norm
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_fileters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module(f'conv_{index}', conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f'batch_norm_{index}', bn)

            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f'leaky_{index}', activn)

        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsampel = nn.Upsample(scale_factor=stride, mode='bilinear')
            module.add_module(f'upsampel_{index}', upsampel)

        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')

            start = int(x['layers'][0])

            try:
                end = int(x['layers'][1])
            except:
                end = 0

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()  # 占位置,对其module_list和blocks[1:]的位置
            module.add_module(f'route_{index}', route)

            # 通道数目合并
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f'shortcur_{index}', shortcut)

        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module(f"Detection_{index}", detection)
        else:
            raise TypeError(f'unsupport type {x["type"]}')

        module_list.append(module)
        prev_fileters = filters
        output_filters.append(filters)

    return (net_info, module_list)


@DETECTORS.register_module
class Darknet(nn.Module):
    def __init__(self, cfgfile, **kwargs):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.inp_dim = int(self.net_info['height'])

    def forward(self, x):
        CUDA = x.is_cuda

        modules = self.blocks[1:]
        outputs = {}

        write = 0
        for i, module in enumerate(modules):
            module_type = (module['type'])
            if module_type in ['convolutional', 'upsample']:
                x = self.module_list[i](x)
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    # x = outputs[i + layers[0]]
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors

                inp_dim = int(self.net_info['height'])

                num_classes = int(module['classes'])

                # 转换特征图的输出
                x = predict_transform(x, inp_dim, anchors, num_classes)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            else:
                raise TypeError(f'unsoporrtd module type {module_type} in layer {i}')

            outputs[i] = x
        return detections

    def write_results(self, prediction, confidence, num_classes, nms_conf=0.4):
        """
        函数的结果为dx8的张量，每个检测有8个属性，
        即检测所属批次图像的索引、四个location, object score, max class score, max class score index
        :param prediction:
        :param confidence:
        :param num_classes:
        :param nms_conf:
        :return:
        """
        # 过滤分数低的bbox，并保留他们，方便后续向量化操作（每张图过滤后的数目不同）
        conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
        prediction = prediction * conf_mask
        # nms: 对每个类别相似的边界框做过滤
        # 转成对角线的坐标的形式，使用两个对角线的坐标的形式更好计算IOU
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
        box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
        box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
        box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
        prediction[:, :, :4] = box_corner[:, :, :4]

        # 每张图片经过nms出来的结果数目不一致，
        # 不能通过向量操作
        batch_size = prediction.size(0)
        write = False  # 是否初始化output的标志

        for ind in range(batch_size):
            image_pred = prediction[ind]

            # 每个边界框有85个属性，其中80个是类别score。
            # 只关心最高分的class score，
            # 每行删除80个类别分数，添加具有最大值的class score的索引和class score
            max_conf, max_conf_index = torch.max(image_pred[:, 5:5 + num_classes], 1)
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_index = max_conf_index.float().unsqueeze(1)
            seq = (image_pred[:, :5], max_conf, max_conf_index)
            image_pred = torch.cat(seq, 1)

            # 过滤分数低的bbox，可能存在没有obj score大于阈值的bbox
            # debug, torch.nonzero出来的是非零元素的索引
            non_zero_ind = torch.nonzero(image_pred[:, 4])
            try:
                image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
            except:
                continue

            # For PyTorch 0.4 compatibility
            # Since the above code with not raise exception for no detection
            # as scalars are supported in PyTorch 0.4
            if image_pred_.shape[0] == 0:
                continue

            img_classes = unique(image_pred_[:, -1])

            # 按类别执行NMS
            for cls in img_classes:
                # perform NMS
                # 1. 提取特定类的检测值
                cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
                class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
                image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

                conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
                image_pred_class = image_pred_class[conf_sort_index]
                idx = image_pred_class.size(0)

                for i in range(idx):
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                    except ValueError:
                        break

                    except IndexError:
                        break

                    # 将iou > threshold 的bbox置为零， 留下iou < threshold的bbox
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i + 1:] *= iou_mask
                    # 消除iou > nms_conf 的bbox， 留下iou < threshold的bbox
                    non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
                # Repeat the batch_id for as many detections of the class cls in the image
                seq = batch_ind, image_pred_class
                # 函数的结果为dx8的张量，每个检测有8个属性，
                # 即检测所属批次图像的索引、四个location, object score, max class score, max class score index
                if not write:
                    output = torch.cat(seq, 1)
                    write = True
                else:
                    out = torch.cat(seq, 1)
                    output = torch.cat((output, out))

            try:
                return output
            except:
                return 0

    def draw_result(self, img, output):
        im_dim_list = [(img.shape[1], img.shape[0])]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

        if next(self.parameters()).is_cuda:
            im_dim_list = im_dim_list.cuda()

        im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

        scaling_factor = torch.min(self.inp_dim / im_dim_list, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (self.inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

        # 现在，我们的坐标符合填充区域上图像的尺寸。
        # 但是，在letterbox_image函数中，我们已经通过缩放因子调整了图像的两个尺寸（请记住，两个尺寸都用一个公共因子进行划分以保持纵横比）。
        # 现在，我们撤消此重新缩放，以获取原始图像上边界框的坐标。
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

        list(map(lambda x: draw_func(x, [img], (255, 0, 0)), output))
        det_name = "{}/det_{}".format('data/det', 'result.png')
        cv2.imwrite(det_name, img)

    def letterbox_image(self, img):
        '''resize image with unchanged aspect ratio using padding'''
        img_w, img_h = img.shape[1], img.shape[0]
        w, h = self.inp_dim
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))
        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        canvas = np.full((self.inp_dim[1], self.inp_dim[0], 3), 128)

        canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

        return canvas

    def load_weights(self, weightfile):
        # 权重文件的前160个字节存储5个int32值，这些值构成文件的头。
        with open(weightfile, 'rb') as fp:
            # The first 5 values are header information
            # 1. Major version number
            # 2. Minor Version Number
            # 3. Subversion number
            # 4,5. Images seen by the network (during training)
            header = np.fromfile(fp, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            # 剩余权重
            weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0  # ptr以跟踪权重数组中的位置
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]['type']
            if module_type == 'convolutional':
                # nn.Sequential
                model = self.module_list[i]

                batch_normalize = int(self.blocks[i + 1].get('batch_normalize', 0))

                conv = model[0]

                # 加载偏差，要么加载batch norm的bias，要么加载conv的bias
                if batch_normalize:
                    bn = model[1]

                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
