import numpy as np
import cv2
import time
import os
import mmcv
import pycocotools.mask as maskUtils


def expand_bbox(bbox, ratio, width, height):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    expand_w = (w * ratio) / 2
    expand_h = (h * ratio) / 2
    x1 = max(x1 - expand_w, 0)
    y1 = max(y1 - expand_h, 0)
    x2 = min(x2 + expand_w, width - 1)
    y2 = min(y2 + expand_h, height - 1)
    return np.asarray([x1, y1, x2, y2], dtype=np.int32)


def save_result(img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                out_dir=None,
                save_classes=None,
                out_file=None,
                expand_ratio=0.3,
                min_size=30):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    height, width, _ = img.shape
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    for bbox, label in zip(bboxes, labels):
        if save_classes is not None and label not in save_classes:
            continue
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        w = bbox_int[2] - bbox_int[0]
        h = bbox_int[3] - bbox_int[1]
        if w < min_size or h < min_size:
            continue
        e_bbox = expand_bbox(bbox_int[0:4], expand_ratio, width, height)
        # cv2.rectangle(
        #     img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        # if len(bbox) > 4:
        #     label_text += '|{:.02f}'.format(bbox[-1])
        # cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
        #             cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        crop_image = img[e_bbox[1]:e_bbox[3], e_bbox[0]:e_bbox[2]]
        # t = time.asctime(time.localtime(time.time()))
        t = time.time()
        filename = '{}_{:.4f}.jpg'.format(t, bbox[4])
        out_file = os.path.join(out_dir, label_text, filename)
        out_file = os.path.abspath(out_file)
        out_dir_label = os.path.join(out_dir, label_text)
        if not os.path.exists(out_dir_label):
            os.makedirs(out_dir_label)
        cv2.imwrite(out_file, crop_image)
        print(out_file)

    # if out_file is not None:
    #     imwrite(img, out_file)
