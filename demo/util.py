import numpy as np
import cv2
import time
import os
import mmcv
import pycocotools.mask as maskUtils


def save_result(img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                out_dir=None,
                save_classes=None,
                out_file=None):
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
        # cv2.rectangle(
        #     img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        # if len(bbox) > 4:
        #     label_text += '|{:.02f}'.format(bbox[-1])
        # cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
        #             cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        crop_image = img[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
        # t = time.asctime(time.localtime(time.time()))
        t = time.time()
        filename = '{}_{:.02f}.jpg'.format(t, bbox[4])
        out_file = os.path.join(out_dir, label_text, filename)
        out_file = os.path.abspath(out_file)
        out_dir_label = os.path.join(out_dir, label_text)
        if not os.path.exists(out_dir_label):
            os.makedirs(out_dir_label)
        cv2.imwrite(out_file, crop_image)
        print(out_file)

    # if out_file is not None:
    #     imwrite(img, out_file)
