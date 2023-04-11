import os
import sys
import cv2
import numpy as np
import json
import paddle
from PIL import Image
from evaluation.coco_utils import cocoapi_eval
from paddleseg.utils import metrics, logger


# classification
def eval_cls(gt_path, classification):
    """evalueate classification
    """
    with open(gt_path, 'r') as f:
        gts = f.readlines()

    gt_dict = dict()
    for gt in gts:
        line = gt.strip().split()
        img_path, cls_id = line[0], line[1]
        gt_dict[img_path] = cls_id

    correct = 0
    total_cls = 0
    for cls in classification:
        for key in cls.keys():
            pred_labe = str(cls[key])
            gt_label = gt_dict[key]
            if gt_label == pred_labe:
                correct += 1
        total_cls += 1
    acc = correct * 1.0 / total_cls
    print("cls Acc@1: %.4f" % (correct / total_cls))
    return acc

# detection
def eval_dec(gt_det, pred_det):
    """evalueate detection
    """
    bbox_stats = cocoapi_eval(pred_det, 'bbox', anno_file=gt_det)
    eval_results = {}
    eval_results['bbox'] = bbox_stats
    mAP = eval_results['bbox'][1]
    return mAP


# segmentation
def polygon2mask(polygons):
    """polygon2mask for eval
    """
    save_path = 'polygon2mask/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for img_info in polygons:
        img_name = None
        for img_name in img_info.keys():
            cls_polygons = img_info[img_name]
            img_name = img_name

        mask = np.zeros((720, 1280), dtype=np.uint8)
        for category in cls_polygons.keys():
            polys = []
            if category == 0:
                continue

            cls_polys = cls_polygons[category]

            for poly in cls_polys:
                if len(poly)<=2:
                    continue
                polys.append(np.array(poly, dtype=np.int32))
            cv2.fillPoly(mask, polys, int(category))
        cv2.imwrite(os.path.join(save_path, img_name), mask)


def eval_seg(gt_file, pred_file):
    """evaluate segemntation
    """
    intersect_area_all = paddle.zeros([1], dtype='int64')
    pred_area_all = paddle.zeros([1], dtype='int64')
    label_area_all = paddle.zeros([1], dtype='int64')

    pred_files = os.listdir(pred_file)
    gt_files = os.listdir(gt_file)
    
    pred_files.sort()
    gt_files.sort()
    for i, name in enumerate(gt_files):
        img_path = os.path.join(gt_file, name)
        label = paddle.to_tensor(np.asarray(Image.open(img_path)).astype('int64'))
        label = paddle.unsqueeze(label, axis=[0])

        data_path = os.path.join(pred_file, pred_files[i])
        
        pred = paddle.to_tensor(np.asarray(Image.open(data_path)).astype('int32'))
        pred = paddle.unsqueeze(pred, axis=[0, 1])
        intersect_area, pred_area, label_area = metrics.calculate_area(
                        pred,
                        label,
                        19,
                        ignore_index=255)

        intersect_area_all = intersect_area_all + intersect_area
        pred_area_all = pred_area_all + pred_area
        label_area_all = label_area_all + label_area

    metrics_input = (intersect_area_all, pred_area_all, label_area_all)
    class_iou, miou = metrics.mean_iou(*metrics_input)
    acc, class_precision, class_recall = metrics.class_measurement(
        *metrics_input)

    print_detail = True
    if print_detail:
        logger.info("[EVAL] mIoU: \n" + str(np.round(miou, 4)))
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 3)))
        logger.info("[EVAL] Class Precision: \n" + str(
            np.round(class_precision, 3)))
        logger.info("[EVAL] Class Recall: \n" + str(np.round(class_recall, 3)))
    return miou


def main():

    pred_file = sys.argv[1]  # pred_cls.txt'
    cls_gt_file = sys.argv[2]
    dec_gt_file = sys.argv[3]
    seg_gt_file = sys.argv[4]
    with open(pred_file, 'r') as f:
        obj_json = json.load(f)

    classification = obj_json['cls']
    detection = obj_json['dec']
    segmentation = obj_json['seg']

    acc = eval_cls(cls_gt_file, classification)
    mAP = eval_dec(dec_gt_file, detection)

    polygon2mask(segmentation)
    mIoU = eval_seg(seg_gt_file, 'polygon2mask/')
    
    average_score = round(sum([acc, mAP, mIoU]) / 3.0, 4)
    
    return [acc, mAP, mIoU, average_score]


if __name__ == "__main__" :
    acc, mAP, mIoU, average_score = main()
    print(acc, mAP, mIoU, average_score)
