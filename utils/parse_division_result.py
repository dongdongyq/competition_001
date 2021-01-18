#!/user/bin/python
# -*- coding: utf-8 -*-
"""
@File: parse_division_result.py
@Description: 描述
@Author: dongyanqiang
@Email: 181228331@163.com
@Date: 2021/1/16
"""
import os
import cv2
import numpy as np
from common import get_json_data, check_or_make_dir, draw_box, show, json_data_to_images_data, save_json


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    # Intersection area
    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * max(inter_rect_y2 - inter_rect_y1 + 1, 0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def filter_bbox(bboxes, scores, iou_thr=0.1):
    removes = []
    for i in range(len(bboxes)-1):
        for j in range(i+1, len(bboxes)):
            iou = bbox_iou(bboxes[i], bboxes[j])
            if iou > iou_thr:
                if scores[i] > scores[j]:
                    removes.append(j)
                else:
                    removes.append(i)
    return removes


def filter_score(division_datas, score_thr=0.05):
    bboxes = []
    categories = []
    scores = []
    for j, image_data in enumerate(division_datas):
        pos = image_data["position"]
        box = image_data["bbox"]
        category = image_data["category"]
        score = image_data["score"]
        if score < score_thr:
            continue
        bb = np.array(box)
        pp = np.array(pos)
        bb[:2] += pp + 1
        bb[2:] += pp - 3
        bboxes.append(bb)
        categories.append(category)
        scores.append(score)
    return bboxes, categories, scores


def generate_submit_result(image_name, bboxes, categories, scores, removes):
    result = []
    for i, bbox in enumerate(bboxes):
        if i in removes:
            continue
        res = {
            "name": image_name,
            "category": int(categories[i]),
            "bbox": [round(b, 2) for b in bbox],
            "score": round(scores[i], 6),
        }
        result.append(res)
    return result


# division_img = img[pos[1]:pos[1]+640, pos[0]:pos[0]+640, :]
# draw_box(img, [bb], [category], [score])
# show(img, "1")
# cv2.destroyWindow("1")

def main():
    result_dir = "../result"
    # ori_dataset_dir = r"D:\learnspace\dataset\project_001\tile_round1"
    # labels_json = check_or_make_dir(ori_dataset_dir, "train_annos.json")
    # labels_data = get_json_data(labels_json)
    # images_data = json_data_to_images_data(labels_data)
    save_path = os.path.join("..", "result.json")
    result_jsons = os.listdir(result_dir)
    submit_result = []
    for i, division_json in enumerate(result_jsons):
        division_json_path = check_or_make_dir(result_dir, division_json)
        division_datas = get_json_data(division_json_path)
        image_name = division_json.replace(".json", ".jpg")
        # ori_image_data = images_data[image_name]
        # ori_box = ori_image_data["bbox"]
        # ori_category = ori_image_data["category"]
        # ori_score = [1 for _ in range(len(ori_category))]
        # ori_image_path = os.path.join(ori_dataset_dir, "train_imgs", image_name)
        # img = cv2.imread(ori_image_path)
        # draw_box(img, ori_box, ori_category, ori_score)
        bboxes, categories, scores = filter_score(division_datas, score_thr=0.05)
        removes = filter_bbox(bboxes, scores)
        result = generate_submit_result(image_name, bboxes, categories, scores, removes)
        submit_result.extend(result)
    save_json(save_path, submit_result)


if __name__ == "__main__":
    print("vision/parse_division_result.py")
    main()


