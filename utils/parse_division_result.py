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


def filter_bbox(bboxes, scores, removes, iou_thr=0.1):
    for i in range(len(bboxes)-1):
        for j in range(i+1, len(bboxes)):
            iou = bbox_iou(bboxes[i], bboxes[j])
            if iou > iou_thr:
                if scores[i] > scores[j]:
                    if j not in removes:
                        removes.append(j)
                else:
                    if i not in removes:
                        removes.append(i)
    return removes


def filter_score(division_datas, score_thr=0.05):
    bboxes = []
    categories = []
    scores = []
    for j, image_data in enumerate(division_datas):
        pos = image_data.get("position", None)
        box = image_data["bbox"]
        category = image_data["category"]
        score = image_data["score"]
        if category == "5":
            if score < 0.7:
                continue
        elif category == "3":
            if score < 0.3:
                continue
        else:
            if score < score_thr:
                continue
        bb = np.array(box)
        if pos is None:
            pp = 0
        else:
            pp = np.array(pos)
        bb[:2] += pp
        bb[2:] += pp
        bboxes.append(bb)
        categories.append(category)
        scores.append(score)
    return bboxes, categories, scores


def filter_outside(bboxes, categories, scores, segmentation, removes):
    segmentation = np.array(segmentation).reshape(-1, 2)
    for i, box in enumerate(bboxes):
        cx, cy = np.array(box[2:] + box[:2]) // 2
        w, h = np.array(box[2:] - box[:2])
        flag = cv2.pointPolygonTest(segmentation, (cx, cy), True)
        if flag < 0:
            if i not in removes:
                removes.append(i)
            # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 5)
            continue
        elif flag < 80:
            if int(categories[i]) >= 3:
                if i not in removes:
                    removes.append(i)
                # print(10)
                # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 5)
                continue
        else:
            if int(categories[i]) < 3:
                if i not in removes:
                    removes.append(i)
                # print(20, categories[i])
                # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 5)
                continue
        # cv2.drawContours(img, [segmentation], 0, (0, 255, 255), 5)
        # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 5)
    # cv2.imwrite("test.jpg", img)
    # show(img)
    return removes


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


def draw_pred_box(result, img):
    for i, item in enumerate(result):
        box = item["bbox"]
        cate = item["category"]
        score = item["score"]
        print(box[2]-box[0], box[3]-box[1])
        draw_box(img, [box], [cate], [score])
        s_img = img[int(box[1])-100:int(box[3])+50, int(box[0])-50:int(box[2])+200, :]
        show(s_img, "img", wait=0)


def main():
    ignores = ["198_23", "201_70", "201_68"]
    result_dir = "../result_original"
    ori_dataset_dir = r"D:\python\competition\dataset\tile_round1_testA_20201231"
    ori_outside_dir = check_or_make_dir(ori_dataset_dir, "outside_contour")
    ori_images_dir = check_or_make_dir(ori_dataset_dir, "testA_imgs")
    # labels_json = check_or_make_dir(ori_dataset_dir, "train_annos.json")
    # labels_data = get_json_data(labels_json)
    # images_data = json_data_to_images_data(labels_data)
    save_path = os.path.join("..", "result.json")
    result_jsons = os.listdir(result_dir)
    submit_result = []
    for i, division_json in enumerate(result_jsons):
        print(i, len(result_jsons), division_json)
        if division_json.rsplit("_", 2)[0] in ignores:
            continue
        division_json_path = check_or_make_dir(result_dir, division_json)
        division_datas = get_json_data(division_json_path)
        image_name = division_json.replace(".json", ".jpg")
        img = cv2.imread(check_or_make_dir(ori_images_dir, image_name))
        # outside_data = get_json_data(check_or_make_dir(ori_outside_dir, division_json))
        # segmentation = outside_data["segmentation"]
        # image_lab_data = images_data[image_name]
        # draw_box(img, image_lab_data["bbox"], image_lab_data["category"])
        removes = []
        bboxes, categories, scores = filter_score(division_datas, score_thr=0.1)
        # removes = filter_outside(bboxes, categories, scores, segmentation, removes)
        removes = filter_bbox(bboxes, scores, removes)
        result = generate_submit_result(image_name, bboxes, categories, scores, removes)
        draw_pred_box(result, img)
        # cv2.imwrite("1.jpg", img)
        submit_result.extend(result)
    save_json(save_path, submit_result)


if __name__ == "__main__":
    print("vision/parse_division_result.py")
    main()
