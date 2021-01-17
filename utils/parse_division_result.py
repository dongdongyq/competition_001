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
from common import get_json_data, check_or_make_dir, draw_box, show, json_data_to_images_data


def main():
    result_dir = "../result"
    ori_dataset_dir = r"D:\python\competition\dataset\tile_round1_train_20201231"
    labels_json = check_or_make_dir(ori_dataset_dir, "train_annos.json")
    labels_data = get_json_data(labels_json)
    images_data = json_data_to_images_data(labels_data)
    result_jsons = os.listdir(result_dir)
    for i, division_json in enumerate(result_jsons):
        division_json_path = check_or_make_dir(result_dir, division_json)
        division_datas = get_json_data(division_json_path)
        image_name = division_json.replace(".json", ".jpg")
        ori_image_data = images_data[image_name]
        ori_box = ori_image_data["bbox"]
        ori_category = ori_image_data["category"]
        ori_score = [1 for _ in range(len(ori_category))]
        print(ori_image_data)
        ori_image_path = os.path.join(ori_dataset_dir, "train_imgs", image_name)
        img = cv2.imread(ori_image_path)
        draw_box(img, ori_box, ori_category, ori_score)
        for j, image_data in enumerate(division_datas):
            pos = image_data["position"]
            box = image_data["bbox"]
            category = image_data["category"]
            score = image_data["score"]
            if score < 0.6:
                continue
            bb = np.array(box)
            pp = np.array(pos)
            bb[:2] += pp + 1
            bb[2:] += pp - 3
            print(i, j, category, score, box[0]+pos[0], box[1]+pos[1], box[2]+pos[0], box[3]+pos[1])
            for b in ori_box:
                b = np.array(b)
                print(bb - b)
            print()
            division_img = img[pos[1]:pos[1]+640, pos[0]:pos[0]+640, :]
            draw_box(division_img, [box], [category], [score])
            # show(division_img, "1")
            # cv2.destroyWindow("1")
        # break


if __name__ == "__main__":
    print("vision/parse_division_result.py")
    main()
