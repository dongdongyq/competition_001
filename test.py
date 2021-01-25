#!/user/bin/python
# -*- coding: utf-8 -*-
"""
@File: test.py
@Description: 描述
@Author: dongyanqiang
@Email: 181228331@163.com
@Date: 2021/1/12
"""
import os
from utils.common import check_or_make_dir, get_json_data, json_data_to_images_data


def main():
    ori_dataset_dir = r"D:\python\competition\dataset\tile_round1_train_20201231"
    labels_json = check_or_make_dir(ori_dataset_dir, "train_annos.json")
    labels_data = get_json_data(labels_json)
    images_data = json_data_to_images_data(labels_data)
    for i, division_json in enumerate(images_data.keys()):
        print(i, division_json)
        image_name = division_json.replace(".json", ".jpg")
        image_lab_data = images_data[image_name]
        for box in image_lab_data["bbox"]:
            w = box[2] - box[0]
            h = box[3] - box[1]
            print(w, h, w*h)


if __name__ == "__main__":
    print("vision/test.py")
    main()