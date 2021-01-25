#!/user/bin/python
# -*- coding: utf-8 -*-
"""
@File: generate_anchor.py
@Description: 描述
@Author: dongyanqiang
@Email: 181228331@163.com
@Date: 2021/1/24
"""
# -*- coding=utf-8 -*-
import glob
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from kmeans import kmeans, avg_iou
from common import check_or_make_dir, get_json_data, json_data_to_images_data

# 聚类的数目
CLUSTERS = 9
# 模型中图像的输入尺寸，默认是一样的
SIZE = 640


def main():
    ori_dataset_dir = r"D:\python\competition\dataset\tile_round1_train_20201231"
    labels_json = check_or_make_dir(ori_dataset_dir, "train_annos.json")
    labels_data = get_json_data(labels_json)
    images_data = json_data_to_images_data(labels_data)
    dataset = []
    for i, division_json in enumerate(images_data.keys()):
        print(i, division_json)
        image_name = division_json.replace(".json", ".jpg")
        image_lab_data = images_data[image_name]
        for box in image_lab_data["bbox"]:
            w = box[2] - box[0]
            h = box[3] - box[1]
            print(w, h, w*h)
            dataset.append([w, h])
    dataset = np.array(dataset)
    out = kmeans(dataset, k=CLUSTERS)
    print(out)
    print("Accuracy: {:.2f}%".format(avg_iou(dataset, out) * 100))
    print("Boxes:\n {}-{}".format(out[:, 0], out[:, 1]))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))


if __name__ == "__main__":
    print("vision/generate_anchor.py")
    main()