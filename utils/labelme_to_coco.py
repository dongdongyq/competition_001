# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/18
"""
import os
import cv2
import numpy as np
from common import get_json_data, check_or_make_dir, CocoData, show


def main():
    root_dir = r"D:\learnspace\dataset\project_001\tile_round1"
    ann_dir = check_or_make_dir(root_dir, "Anno")
    ann_files = os.listdir(ann_dir)
    coco = CocoData(root_dir, 1)
    for ann_f in ann_files:
        p = os.path.join(ann_dir, ann_f)
        data = get_json_data(p)
        for k, v in data.items():
            print(k, v)
        image_name = data["imagePath"].split("/")[-1]
        size = [data["imageHeight"], data["imageWidth"]]
        mask = np.zeros(size, dtype=np.uint8)
        contour = np.array(data["shapes"][0]["points"], dtype=np.int32)
        cv2.drawContours(mask, [contour], 0, 100, -1)
        box = cv2.boundingRect(mask)   # 左上角x y w h
        bbox = [box[0], box[1], box[2]+box[0], box[3]+box[1]]
        category = 1
        segmentation = [list(contour.reshape(-1))]
        print(segmentation)
        coco.write(image_name, size, [bbox], [category], segmentation)
    coco.save()


if __name__ == '__main__':
    print("voc_to_seg")
    main()
