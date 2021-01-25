#!/user/bin/python
# -*- coding: utf-8 -*-
"""
@File: coco_to_voc.py
@Description: 描述
@Author: dongyanqiang
@Email: 181228331@163.com
@Date: 2021/1/18
"""
import os
import argparse
from common import check_or_make_dir, get_json_data, json_data_to_images_data
from division_image import VocData, CocoData


def deal_one_image(img_name, image_data, dataset):
    size = [image_data["height"], image_data["width"]]
    dataset.write(img_name, size, image_data["bbox"], image_data["category"])
    # print(save_name, index, item)


def main(args):
    save_path = args.data_dir
    if args.data_name == "coco":
        dataset = CocoData(save_path)
    else:
        dataset = VocData(save_path)
    json_file_path = os.path.join(args.data_dir, "train_annos.json")
    json_data = get_json_data(json_file_path)
    images_data = json_data_to_images_data(json_data)
    for i, (img_name, item) in enumerate(images_data.items()):
        print("{}/{}: ".format(i, len(images_data.keys())), img_name)
        deal_one_image(img_name, item, dataset)
    dataset.save()


def parse_args():
    parser = argparse.ArgumentParser(description='division dataset')
    parser.add_argument('--data_dir', default=r'D:\python\competition\dataset\tile_round1_train_20201231', help='dataset path')
    parser.add_argument('--data_name', default='coco', help='dataset name')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("vision/coco_to_voc.py")
    main(parse_args())