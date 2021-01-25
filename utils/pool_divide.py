#!/user/bin/python
# -*- coding: utf-8 -*-
"""
@File: pool_divide.py
@Description: 描述
@Author: dongyanqiang
@Email: 181228331@163.com
@Date: 2021/1/23
"""
import os
import cv2
import argparse
import random
import tqdm
import numpy as np
from multiprocessing import Pool, Manager
from common import get_json_data, json_data_to_images_data,\
    check_or_make_dir, divide_according_sliding, VocData, CocoData, make_pair
from division_image import deal_bbox, filter_bbox_and_images, generate_images


def get_divide_info(images_dir_path, img_name, images_data, save_path, divide_size, stride, q):
    image_path = os.path.join(images_dir_path, img_name)
    image_data = images_data.get(img_name, None)
    if image_data is None:
        return
    if not os.path.exists(image_path):
        return
    img = cv2.imread(image_path)
    divide_images, original_position = divide_according_sliding(img, divide_size, stride, True)
    bboxes = deal_bbox(original_position, image_data["bbox"])
    valid_images_info = filter_bbox_and_images(bboxes, image_data["category"], divide_size[::-1])
    images_info = dict()
    images_info["valid_images_info"] = valid_images_info
    images_info["img_name"] = img_name
    images_info["divide_size"] = divide_size
    for i, (index, item) in enumerate(valid_images_info.items()):
        save_name = img_name[:-4] + "_{:04}".format(i) + img_name[-4:]
        save_img = divide_images[index]
        generate_images(save_path, save_name, save_img)
    # print(valid_images_info)
    q.put(images_info)


def save_dataset(dataset, q, save_path):
    while q.qsize():
        images_info = q.get()
        img_name = images_info["img_name"]
        divide_size = images_info["divide_size"]
        valid_images_info = images_info["valid_images_info"]
        for i, (index, item) in enumerate(valid_images_info.items()):
            save_name = img_name[:-4] + "_{:04}".format(i) + img_name[-4:]
            item["height"] = divide_size[0]
            item["width"] = divide_size[1]
            # generate_txt_labels(save_path, save_name, item)
            dataset.write(save_name, divide_size, item["bbox"], item["category"])
    dataset.save()


def main(args):
    save_path = args.save_path
    divide_size = make_pair(args.divide_size)
    stride = make_pair(args.stride)
    max_num = int(args.max_num)
    data_name = args.data_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = check_or_make_dir(save_path,
                                  "{}_size{}X{}".format(data_name, divide_size[0], divide_size[1]), mkdir=True)
    if data_name == "voc":
        dataset = VocData(save_path, args.save_seg)
    elif data_name == "coco":
        dataset = CocoData(save_path)
    else:
        raise ValueError("not exists {}, please set voc or coco".format(data_name))

    images_dir_path = os.path.join(args.data_dir, "train_imgs")
    json_file_path = os.path.join(args.data_dir, "train_annos.json")

    images_name = os.listdir(images_dir_path)
    json_data = get_json_data(json_file_path)
    images_data = json_data_to_images_data(json_data)
    random.shuffle(images_name)
    if max_num == -1:
        max_num = len(images_name)
    p = Pool(8)
    q = Manager().Queue()
    for i, img_name in enumerate(images_name):
        if i >= max_num:
            continue
        print(i, max_num, img_name)
        p.apply_async(get_divide_info, args=(images_dir_path, img_name, images_data, divide_size, stride, q, ))
    p.close()
    p.join()
    save_dataset(dataset, q, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='division dataset')
    parser.add_argument('--data_dir', default=r'D:\python\competition\dataset\tile_round1_train_20201231', help='dataset path')
    parser.add_argument('--data_name', default='coco', help='dataset name')
    parser.add_argument('--save_seg', action='store_false', help='only data_name is voc. if true, save segmentation')
    parser.add_argument('--save_path', default=r'D:\python\competition\dataset\tile_round1_test', help='save_path')
    parser.add_argument('--divide_size', nargs="+", type=int, default=[128, 128], help='size')
    parser.add_argument('--stride', nargs="+", type=int, default=[120, 120], help='stride')
    parser.add_argument('--max_num', type=int, default=10, help='if -1, save all')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("division_image")
    main(parse_args())
