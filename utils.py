# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/5
"""
import os
import shutil
import random
import json
import cv2
import numpy as np


def get_json_data(file_path):
    if not os.path.exists(file_path):
        print("Error ", file_path)

    with open(file_path, "r") as fp:
        return json.loads(fp.read())


def json_data_to_images_data(json_data):
    """
    将给定的label转换成以图片名为key的字典
    :param json_data:
    :return:
    """
    images_data = {}
    for item in json_data:
        if item["name"] not in images_data:
            images_data[item["name"]] = {
                'height': item["image_height"],
                'width': item["image_width"],
                'category': [item["category"]],
                'bbox': [item["bbox"]],
            }
        else:
            images_data[item["name"]]["category"].append(item["category"])
            images_data[item["name"]]["bbox"].append(item["bbox"])
    return images_data


def read_txt_label(label_path):
    data = []
    with open(label_path, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            if line.strip():
                data.append(line.replace("\n", ""))
    return data


def draw_box(img, bbox):
    for box in bbox:
        box = np.array(box, dtype=np.int)
        point_top = tuple(box[:2])
        point_bottom = tuple(box[2:])
        cv2.rectangle(img, point_top, point_bottom, (0, 0, 255), 2)
    return img


def show(img, win_name="show"):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def check_or_make_dir(root, dir_name, mkdir=False):
    dir_path = os.path.join(root, dir_name)
    if not os.path.exists(dir_path):
        if mkdir:
            os.mkdir(dir_path)
        else:
            raise ValueError("Error path: " + dir_path)
    return dir_path


def generate_mask_labels(images_dir, labels_dir):
    images_name = os.listdir(images_dir)
    for name in images_name:
        image_path = check_or_make_dir(images_dir, name)
        label_path = check_or_make_dir(labels_dir, name.split(".")[0] + ".txt")
        img = cv2.imread(image_path)
        size = img.shape[:2]
        zero = np.zeros(size, dtype=np.uint8)
        label_data = read_txt_label(label_path)   # ['4 0.575109 0.663859 0.018750 0.012500']
        for data in label_data:
            bbox_data = data.split(" ")[1:]
            center_bbox = np.array(bbox_data, dtype=np.float)
            point_lt = center_bbox[:2] - center_bbox[2:] / 2
            point_rb = center_bbox[:2] + center_bbox[2:] / 2
            point_lt = np.array(point_lt * size[::-1], dtype=np.int)
            point_rb = np.array(point_rb * size[::-1], dtype=np.int)
            # print(name, center_bbox, point_lt, point_rb)
            cv2.rectangle(img, tuple(point_lt), tuple(point_rb), (0, 0, 255), 2)
        show(img)


def generate_mask_dataset(images_dir_path, labels_dir_path, save_mask_path):
    train_images_dir = check_or_make_dir(images_dir_path, "train")
    val_images_dir = check_or_make_dir(images_dir_path, "val")
    train_labels_dir = check_or_make_dir(labels_dir_path, "train")
    val_labels_dir = check_or_make_dir(labels_dir_path, "val")

    generate_mask_labels(train_images_dir, train_labels_dir)
    # p1 = (int(bbox[0]), int(bbox[1]))
    # p2 = (int(bbox[2]), int(bbox[3]))
    # if os.path.exists(label_path):
    #     label = cv2.imread(label_path, 0)
    # else:
    #     label = np.zeros([height, width])
    # cv2.rectangle(label, p1, p2, 255, -1)
    # cv2.imwrite(label_path, label)


def generate_val_dataset(root, p=0.3):
    train_images_path = os.path.join(root, "images", "train")
    val_images_path = os.path.join(root, "images", "val")
    if not os.path.exists(val_images_path):
        os.mkdir(val_images_path)

    train_labels_path = os.path.join(root, "labels", "train")
    val_labels_path = os.path.join(root, "labels", "val")
    if not os.path.exists(val_labels_path):
        os.mkdir(val_labels_path)
    images_files = os.listdir(train_images_path)
    labels_files = os.listdir(train_labels_path)

    assert len(images_files) == len(labels_files)
    print(images_files)
    print(labels_files)

    random.shuffle(images_files)

    train_len = int(len(images_files) * p)
    for i in range(train_len):
        img_name = images_files[i]
        label_name = img_name.replace(".jpg", ".txt")
        ori_img_file = os.path.join(train_images_path, img_name)
        ori_lab_file = os.path.join(train_labels_path, label_name)

        new_img_file = os.path.join(val_images_path, img_name)
        new_lab_file = os.path.join(val_labels_path, label_name)

        shutil.move(ori_img_file, new_img_file)
        shutil.move(ori_lab_file, new_lab_file)


def main():
    root = r"D:\python\competition\dataset\tile_round1"
    images_dir = "images"
    labels_dir = "labels"
    labels_mask_dir = "labels_mask"

    # generate_val_dataset(root)

    images_dir_path = check_or_make_dir(root, images_dir)
    labels_dir_path = check_or_make_dir(root, labels_dir)

    save_mask_path = check_or_make_dir(root, labels_mask_dir, mkdir=True)
    generate_mask_dataset(images_dir_path, labels_dir_path, save_mask_path)


if __name__ == '__main__':
    print("utils")
    main()
    # generate_val_dataset(r"D:\learnspace\dataset\project_001\divide_tile_round1", p=0.3)
    # print(np.array([2, 3, 4, 5]) * np.array([4, 5]))
