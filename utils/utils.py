# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/5
"""
import os
import shutil
import random
import cv2
import torch
import numpy as np
from utils.common import check_or_make_dir, get_json_data, json_data_to_images_data


def read_txt_label(label_path):
    data = []
    with open(label_path, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            if line.strip():
                data.append(line.replace("\n", ""))
    return data


def parse_txt(line, size):
    height, width = size
    line = line.split(" ")
    category = int(line[0])
    box = [float(b) for b in line[1:]]  # x, y, w, h
    box[0] = round(box[0] * width)
    box[1] = round(box[1] * height)
    box[2] = round(box[2] * width)
    box[3] = round(box[3] * height)
    return category, box


def xywh_to_xyxy(box):
    box = np.array(box, dtype=np.int)
    xy = box[:2]
    wh = box[2:]
    box[:2] = xy - wh//2
    box[2:] = xy + wh//2
    return list(box)


def show(img, win_name="show"):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


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


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.mat = None

    def update(self, a, b):
        if self.mat is None:
            self.mat = torch.zeros((3, ), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            inds = a.to(torch.int64) + b
            count = torch.bincount(inds)
            for i in range(count.shape[0]):
                # print(i, count[i])
                self.mat[i] += count[i]
            # print(self.mat)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # acc_global = torch.diag(h).sum() / h.sum()
        # acc = torch.diag(h) / h.sum(1)
        iu = h[2] / (h[1] + h[2])
        return iu

    def __str__(self):
        iu = self.compute()
        return 'mean IoU: {:.3f}'.format(
                iu.mean().item() * 100)


def draw_box(img, boxes, names, colors):
    for i, box in enumerate(boxes):
        p1 = int(box[0]), int(box[1])
        p2 = int(box[2]), int(box[3])
        cv2.rectangle(img, p1, p2, colors[names[i]], 2)
        cv2.putText(img, str(names[i]), p1, cv2.FONT_HERSHEY_COMPLEX, 5, colors[names[i]], 12)
    return img


def main():
    root = r"D:\learnspace\dataset\project_001\tile_round1"
    images_dir = "train_imgs"
    json_file = "train_annos.json"
    colors = [(0, 255, 255), (0, 0, 255), (255, 0, 0),
              (0, 255, 0), (255, 255, 0), (255, 0, 255)]

    images_dir_path = check_or_make_dir(root, images_dir)
    json_file_path = check_or_make_dir(root, json_file)
    images_file = os.listdir(images_dir_path)
    json_data = get_json_data(json_file_path)
    images_data = json_data_to_images_data(json_data)

    images_info = {}
    for name in images_file:
        img_name = name.rsplit("_", 2)[0]
        # print(name, img_name, images_data[name])
        if img_name not in images_info:
            images_info[img_name] = [images_data[name]]
        else:
            images_info[img_name].append(images_data[name])
        img = cv2.imread(os.path.join(images_dir_path, name))
        img1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, img1_thres = cv2.threshold(img1_gray, 50, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(img1_thres, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        # blur = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯滤波处理原图像降噪
        # canny = cv2.Canny(blur, 100, 200)  # 50是最小阈值,150是最大阈值
        # cv2.imshow('canny', canny)
        show(img)

    # for k, v in images_info.items():
    #     if len(v) > 1:
    #         for item in v:
    #             print(item)
    #             name = item["name"]
    #             boxes = item["bbox"]
    #             categories = item["category"]
    #             img = cv2.imread(os.path.join(images_dir_path, name))
    #             img = draw_box(img, boxes, categories, colors)
    #             show(img, name)


if __name__ == '__main__':
    print("utils")
    main()
