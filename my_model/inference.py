# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/7
"""
import os
import argparse
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from my_model.dataset import get_dataloader
from torchvision import transforms
from torchvision.transforms import functional as F
from dataset import get_transform
from utils.common import show, save_json, check_or_make_dir


def save_output(pred, target=None, index=0):
    # pred = torch.sigmoid(pred)

    # cv2.waitKey(10)
    zero = np.zeros(pred.shape[-2:], dtype=np.uint8)
    output = pred.detach().numpy()
    output = output[0, 0, :, :] * 255
    output = np.where(output > 200, 255, 0)
    output = np.array(output, dtype=np.uint8)

    if target is not None:
        target = target.detach().numpy()
        target = target[0, 0, :, :] * 255
        target = np.array(target, dtype=np.uint8)

    save_img = np.stack((zero, output, zero), axis=-1)
    print(save_img.shape)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 300, 300)
    cv2.imshow("output", save_img)
    # cv2.imwrite("./outputs/index{:05}.jpg".format(index), save_img)


def detect_one_img(img, model, label=None, i=0):

    pred = model(img)

    save_output(pred, label, i)
    return pred


def array_to_tensor(img):
    print(img.shape)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(2048),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img)
    return torch.unsqueeze(tensor, dim=0)


def unnormalize(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    um = [-m / s for m, s in zip(mean, std)]
    us = [1 / s for _, s in zip(mean, std)]
    transform = transforms.Compose([
        transforms.Normalize(mean=um,
                             std=us)
    ])
    tensor = transform(img)
    return tensor


def get_min_box(output, ori_size, thr=230):
    output = output.detach().numpy()
    output = output[0, 0, :, :] * 255
    output = np.where(output > thr, 255, 0)
    output = np.array(output, dtype=np.uint8)

    output = cv2.resize(output, ori_size)
    contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for c in contours:
        area = cv2.contourArea(c)
        areas.append(area)
    index = areas.index(max(areas))
    box = cv2.boundingRect(contours[index])
    min_box = cv2.minAreaRect(contours[index])  # 中心点x, y, w, h, angle
    min_box = cv2.boxPoints(min_box)
    min_box = np.array(min_box, dtype=np.int32)
    return min_box, box


def main(opt):
    root = r"D:\python\competition\dataset\tile_round1_testA_20201231"
    ckpt = torch.load(opt.weights)
    model = ckpt["model"]
    trans = get_transform(False)
    source = opt.source
    source = os.path.join(root, source)
    save_path = check_or_make_dir(root, "outside_contour", mkdir=True)
    files = os.listdir(source)
    for i, file in enumerate(files):
        file_path = os.path.join(source, file)
        print(i, len(files), file_path)
        ori_img = Image.open(file_path)
        ori_img_arr = np.array(ori_img)
        tensor, _ = trans(ori_img, None)
        tensor = torch.unsqueeze(tensor, 0)
        pred = model(tensor)
        min_box, box = get_min_box(pred, ori_img_arr.shape[:2][::-1])
        cv2.drawContours(ori_img_arr, [min_box], 0, (0, 0, 255), 5)
        # show(ori_img_arr, "out")
        seg = [int(p) for p in min_box.reshape(1, -1)[0]]
        data = {
            "name": file,
            "bbox": [box[0], box[1], box[0]+box[2], box[1]+box[3]],
            "segmentation": [seg],
        }
        save_json(os.path.join(save_path, file.replace(".jpg", ".json")), data)


if __name__ == '__main__':
    print("inference")
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../weights/model_10.pt', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--source', type=str, default='testA_imgs', help='hyperparameters path')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    opt = parser.parse_args()
    main(opt)
