# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/15
"""
import os
import json
import cv2
import numpy as np
import subprocess
import logging

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
          (0, 255, 255), (255, 0, 255), (255, 255, 0)]


def execute_cmd(cmd):
    logging.info(cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while p.poll() is None:
        line = p.stdout.readline().decode("utf-8")
        line = line.strip()
        if line:
            print('{} command output: {}'.format(cmd, line))
    if p.returncode == 0:
        logging.info("success execute " + cmd)
        return True
    else:
        logging.ERROR("fail execute " + cmd)
        return False


def show(img, win_name):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


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
                "name": item["name"],
                'height': item["image_height"],
                'width': item["image_width"],
                'category': [item["category"]],
                'bbox': [item["bbox"]],
            }
        else:
            images_data[item["name"]]["category"].append(item["category"])
            images_data[item["name"]]["bbox"].append(item["bbox"])
    return images_data


def check_or_make_dir(root, dir_name, mkdir=False):
    dir_path = os.path.join(root, dir_name)
    if not os.path.exists(dir_path):
        if mkdir:
            os.mkdir(dir_path)
        else:
            raise ValueError("Error path: " + dir_path)
    return dir_path


def save_json(save_path, save_data):
    if not save_path.endswith(".json"):
        raise ValueError(save_path)
    with open(save_path, 'w') as fw:
        json.dump(save_data, fw)


def xywh_to_xyxy(box):
    box = np.array(box, dtype=np.float)
    xy = box[:2]
    wh = box[2:]
    box[:2] = xy - wh//2
    box[2:] = xy + wh//2
    return list(box)


def divide_according_sliding(image, size, stride, padding=False):
    """
    通过滑动窗口切分图片
    :param image: 要切分的图片
    :param size: 切分后图片的大小
    :param stride: 滑动步长
    :param padding: 边缘填充
    :return: 切分后的图片列表
    """
    ori_size = image.shape[:2]
    size = np.array(size)
    stride = np.array(stride)
    # 计算分割后的图像数量
    num = (ori_size - size) // stride + 1
    if padding:
        num += 1
        pad = (num - 1) * stride + size - ori_size
        image = np.pad(image, ((0, pad[0]), (0, pad[1]), (0, 0)), 'constant', constant_values=0)
    else:
        crop = ori_size - (num - 1) * stride - size
        image = image[:-crop[0], :-crop[1], :]
    division_images = []
    original_position = []
    for i in range(num[0]):
        for j in range(num[1]):
            start = np.array([i, j]) * stride
            end = start + size
            crop_img = image[start[0]:end[0], start[1]:end[1], :]
            division_images.append(crop_img)
            original_position.append(list(start[::-1]))
    return division_images, original_position


def divide_according_point(image, size, boxes):
    """
    通过指定点的方式分割图片
    :param image: 要分割的图像
    :param size: 分割后的图像大小
    :param boxes: 瑕疵box list
    :return: 分割后的图像列表
    """
    ori_size = image.shape[:2]
    size = np.array(size)
    num = len(boxes)
    division_images = []
    for i in range(num):
        box = np.array(boxes[i], dtype=np.int)
        point = np.array(box[:2], dtype=np.int)
        box_size = box[2:] - box[:2]
        high = np.where(size - box_size <= 10, size, size - box_size)
        crop_y = np.random.randint(0, high[0])
        crop_x = np.random.randint(0, high[1])
        crop_lt = np.array([crop_x, crop_y])
        crop_lt = point - crop_lt
        crop_lt = np.where(crop_lt < 0, 0, crop_lt)
        crop_tl = crop_lt[::-1]
        crop_br = crop_tl + size
        crop_br = np.where(crop_br > ori_size, ori_size, crop_br)
        crop_tl = crop_br - size
        crop = np.append(crop_tl, crop_br)
        print(ori_size, crop)
        crop_img = image[crop[0]:crop[2], crop[1]:crop[3], :]
        division_images.append(crop_img)
    return division_images


def make_pair(obj):
    if isinstance(obj, int):
        return obj, obj
    elif isinstance(obj, list) or isinstance(obj, tuple):
        if len(obj) == 1:
            return int(obj[0]), int(obj[0])
        return [int(o) for o in obj]
    elif isinstance(obj, str):
        return int(obj), int(obj)
    else:
        raise ValueError(obj)


def draw_box(img, boxes, names, scores=(1., )):
    for i, box in enumerate(boxes):
        p1 = int(box[0]), int(box[1])
        p2 = int(box[2]), int(box[3])
        if scores[i] == 1.:
            color = (50, 100, 200)
        else:
            color = COLORS[int(names[i])]
        cv2.rectangle(img, p1, p2, color, 2)
        cv2.putText(img, str(names[i]) + " {:.3f}".format(scores[i]), p1, cv2.FONT_HERSHEY_COMPLEX, 2, color, 6)
    # return img


if __name__ == '__main__':
    print("common")
