#!/user/bin/python
# -*- coding: utf-8 -*-
"""
@File: inference.py
@Description: 描述
@Author: dongyanqiang
@Email: 181228331@163.com
@Date: 2021/1/16
"""
import os
import argparse
import paddlex as pdx
import tqdm
import json
import cv2
from paddle_model.dataset import eval_transforms
from utils.common import divide_according_sliding, make_pair

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(args):
    model_dir = args.model_save_dir
    save_dir = args.save_dir
    valid_suffix = [
            'JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png'
        ]
    image_path = args.data_dir
    image_list = os.listdir(image_path)
    # 设置置信度阈值
    # score_threshold = 0.9
    divide_size = make_pair(args.divide_size)
    stride = make_pair(args.stride)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = pdx.load_model(model_dir)
    result = []

    for i, img_file in enumerate(image_list):
        if img_file.split(".")[-1] not in valid_suffix:
            continue
        img_path = os.path.join(image_path, img_file)
        img = cv2.imread(img_path)
        divide_images, original_position = divide_according_sliding(img, divide_size, stride, True)
        pb = tqdm.tqdm(divide_images)
        divide_result = []
        for j, divide_img in enumerate(pb):
            pb.set_description("{}/{}".format(i, len(image_list)))
            res = model.predict(divide_img, eval_transforms)
            for k in range(len(res)):
                box = res[k]['bbox']
                xyxy = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
                res_data = {
                    'name': img_file,
                    'position': [int(p) for p in original_position[j]],
                    'category': res[k]['category'],
                    'bbox': xyxy,
                    'score': res[k]['score']
                }
                divide_result.append(res_data)
        result.append(divide_result)

    with open(os.path.join(save_dir, 'division_result.json'), 'w') as fp:
        json.dump(result, fp, indent=4, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description='division dataset')
    parser.add_argument('--data_dir', default='mydata/tile_round1_testA_20201231/testA_imgs', help='dataset path')
    parser.add_argument('--model_save_dir', default='output', help='save_path')
    parser.add_argument('--save_dir', default='result', help='predict result')
    parser.add_argument('--divide_size', nargs="+", type=int, default=[640, 640], help='size')
    parser.add_argument('--stride', nargs="+", type=int, default=[600, 600], help='stride')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("vision/inference.py")
    main(parse_args())
