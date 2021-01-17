#!/user/bin/python
# -*- coding: utf-8 -*-
"""
@File: main.py
@Description: 描述
@Author: dongyanqiang
@Email: 181228331@163.com
@Date: 2021/1/16
"""
import os
import argparse
import logging
from utils.common import execute_cmd, check_or_make_dir, make_pair
root = os.getcwd()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
    filename='tmp.log',
    filemode='w')


def main(args):
    # 1、切分图片
    dataset_dir = check_or_make_dir(root, args.data_dir)
    save_path = dataset_dir + "/tile_round1_divide"
    divide_size = make_pair(args.divide_size)
    stride = make_pair(args.stride)
    assert divide_size[0] > stride[0] and divide_size[1] > stride[1]
    division_image_cmd = "python work/utils/division_image.py"
    division_image_cmd += " --data_dir " + dataset_dir + "/tile_round1_train_20201231/"
    division_image_cmd += " --data_name voc"
    division_image_cmd += " --save_path " + save_path
    division_image_cmd += " --divide_size " + "{} {}".format(divide_size[0], divide_size[1])
    division_image_cmd += " --stride " + "{} {}".format(stride[0], stride[1])
    division_image_cmd += " --max_num -1"
    ret = execute_cmd(division_image_cmd)
    if not ret:
        return
    # 2、切分数据集。生成训练集文件列表、验证集文件列表
    data_dir = save_path + "/voc_size{}X{}".format(divide_size, divide_size)
    division_dataset_cmd = "python work/utils/division_dataset.py"
    division_dataset_cmd += " --data_dir " + data_dir
    division_dataset_cmd += " --data_name voc"
    division_dataset_cmd += " --ratio 0.2"
    ret = execute_cmd(division_dataset_cmd)
    if not ret:
        return
    # 3、训练模型
    train_cmd = "python work/paddle_model/paddle_train.py"
    train_cmd += " --data_dir " + data_dir
    train_cmd += " --backbone ResNet50_vd_ssld"
    train_cmd += " --save_dir " + args.model_save_dir
    execute_cmd(train_cmd)


def parse_args():
    parser = argparse.ArgumentParser(description='division dataset')
    parser.add_argument('--data_dir', default='../dataset/project_001/tile_round1', help='dataset path')
    parser.add_argument('--model_save_dir', default='output', help='save_path')
    parser.add_argument('--divide_size', nargs="+", type=int, default=[640, 640], help='size')
    parser.add_argument('--stride', nargs="+", type=int, default=[600, 600], help='stride')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("vision/main.py")
    main(parse_args())