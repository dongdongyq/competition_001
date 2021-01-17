#!/user/bin/python
# -*- coding: utf-8 -*-
"""
@File: paddle_train.py
@Description: 描述
@Author: dongyanqiang
@Email: 181228331@163.com
@Date: 2021/1/16
"""
import os
import argparse
# 环境变量配置，用于控制是否使用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddle_model.dataset import get_dataset
import paddlex as pdx


def train(args):
    root_base = args.data_dir
    train_dataset, eval_dataset = get_dataset(root_base)
    # num_classes 需要设置为包含背景类的类别数，即: 目标类别数量 + 1
    num_classes = len(train_dataset.labels) + 1

    model = pdx.det.FasterRCNN(
        num_classes=num_classes,
        backbone=args.backbone,
        with_dcn=True,
        fpn_num_channels=64,
        with_fpn=True,
        test_pre_nms_top_n=500,
        test_post_nms_top_n=300)

    model.train(
        num_epochs=100,
        train_dataset=train_dataset,
        train_batch_size=32,
        eval_dataset=eval_dataset,
        save_interval_epochs=1,
        log_interval_steps=10,
        learning_rate=0.0025,
        lr_decay_epochs=[60, 70],
        warmup_steps=5000,
        save_dir=args.save_dir,
        use_vdl=True)


def parse_args():
    parser = argparse.ArgumentParser(description='division dataset')
    parser.add_argument('--data_dir', default='', help='dataset path')
    parser.add_argument('--backbone', default='ResNet50_vd_ssld', help='backbone')
    parser.add_argument('--save_dir', default="output/faster_rcnn_r50_vd_dcn", help='save dir')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("train")
    train(parse_args())