#!/user/bin/python
# -*- coding: utf-8 -*-
"""
@File: dataset.py
@Description: 描述
@Author: dongyanqiang
@Email: 181228331@163.com
@Date: 2021/1/16
"""
from paddlex.det import transforms
import paddlex as pdx

train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250), transforms.RandomDistort(),
    transforms.RandomExpand(), transforms.RandomCrop(), transforms.Resize(
        target_size=608, interp='RANDOM'), transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.Resize(
        target_size=608, interp='CUBIC'), transforms.Normalize()
])

# faster-rcnn
# train_transforms = transforms.Compose([
#     transforms.RandomDistort(), transforms.RandomCrop(),
#     transforms.RandomHorizontalFlip(),
#     # transforms.ResizeByShort(
#     #     short_size=[800], max_size=1333),
#     transforms.Normalize(
#             mean=[0.5], std=[0.5]), transforms.Padding(coarsest_stride=32)
# ])
#
# eval_transforms = transforms.Compose([
#     # transforms.ResizeByShort(
#     #     short_size=800, max_size=1333),
#     transforms.Normalize(),
#     transforms.Padding(coarsest_stride=32),
# ])


def get_dataset(root_base):
    # 定义训练和验证所用的数据集
    train_dataset = pdx.datasets.VOCDetection(
        data_dir=root_base,
        file_list=root_base+'train.txt',
        label_list=root_base+'labels.txt',
        transforms=train_transforms,
        num_workers=8,
        shuffle=True)
    eval_dataset = pdx.datasets.VOCDetection(
        data_dir=root_base,
        file_list=root_base+'val.txt',
        label_list=root_base+'labels.txt',
        num_workers=8,
        transforms=eval_transforms)
    return train_dataset, eval_dataset

