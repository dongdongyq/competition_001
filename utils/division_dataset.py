# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/15
"""
import os
import argparse
import copy
import random
from common import check_or_make_dir, get_json_data, save_json


def save_coco_data(json_data, save_path, images, annotations):
    data = copy.deepcopy(json_data)
    data["images"] = images
    data["annotations"] = annotations
    save_json(save_path, data)


def save_voc_data(save_path, data):
    with open(save_path, "a") as fp:
        fp.write(data + "\n")


def generate_coco_train_val(root_dir, ann_file, ratio=0.3):
    ann_file_path = check_or_make_dir(root_dir, ann_file)
    json_data = get_json_data(ann_file_path)
    images = json_data["images"]
    annotations = json_data["annotations"]
    annotations_img_id = dict()
    for i, ann_item in enumerate(annotations):
        img_id = ann_item["image_id"]
        if img_id not in annotations_img_id:
            annotations_img_id[img_id] = [ann_item]
        else:
            annotations_img_id[img_id].append(ann_item)
    all_num = len(images)
    train_num = int(all_num * (1-ratio))
    random.shuffle(images)
    train_images = []
    train_annotations = []
    val_images = []
    val_annotations = []
    for i, image_item in enumerate(images):
        img_id = image_item["id"]
        print(i, img_id, image_item, annotations_img_id[img_id])
        if i < train_num:
            train_images.append(image_item)
            train_annotations.extend(annotations_img_id[img_id])
        else:
            val_images.append(image_item)
            val_annotations.extend(annotations_img_id[img_id])
    print("all data is {}, train data is {}, val data is {}".format(all_num, train_num, all_num-train_num))

    train_save_path = os.path.join(root_dir, "train.json")
    save_coco_data(json_data, train_save_path, train_images, train_annotations)

    val_save_path = os.path.join(root_dir, "val.json")
    save_coco_data(json_data, val_save_path, val_images, val_annotations)


def generate_voc_train_val(root_dir, ratio=0.3):
    image_dir = "images"
    ann_dir = "Annotations"
    seg_dir = "Segmentation"
    images_path = check_or_make_dir(root_dir, "images")
    images_file = os.listdir(images_path)
    all_num = len(images_file)
    train_num = int(all_num * (1-ratio))
    random.shuffle(images_file)
    train_path = os.path.join(root_dir, "train.txt")
    if os.path.exists(train_path):
        os.remove(train_path)
    val_path = os.path.join(root_dir, "val.txt")
    if os.path.exists(val_path):
        os.remove(val_path)

    seg_train_path = os.path.join(root_dir, "seg_train.txt")
    if os.path.exists(train_path):
        os.remove(train_path)
    seg_val_path = os.path.join(root_dir, "seg_val.txt")
    if os.path.exists(val_path):
        os.remove(val_path)
    print("all data is {}, train data is {}, val data is {}".format(all_num, train_num, all_num - train_num))
    for i, img_name in enumerate(images_file):
        xml_name = "{}.xml".format(img_name.split(".")[0])
        png_name = "{}.png".format(img_name.split(".")[0])
        data_xml = "{}/{} {}/{}".format(image_dir, img_name, ann_dir, xml_name)
        data_seg = "{}/{} {}/{}".format(image_dir, img_name, seg_dir, png_name)
        if i < train_num:
            save_voc_data(train_path, data_xml)
            save_voc_data(seg_train_path, data_seg)
        else:
            save_voc_data(val_path, data_xml)
            save_voc_data(seg_val_path, data_seg)


def main(args):
    root_dir = args.data_dir
    data_name = args.data_name
    ratio = float(args.ratio)
    assert data_name in ["coco", "voc"]
    if data_name == "coco":
        ann_file = "ann.json"
        generate_coco_train_val(root_dir, ann_file, ratio)
    else:
        generate_voc_train_val(root_dir, ratio)


def parse_args():
    parser = argparse.ArgumentParser(description='division dataset')
    parser.add_argument('--data_dir', default='../dataset/project_001/tile_round1', help='dataset path')
    parser.add_argument('--data_name', default='coco', help='dataset name')
    parser.add_argument('--ratio', default=0.3, help='val')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("division_dataset")
    main(parse_args())
