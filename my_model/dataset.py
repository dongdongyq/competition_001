# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/5
"""
import os
import torch
from PIL.ImageDraw import ImageDraw, Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import transforms as T

from utils.utils import check_or_make_dir, get_json_data


class MyCocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None, max_len=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_dir = check_or_make_dir(root_dir, "images")
        ann_file_path = os.path.join(root_dir, ann_file)
        self.ann_data = get_json_data(ann_file_path)
        self.max_len = max_len
        self.transform = transform

        self.category = self.get_category()
        self.images = self.ann_data["images"]
        self.annotations = self.get_annotations()

    def __len__(self):
        if self.max_len is not None:
            return min(len(self.ann_data["images"]), self.max_len)
        return len(self.ann_data["images"])

    def get_category(self):
        category = dict()
        categories = self.ann_data["categories"]
        for item in categories:
            category[item["id"]] = item["name"]
        return category

    def get_annotations(self):
        annotations = dict()
        anno = self.ann_data["annotations"]
        for item in anno:
            if item["image_id"] not in annotations:
                annotations[item["image_id"]] = [item]
            else:
                annotations[item["image_id"]].append(item)
        return annotations

    def get_image(self, idx):
        image_item = self.images[idx]
        image_path = check_or_make_dir(self.images_dir, image_item["file_name"])
        img = Image.open(image_path)
        return img

    def get_label(self, idx):
        image_item = self.images[idx]
        img_id = image_item["id"]
        height = image_item["height"]
        width = image_item["width"]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask = Image.fromarray(mask)
        draw = ImageDraw(mask)
        annotations = self.annotations[img_id]
        for ann_item in annotations:
            segmentation = ann_item["segmentation"]
            seg = np.array(segmentation, dtype=np.int)
            draw.polygon(tuple(seg[0]), fill=(1, ))
            # cv2.drawContours(mask, [seg], 0, 255, -1)
        return mask

    def __getitem__(self, idx):
        img = self.get_image(idx)
        label = self.get_label(idx)
        if self.transform:
            img, label = self.transform(img, label)
        return img, label


def make_division(tensor, md=32):
    height = tensor.shape[2]
    width = tensor.shape[3]
    phl, phr, pwl, pwr = 0, 0, 0, 0
    if height % md != 0:
        phl = (md - height % md) // 2
        phr = (md - height % md) - phl
    if width % md != 0:
        pwl = (md - width % md) // 2
        pwr = (md - width % md) - pwl
    zeroPad2d = torch.nn.ZeroPad2d((pwl, pwr, phl, phr))
    return zeroPad2d(tensor)


def get_transform(train):
    base_size = 520
    crop_size = 480

    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transforms = list()
    # transforms.append(T.RandomResize(min_size, max_size))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(T.RandomCrop(crop_size))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def get_dataloader(root, ann_file, train=True, batch_size=1, max_len=None):
    data_transform = get_transform(train)
    dataset = MyCocoDataset(root_dir=root, ann_file=ann_file, transform=data_transform, max_len=max_len)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset_loader


if __name__ == '__main__':
    print("dataset")
    import time
    from utils.common import show
    print(int(time.time()))
    root = r"D:\learnspace\dataset\project_001\tile_round1_divide\coco_size128X128"
    ann_file = "val.json"
    dataloader = get_dataloader(root, ann_file, False)
    for img, label in dataloader:
        # img = make_division(img)
        # label = make_division(label)
        print(img.shape, label.shape, torch.sum(label))
        b_count = torch.bincount(label.to(torch.int64).flatten())
        print(b_count, torch.sum(label))


