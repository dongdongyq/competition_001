# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/5
"""
import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import check_or_make_dir


class MyDataset(Dataset):
    def __init__(self, root_dir, is_train=True, max_len=None, transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        images_dir = check_or_make_dir(root_dir, "images")
        labels_dir = check_or_make_dir(root_dir, "labels")
        if is_train:
            dir_name = "train"
        else:
            dir_name = "val"
        self.images_dir_path = check_or_make_dir(images_dir, dir_name)
        self.labels_dir_path = check_or_make_dir(labels_dir, dir_name)

        self.images_name = os.listdir(self.images_dir_path)
        self.max_len = max_len
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if self.max_len is not None:
            return min(len(self.images_name), self.max_len)
        return len(self.images_name)

    def get_image(self, idx):
        image_path = check_or_make_dir(self.images_dir_path, self.images_name[idx])
        img = cv2.imread(image_path)
        return img

    def get_label(self, idx):
        boxes = []
        masks = []
        categorys = []
        label_path = check_or_make_dir(self.labels_dir_path, self.images_name[idx].split(".")[0]+".txt")
        with open(label_path, "r") as fp:
            for line in fp.readlines():
                if line.strip():
                    line = line.replace("\n", "").split(" ")
                    boxes.append(line[1:])
                    masks.append(line[1:])
                    categorys.append(line[0])
        labs = {
            "boxes": torch.tensor(np.array(boxes, dtype=np.float), dtype=torch.float),
            "masks": torch.tensor(np.array(masks, dtype=np.float), dtype=torch.float),
            "categorys": torch.tensor(np.array(categorys, dtype=np.int), dtype=torch.int),
            "image_id": idx,
        }
        return labs

    def get_label_mask(self, img, labels):
        size = img.shape[:2]
        label_mask = np.zeros(size, dtype=np.uint8)
        size = torch.tensor(size[::-1])
        masks = labels["masks"]
        for mask in masks:
            center = mask[:2] * size
            wh = mask[2:] * size
            point_lt = tuple(np.array(center - wh/2, dtype=np.int))
            point_rb = tuple(np.array(center + wh/2, dtype=np.int))
            cv2.rectangle(label_mask, point_lt, point_rb, 255, -1)
        return label_mask

    def __getitem__(self, idx):
        img = self.get_image(idx)
        labels = self.get_label(idx)
        label_mask = self.get_label_mask(img, labels)
        if self.transform:
            img = self.transform(img)
        label_mask = torch.tensor(label_mask/255, dtype=torch.float)
        label_mask = torch.unsqueeze(label_mask, 0)
        return img, labels, label_mask


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


def show(img, win_name):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def collate_fn(batch):
    images, labels, label_masks = list(zip(*batch))
    batched_imgs = torch.stack(images)
    batched_masks = torch.stack(label_masks)
    return batched_imgs, list(labels), batched_masks


def get_dataloader(root, is_train, batch_size=2, max_len=None):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = MyDataset(root_dir=root, is_train=is_train, max_len=max_len, transform=data_transform)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataset_loader


if __name__ == '__main__':
    print("dataset")
    import time
    print(int(time.time()))
    root = r"D:\learnspace\dataset\project_001\divide_tile_round1"
    dataloader = get_dataloader(root, True, 10)
    for img, label, label_mask in dataloader:
        # img = make_division(img)
        # label = make_division(label)
        print(img.shape, len(label), label_mask.shape, torch.sum(label_mask))

