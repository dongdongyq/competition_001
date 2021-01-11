# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/5
"""
import os
import torch
import cv2
import numpy as np
import numbers
from PIL import Image
from collections.abc import Sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class MyCrop(torch.nn.Module):
    @staticmethod
    def get_params(img, center, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = F._get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = center[1] - th // 2
        j = center[0] - tw // 2

        return i, j, th, tw

    def __init__(self, center, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))
        self.center = tuple(_setup_size(
            center, error_msg="Please provide only two dimensions (x, y) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.center, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        train_imgs_dir = os.path.join(root_dir, "train_imgs")
        train_labels_dir = os.path.join(root_dir, "train_labels")
        image_files_list = os.listdir(train_imgs_dir)
        label_files_list = os.listdir(train_labels_dir)
        assert len(image_files_list) == len(label_files_list)
        self.images_list = [os.path.join(train_imgs_dir, file) for file in image_files_list]
        self.labels_list = [os.path.join(train_labels_dir, file) for file in label_files_list]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        # img = Image.open(self.images_list[idx])
        # label = Image.open(self.labels_list[idx])
        img = cv2.imread(self.images_list[idx])
        # img = cv2.resize(img, (1024, 1024))
        label = cv2.imread(self.labels_list[idx], 0)
        # label = cv2.resize(label, (1024, 1024))
        label = np.where(label > 0, 255, 0)
        label = np.array(label, dtype=np.uint8)
        centers_pos = get_crop_box(label)
        label = label[:, :, np.newaxis]
        data = np.concatenate((img, label), axis=-1)
        if self.transform:
            data = self.transform(data)
        imgs = []
        labels = []
        assert len(centers_pos) > 0
        for x, y, area in centers_pos:
            out_data = MyCrop((x, y), (128, 128))(data)
            img = out_data[:3, :, :]
            label = out_data[3:, :, :]
            imgs.append(img)
            labels.append(label)
        imgs = torch.stack(imgs)
        labels = torch.stack(labels)
        if self.target_transform:
            imgs = self.target_transform(imgs)
        return imgs, labels


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


def filter_center_pos(center_x, center_y, centers_pos, filter=120):
    for x, y, _ in centers_pos:
        if abs(center_x - x) < filter and abs(center_y - y) < filter:
            return False
    return True


def get_crop_box(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    centers_pos = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 8:
            bbox = cv2.boundingRect(contour)
            center_x = bbox[0] + bbox[2] // 2
            center_y = bbox[1] + bbox[3] // 2
            if filter_center_pos(center_x, center_y, centers_pos):
                centers_pos.append((center_x, center_y, area))
    return centers_pos


def show(img, win_name):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(10)


def show_tensor(img, name="show"):
    img = img[0, :, :, :] * 255
    img = img.permute(1, 2, 0)
    img = img.detach().numpy()
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    show(img, name)


def get_dataloader(root):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.ToPILImage(),
        # transforms.FiveCrop(1024),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        # transforms.RandomResizedCrop(3072),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = MyDataset(root_dir=root, transform=data_transform, target_transform=target_transform)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataset_loader


if __name__ == '__main__':
    print("dataset")
    root = r"D:\learnspace\dataset\project_001\tile_round1"
    dataloader = get_dataloader(root)
    for img, label in dataloader:
        # img = make_division(img)
        # label = make_division(label)
        img = torch.squeeze(img, dim=0)
        label = torch.squeeze(label, dim=0)
        print(img.size(), label.size())
        print(torch.sum(label))
        target = label.detach().numpy()
        target = target[0, 0, :, :] * 255
        target = np.array(target, dtype=np.uint8)
        print(np.sum(target))
        show_tensor(label, "label")
        show_tensor(img)
