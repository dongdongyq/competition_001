# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/7
"""
import os
from tqdm import tqdm
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from model import MyModel
from dataset import get_dataloader, make_division
from torchvision import transforms
from torchvision.transforms import functional as F


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


def show_ori_img(img):
    if isinstance(img, torch.Tensor):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        mean = -mean / std
        std = 1 / std
        ori_img = F.normalize(img, mean, std)
        print(ori_img.shape)
        ori_img = ori_img.detach().numpy()
        ori_img = ori_img[0, 0, :, :] * 255
    else:
        ori_img = img
    # output = np.where(output > 100, 255, 0)
    ori_img = np.array(ori_img, dtype=np.uint8)
    cv2.namedWindow("ori_img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ori_img", 800, 800)
    cv2.imshow("ori_img", ori_img)


def infer(model, data_loader, criterion):

    pbar = enumerate(data_loader)
    # pbar = tqdm(pbar, total=len(data_loader))
    for i, (img, label) in pbar:
        # img = make_division(img)
        # label = make_division(label)
        img = torch.squeeze(img, dim=0)
        label = torch.squeeze(label, dim=0)
        show_ori_img(img)
        pred = detect_one_img(img, model, label, i)
        output = pred.view(pred.size(0), -1)
        target = label.view(label.size(0), -1)
        loss = criterion(output, target)
        # loss = compute_loss(pred, label, criterion)

        s = "loss: %.6f" % loss
        # pbar.set_description(s)
        print(s)


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


def main(opt):
    root = r"D:\learnspace\dataset\project_001\tile_round1"
    # model
    # model = MyModel()

    ckpt = torch.load(opt.weights)
    model = ckpt["model"]
    criterion = nn.BCELoss(reduction='mean')
    # criterion = my_criterion
    # trainloader
    train_loader = get_dataloader(root)

    # infer(model, train_loader, criterion)

    source = opt.source
    source = os.path.join(root, source)
    if os.path.isdir(source):
        files = os.listdir(source)
        for file in files:
            file_path = os.path.join(source, file)
            print(file_path)
            ori_img = cv2.imread(file_path)
            tensor = array_to_tensor(ori_img)
            print(tensor.shape)
            # show_ori_img(ori_img)

            detect_one_img(tensor, model)

            cv2.waitKey(0)



if __name__ == '__main__':
    print("inference")
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/model_epoch002_index02050.pt', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--source', type=str, default='train_imgs', help='hyperparameters path')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    opt = parser.parse_args()
    main(opt)
