# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/5
"""
import math
from tqdm import tqdm
import argparse
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from my_model.model import MyModel
from my_model.dataset import get_dataloader

from utils.utils import check_or_make_dir, ConfusionMatrix


def compute_loss(output, target, criterion):
    # print(output.shape, target.shape)
    target = target.to(torch.float32)
    loss = criterion(output, target)
    return loss


def save(epoch, index, loss, save_dir):
    with open(save_dir + "/loss.txt", "a") as fp:
        fp.write("epoch: %d, index: %d, loss: %.6f\n" % (epoch, index, loss))


def save_output(pred, target, epoch, index, save_path):
    # pred = torch.sigmoid(pred)
    output = pred.detach().numpy()
    output = output[0, 0, :, :] * 255
    output = np.where(output > 120, 255, 0)
    output = np.array(output, dtype=np.uint8)

    target = target.detach().numpy()
    target = target[0, :, :] * 255
    target = np.array(target, dtype=np.uint8)
    save_img = np.concatenate((output, target), axis=1)
    cv2.imwrite(save_path + "/epoch{:02}_index{:03}.jpg".format(epoch, index), save_img)


def train_one_epoch(model, device, train_loader, optimizer, epoch, criterion, save_dir):
    save_path = check_or_make_dir(save_dir, "outputs", mkdir=True)
    pbar = enumerate(train_loader)
    pbar = tqdm(pbar, total=len(train_loader))
    for i, (img, label) in pbar:
        img = img.to(device)
        label_mask = label.to(device)
        pred = model(img)

        output = pred.view(pred.size(0), -1)
        target = label_mask.view(label_mask.size(0), -1)
        loss = compute_loss(output, target, criterion)
        # loss = compute_loss(pred, label, criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        s = "epoch: %d, loss: %.6f" % (epoch, loss)
        pbar.set_description(s)
        save(epoch, i, loss, save_dir)
        if i % 50 == 0:
            save_output(pred, label_mask, epoch, i, save_path)


def evaluate(val_dataset, model, device, epoch, save_dir):
    save_path = check_or_make_dir(save_dir, "evaluate", mkdir=True)
    confmat = ConfusionMatrix()
    model.eval()
    pbar = enumerate(val_dataset)
    pbar = tqdm(pbar, total=len(val_dataset))
    for i, (img, label) in pbar:
        img = img.to(device)
        label_mask = label.to(device)
        pred = model(img)
        confmat.update(pred.flatten(), label_mask.flatten())
        if i % 10 == 0:
            save_output(pred, label_mask, epoch, i, save_path)
    return confmat


def main(opt):
    device = torch.device(opt.device)
    # model
    model = MyModel()
    model.to(device)
    pretrained = opt.pretrained
    if pretrained:
        model_url = pretrained
        ckpt = torch.load(model_url)
        model = ckpt["model"]
        model.to(device)

    # optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=1e-4, betas=(0.98, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=1e-3, momentum=0.98, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': 0.001})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    criterion = nn.BCELoss(reduction='mean')
    # criterion = my_criterion

    # trainloader
    root = r"D:\learnspace\dataset\project_001\tile_round1_divide\coco_size128X128"
    train_loader = get_dataloader(root, "train.json", True, batch_size=1)
    val_loader = get_dataloader(root, "val.json", False, batch_size=1)

    time_int = int(time.time())
    save_dir = check_or_make_dir("../weights", "train_{}".format(time_int), mkdir=True)
    # train
    start_epoch = 0
    for epoch in range(start_epoch, opt.epochs):
        model.train()
        train_one_epoch(model, device, train_loader, optimizer, epoch, criterion, save_dir)
        ckpt = {'epoch': epoch,
                'model': model,
                'optimizer': optimizer.state_dict()}
        torch.save(ckpt, save_dir + "/model_{:02}.pt".format(epoch))
        scheduler.step()
        confmat = evaluate(val_loader, model, device, epoch, save_dir)
        print(confmat)


if __name__ == '__main__':
    print("train")
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    opt = parser.parse_args()
    main(opt)