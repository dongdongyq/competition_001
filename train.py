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
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from model import MyModel
from dataset import get_dataloader, make_division


def my_criterion(output, target):
    return -torch.log(output)


def compute_loss(output, target, criterion):
    # print(output.shape, target.shape)
    pos = torch.sum(target)
    area = target.shape[-1]
    scale = pos / area
    # target = torch.clip(target, 0.0001, 0.9999)
    # output = torch.clip(output, 0.0001, 0.9999)
    # loss = (area - pos) * target * criterion(output, target) + pos * (1 - target) * criterion(1 - output, target)
    # loss = torch.mean(loss)
    loss = criterion(output, target)
    return loss


def save(epoch, index, loss):
    with open("./weights/loss.txt", "a") as fp:
        fp.write("epoch: %d, index: %d, loss: %.6f\n" % (epoch, index, loss))


def save_output(pred, target, epoch, index):
    # pred = torch.sigmoid(pred)
    output = pred.detach().numpy()
    output = output[0, 0, :, :] * 255
    # output = np.where(output > 100, 255, 0)
    output = np.array(output, dtype=np.uint8)

    target = target.detach().numpy()
    target = target[0, 0, :, :] * 255
    target = np.array(target, dtype=np.uint8)

    save_img = np.concatenate((output, target), axis=1)
    cv2.imwrite("./outputs/epoch{:03}_index{:05}.jpg".format(epoch, index), save_img)


def train_one_epoch(model, train_loader, optimizer, epoch, criterion):
    pbar = enumerate(train_loader)
    pbar = tqdm(pbar, total=len(train_loader))
    for i, (img, label) in pbar:
        # img = make_division(img)
        # label = make_division(label)
        img = torch.squeeze(img, dim=0)
        label = torch.squeeze(label, dim=0)
        pred = model(img)

        output = pred.view(pred.size(0), -1)
        target = label.view(label.size(0), -1)
        loss = compute_loss(output, target, criterion)
        # loss = compute_loss(pred, label, criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        s = "epoch: %d, loss: %.6f" % (epoch, loss)
        pbar.set_description(s)
        save(epoch, i, loss)
        # save_output(pred, label, epoch, i)
        # if i % 10 == 0:
        #     s = "epoch: %d, loss: %.4f" % (epoch, loss)
        #     pbar.set_description(s)
        #     save(epoch, i, loss)
        if i % 10 == 0:
            save_output(pred, label, epoch, i)
        if i % 50 == 0:
            ckpt = {'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer.state_dict()}
            torch.save(ckpt, "./weights/model_epoch{:03}_index{:05}.pt".format(epoch, i))


def main(opt):
    # model
    model = MyModel()

    # optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
    # if opt.adam:
    #     optimizer = optim.Adam(pg0, lr=1e-2, betas=(0.98, 0.999))  # adjust beta1 to momentum
    # else:
    optimizer = optim.SGD(pg0, lr=1e-2, momentum=0.98, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': 0.001})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    criterion = nn.BCELoss(reduction='mean')
    # criterion = my_criterion

    # trainloader
    root = r"D:\learnspace\dataset\project_001\tile_round1"
    train_loader = get_dataloader(root)

    # train
    start_epoch = 0
    for epoch in range(start_epoch, opt.epochs):
        model.train()
        train_one_epoch(model, train_loader, optimizer, epoch, criterion)
        ckpt = {'epoch': epoch,
                'model': model,
                'optimizer': optimizer.state_dict()}
        torch.save(ckpt, "./weights/model_{}.pt".format(epoch))
        scheduler.step()


if __name__ == '__main__':
    print("train")
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='epochs')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    opt = parser.parse_args()
    main(opt)