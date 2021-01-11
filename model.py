# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/4
"""
import torch.nn as nn
from torchsummary import summary


class CBABlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=0, b=False, act=True):
        super(CBABlock, self).__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=b),
            nn.BatchNorm2d(c_out),
            nn.ReLU6() if act is True else nn.Identity()
        )

    def forward(self, x):

        return self.cba(x)


class CTBABlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=0, op=0, b=False, act=True):
        super(CTBABlock, self).__init__()
        self.ctba = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, k, s, p, op, bias=b),
            nn.BatchNorm2d(c_out),
            nn.ReLU6() if act is True else nn.Identity()
        )

    def forward(self, x):
        return self.ctba(x)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        c_in = 3
        c_out = 1
        encode = [16, 32, 64, 128]
        decode = encode[::-1]
        self.encode_list = nn.ModuleList()
        for c in encode:
            self.encode_list.append(CBABlock(c_in, c, k=1, act=False))
            self.encode_list.append(CBABlock(c, c, s=2, p=1))
            c_in = c

        self.decode_list = nn.ModuleList()
        for c in decode:
            self.decode_list.append(CTBABlock(c_in, c, k=1, act=False))
            self.decode_list.append(CTBABlock(c, c, s=2, p=1, op=1))
            c_in = c
        self.conv = nn.Conv2d(c_in, c_out, 1, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        for m in self.encode_list:
            x = m(x)
        for m in self.decode_list:
            x = m(x)
        x = self.conv(x)
        x = self.act(x)
        return x


if __name__ == '__main__':
    print("model")
    model = MyModel()
    summary(model, (3, 224, 224))
    # print(model)
