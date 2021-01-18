#!/user/bin/python
# -*- coding: utf-8 -*-
"""
@File: test.py
@Description: 描述
@Author: dongyanqiang
@Email: 181228331@163.com
@Date: 2021/1/12
"""
import torch
import torch.utils
import torch.utils.cpp_extension
from torch.backends import cudnn

# CUDA TEST
x = torch.Tensor([1.0])
xx = x.cuda()
print(xx)

# cudnn test
print(cudnn.is_acceptable(xx))

print(torch.version.cuda)
print(torch.utils.cpp_extension.CUDA_HOME)

# https://github.com/pytorch/pytorch/blob/master/torch/utils/cpp_extension.py#L24

if __name__ == "__main__":
    print("vision/test.py")
    # main()