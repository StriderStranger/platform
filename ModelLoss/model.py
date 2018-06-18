# CONTENT:
#   * a model class inherit from nn.Module.
#   * whose __init__() function integrate basic components of net.
#   * whose forward() function establish the procedure of net.
#
# Copyright (c) 2018 @WiederSeele.
# ========================================================================
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

}


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.base = 
    # TODO: 定义各种需要的层 eg. Conv2d, BatchNorm2d, Linear

  def forward(self, x):
    feat = self.base(x)
    # TODO: 用init()中定义的层搭建网络,返回最终的特征矩阵
    return feat
    

