# 训练模板
# Content:
#   1. 数据加载 (通过dataset类产生batch)
#   2. 模型,损失函数,优化器加载
#   3. Log的预定义 (给出需要记录的变量)
#   4. 训练循环  (epoch_done控制)
#   5. 将训练结果和模型保存
#
# Copyright (c) 2018 @WiederSeele.
# =============================================

import sys
sys.path.insert(0, '.')
import time
import numpy as np
import os
import os.path as osp
import pickle as pickle

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from DataProcess.dataset import createDataset
from ModelLoss.model import Model
from Tools.utils import set_devices
from Tools.utils import adjust_lr_exp
from Tools.utils import Log
from Tools.utils import save_ckpt
from Tools.utils import save_pickle


# 重要的全局参数
base_lr = 2e-4
total_epochs = 100
decay_ep = 50
ckpt_path = '/home/iris/Dataset/exp/model_ckpt.pth'    # 这里是模型参数保存的文件
data_dir = '/home/iris/Dataset/'
log_path = '/home/iris/Dataset/exp/train_log.pkl'     # log信息

if not osp.exists(osp.dirname(ckpt_path)):
  os.makedirs(osp.dirname(ckpt_path))
if not osp.exists(osp.dirname(log_path)):
  os.makedirs(osp.dirname(log_path))



def main():
  ##########
  # DEVICE #
  ##########
  TVT, TMO = set_devices((0,))

  ###########
  # Dataset #
  ###########
  # TODO: 需要传递给Dataset的参数
  # >>> #
  dataset_kwargs = dict(
    num_threads = 2,
    batch_size = 64,
    crop_prob = 0,
    crop_ratio=1,
    scale=True,
    resize_h_w=(256, 128),
    im_mean=[0.486, 0.459, 0.408],
    im_std=[0.229, 0.224, 0.225],
    batch_dims='NCHW',
    mirror_type=['random', 'always', None][0])
  train_set = createDataset(data_dir, 'trainval', **dataset_kwargs)
  # <<< #

  ##########
  # MODELS #
  ##########
  model = Model()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0005)
  modules_optims = [model, optimizer]
  TMO(modules_optims)


  ##############
  # Log 预准备  #
  ##############
  # >>> #
  AMlist = ['l_prec', 'l_d_ap', 'l_d_an', 'local_loss', 'loss']
  curves = {}
  for item in AMlist:
    curves[item] = []
  # >>> #


  ############
  # Training #
  ############
  print('Start Training ! ! ! ')
  for ep in range(0, total_epochs):
    adjust_lr_exp(optimizer, base_lr, ep+1, total_epochs, decay_ep)
    modules_optims[0].train()
    
    eplog = Log(AMlist)
    ep_st = time.time()
    step = 0
    epoch_done = False
    while not epoch_done:
      step += 1
      if step%30 == 0:
        print(step)
      step_st = time.time()

      # TODO: >>>>>> 加载并训练一个batch >>>>> 
      # >>> #
      ims, labels, im_names, epoch_done = train_set.next_batch()
      ims_var = Variable(TVT(torch.from_numpy(ims).float()))
      labels_v = Variable(TVT(torch.from_numpy(labels).long()))
      feat = model(ims_var)
      loss = criterion(feat)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # <<< #

      # TODO: 每一步的log更新   Eg. g_dist_an, g_dist_ap是自定义损失函数的返回值
      loss_meter = loss.data.cpu().numpy().flatten()[0]
      eplog.update({'loss':loss_meter})
    
    eplog.printlog(ep, time.time() - ep_st)
    for k, value in eplog.allMeters.items():
      curves[k].append(value.avg)

    ##############
    # SAVE MOdel #
    ##############
    save_ckpt(modules_optims, ep + 1, 0, ckpt_path)

  save_pickle(curves, log_path)
  print('Training done.')
  return curves






if __name__ == "__main__":
  main()
