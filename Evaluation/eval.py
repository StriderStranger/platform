# 测试模板
# Content:
#   1. 数据加载 (通过dataset类产生batch)
#   2. 模型加载
#   3. 测试循环
#   4. 整合模型的输出(pred/logit/feat),并保存在字典中
# 
# 常用的评估方法 [accuracy, TP/TN/FP/FN, precision, recall, f1, PRC, ROC, AUC, map, rank1~10]
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
from DataProcess.dataset import createDataset
from ModelLoss.model import Model
from Tools.utils import set_devices
from Tools.utils import load_ckpt
from Tools.utils import save_pickle



# 重要的全局参数
ckpt_path = '/home/iris/Dataset/exp/model_ckpt.pth'    # 这里是模型参数保存的文件
data_dir = '/home/iris/Dataset/'
test_path = '/home/iris/Dataset/exp/test_log.pkl'


def extract(model, ims):
  old_train_eval_model = model.training
  model.eval()
  feat = model(ims)
  model.train(old_train_eval_model)
  return feat



def main():
  TVT, TMO = set_devices((0,))

  ###########
  # TestSet #
  ###########
  dataset_kwargs = dict(
    num_threads = 2,
    batch_size = 32,
    crop_prob = 0,
    crop_ratio=1,
    scale=True,
    resize_h_w=(224, 224),
    im_mean=[0.486, 0.459, 0.408],
    im_std=[0.229, 0.224, 0.225],
    batch_dims='NCHW',
    mirror_type=['random', 'always', None][2])
  test_set = createDataset(data_dir, 'test', **dataset_kwargs)

  #########
  # Model #
  #########
  model = Model()
  TMO([model])
  load_ckpt([model], ckpt_path)

  #########
  # Test  #
  #########
  # TODO: 以batch为单位,用模型提取特征
  # >>> #
  total_batches = test_set.dataset_size // test_set.batch_size + 1
  feats, labels, im_names = [], [], []
  step = 0
  st = time.time()
  done = False
  while not done:
    ims_, labels_, im_names_, done = test_set.next_batch()
    ims_var = Variable(TVT(torch.from_numpy(ims_).float()))
    feat_ = extract(model, ims_var)
    feats.append(feat_)
    labels.append(label_)
    im_names.append(im_names_)
 
    step += 1
    if step % 20 == 0:
      print('{}/{} batches done, total {:.2f}s  [{}]'
            .format(step, total_batches, time.time() - st, part))
  feats = np.vstack(feats)
  labels = np.hstack(labels)
  im_names = np.hstack(im_names)
  # <<< #

  test_dict = dict(feats=feats,
                    labels=labels,
                    im_names=im_names)
  save_pickle(test_dict, test_path)
  print('extract all test data done.')
  return test_dict



if __name__ == "__main__":
  main()