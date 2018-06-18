# 通用数据整理统一   (根据需求修改对应的内容,对于简单的数据可以跳过这步)
#   1 读入所有数据文件名
#   2 统一路径,拷贝数据,统一文件名
#   3 按训练,测试划分数据,记录在字典中
#
# Copyright (c) 2018 @wiederSeele
# ===========================================================

import os
import os.path as osp
import numpy as np
import glob
import shutil
from Tools.utils import save_pickle


def load_im_name(from_dir):
  im_names = []
  nums = []
  # TODO: 将所有文件名读入im_names, 并将每个子集的数量记录在num中
  # >>> #
  sub_dirs = ['train', 'test', 'query']
  for sub_dir in sub_dirs:
    im_dir = osp.join(from_dir, sub_dir)
    im_names_ = glob.glob(osp.join(im_dir, '*.jpg'))
    im_names += list(im_names_)
    nums.append(len(im_names_))
  # <<< #
  return im_names, nums


def copy_and_rename(old_im_names, to_dir):
  new_im_names = []
  for im_name in old_im_names:
    base_name = osp.basename(im_name)

    # TODO: 定义新的文件名规则 (包含类别信息)
    # >>> #
    new_im_name_tmpl = '{:08d}_{:04d}_{:08d}.jpg'
    info1 = parse_origin_name(base_name)
    info2 = parse_origin_name(base_name)
    info3 = parse_origin_name(base_name)
    new_im_name = new_im_name_tmpl.format(info1, info2, info3)
    # <<< #
    shutil.copy(im_name, osp.join(to_dir, new_im_name))
    new_im_names.append(new_im_name)
  return new_im_names


def split_dataset(im_names, nums):
  split = dict()
  # TODO: 根据nums划分数据
  keys = ['trainval', 'test', 'query']
  inds = [0] + nums
  inds = np.cumsum(np.array(inds))
  for i, k in enumerate(keys):
    split[k] = im_names[inds[i] : inds[i+1]]
  return split


def transform(from_dir, to_dir):
  if not osp.exists(to_dir):
    os.makedirs(to_dir)

  # TODO: 加载原文件名, 拷贝文件并统一文件名, 将新文件名按划分记录在字典中 (根据需求选择)
  # >>> #
  old_names, nums = load_im_name(from_dir)
  new_names = copy_and_rename(old_names, to_dir)
  split = split_dataset(new_names, nums)
  # <<< #

  save_pickle(split, osp.join(to_dir, 'dataset_split.pkl'))
  print('Packing data done.')
  for k,value in split.items():
    print('{} has {} items.'.format(k, len(value)))



if __name__ == '__main__':
  from_dir = '/home/iris/Dataset/'
  to_dir = '/home/iris/Dataset/images/'
  transform(from_dir, to_dir)