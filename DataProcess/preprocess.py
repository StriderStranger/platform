# 标准的图像预处理类
#   1 resize
#   2 crop
#   3 scale
#   4 whiten
#   5 mirror
#   6 batch_dim
#
# Copyright (c) 2018 @wiederSeele
# ==========================================

import numpy as np
import cv2

class PreProcessIm(object):
  def __init__(self, crop_prob=0, crop_ratio=1.0, resize_h_w=None, scale=True, im_mean=None, im_std=None, mirror_type=None,
                batch_dims='NCHW')
    self.crop_prob = crop_prob
    self.crop_ratio = crop_ratio
    self.resize_h_w = resize_h_w
    self.scale = scale
    self.im_mean = im_mean
    self.im_std = im_std
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type
    self.check_batch_dims(batch_dims)
    self.batch_dims = batch_dims
    self.rand = np.random

  def __call__(self, im):
    return self.pre_process_im(im)

  @staticmethod
  def check_mirror_type(mirror_type):
    assert mirror_type in [None, 'random', 'always']

  @staticmethod
  def check_batch_dims(batch_dims):
    # 'N': batch size, 'C': num channels, 'H': im height, 'W': im width
    # PyTorch uses 'NCHW', while TensorFlow uses 'NHWC'.
    assert batch_dims in ['NCHW', 'NHWC']

  def set_mirror_type(self, mirror_type):
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type


  @staticmethod
  def rand_crop_im(im, new_size, rand=np.random):
    if (new_size[0] == im.shape[1]) and (new_size[1] == im.shape[0]):
      return im
    h_start = rand.randint(0, im.shape[0] - new_size[1])
    w_start = rand.randint(0, im.shape[1] - new_size[0])
    im = np.copy(
      im[h_start: h_start + new_size[1], w_start: w_start + new_size[0], :])
    return im


  def pre_process_im(self, im):
    """ `im` is a numpy array with shape [H,W,3] """
    # Crop.
    if (self.crop_ratio < 1) and (self.crop_prob > 0) and (self.rand.uniform() < self.crop_prob):
      h_ratio = self.rand.uniform(self.crop_ratio, 1)
      w_ratio = self.rand.uniform(self.crop_ratio, 1)
      crop_h = int(im.shape[0] * h_ratio)
      crop_w = int(im.shape[1] * w_ratio)
      im = self.rand_crop_im(im, (crop_w, crop_h))

    # Resize.
    if (self.resize_h_w is not None) and (self.resize_h_w != (im.shape[0], im.shape[1])):
      im = cv2.resize(im, self.resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)

    # Scale.
    if self.scale:
      im = im / 255.

    # Whiten.
    if self.im_mean is not None:
      im = im - np.array(self.im_mean)
    if self.im_mean is not None and self.im_std is not None:
      im = im / np.array(self.im_std).astype(float)

    # Mirror.
    mirrored = False
    if self.mirror_type == 'always' or (self.mirror_type == 'random' and self.rand.uniform() > 0.5):
      im = im[:, ::-1, :]
      mirrored = True

    # The original image has dims 'HWC', transform it to 'CHW'.
    if self.batch_dims == 'NCHW':
      im = im.transpose(2, 0, 1)

    return im, mirrored
