# 多线程数据打包功能
# Content:
#   3个类 (Counter, Enqueuer, Dataset) 实现并行数据加载
#   一个函数 (createDataset) 根据train和test的需求分别返回对应的对象
#   对象中的get_sample()成员负责加载一份数据和标签,并对数据预处理
#   对象中的next_batch()成员负责整合并返回一个batch
#
# 注意: 在createDataset()中传递的dataset_size这个参数,多线程索引的总量依据
# 
# Copyright (c) 2018 @wiederSeele
# ========================================================================

import threading 
import queue
import time
from PIL import Image
import numpy as np
from collections import defaultdict

from .preprocess import PreProcessIm
from Tools.utils import load_pickle



class Counter(object):
  """ A thread safe counter. """
  def __init__(self, val=0, max_val=0):
    self._value = val
    self.max_value = max_val
    self._lock = threading.Lock()

  def reset(self):
    with self._lock:
      self._value = 0

  def set_max_value(self, max_val):
    self.max_value = max_val

  def increment(self):
    with self._lock:
      if self._value < self.max_value:
        self._value += 1
        incremented = True
      else:
        incremented = False
      return incremented, self._value

  def get_value(self):
    with self._lock:
      return self._value


class Enqueuer(object):
  def __init__(self, get_element, num_elements, num_threads=1, queue_size=20):
    """
      Args:
        get_element: a function that takes a pointer and returns an element
        num_elements: total number of elements to put into the queue
        num_threads: num of parallel threads, >= 1
        queue_size: the maximum size of the queue. Set to some positive integer to save memory, otherwise, set to 0.
    """
    self.get_element = get_element
    assert num_threads > 0
    self.num_threads = num_threads
    self.queue_size = queue_size
    self.queue = queue.Queue(maxsize=queue_size)
    # The pointer shared by threads.
    self.ptr = Counter(max_val=num_elements)
    # The event to wake up threads, it's set at the beginning of an epoch.
    # It's cleared after an epoch is enqueued or when the states are reset.
    self.event = threading.Event()
    self.reset_event = threading.Event()
    self.stop_event = threading.Event()
    self.threads = []
    for _ in range(num_threads):
      thread = threading.Thread(target=self.enqueue)
      # Set the thread in daemon mode, so that the main program ends normally.
      thread.daemon = True
      thread.start()
      self.threads.append(thread)

  def start_ep(self):
    """Start enqueuing an epoch."""
    self.event.set()

  def end_ep(self):
    """When all elements are enqueued, let threads sleep to save resources."""
    self.event.clear()
    self.ptr.reset()

  def reset(self):
    """Reset the threads, pointer and the queue to initial states. In common
    case, this will not be called."""
    self.reset_event.set()    # 打断了equeue()中的queue.put()
    self.event.clear()
    time.sleep(5)
    self.reset_event.clear()
    self.ptr.reset()
    self.queue = queue.Queue(maxsize=self.queue_size)

  def set_num_elements(self, num_elements):
    """Reset the max number of elements."""
    self.reset()
    self.ptr.set_max_value(num_elements)

  def stop(self):
    """Wait for threads to terminate."""
    self.stop_event.set()
    for thread in self.threads:
      thread.join()

  def enqueue(self):
    while not self.stop_event.isSet():
      if not self.event.wait(0.5): continue
      incremented, ptr = self.ptr.increment()
      if incremented:
        element = self.get_element(ptr - 1)
        while not self.stop_event.isSet() and not self.reset_event.isSet():
          try:
            self.queue.put(element, timeout=0.5)
            break
          except:
            pass
      else:
        print('can not incremented')
        self.end_ep()
    print('Exiting thread {}!!!!!!!!'.format(threading.current_thread().name))


class Dataset(object):
  def __init__(self, im_dir, im_names, batch_size, dataset_size, num_threads=2, **pre_process_im_kwargs):
    """实现将数据打包.
    Args:
      dataset_size: all dataset length.
      batch_size: batch size.
      pre_process_im_kwargs: args for preprocess function
    """
    self.im_dir = im_dir
    self.im_names = im_names

    self.pre_process_im = PreProcessIm(**pre_process_im_kwargs)
    self.enqueuer = Enqueuer(get_element=self.get_sample, num_elements=dataset_size, num_threads=num_threads, queue_size=20)
    self.batch_size = batch_size
    self.ptr = 0
    self.dataset_size = dataset_size
    self.ep_done = True

    # TODO: 一些数据信息的初始化: 如图片地址,文件名集合等

  def stop(self):
    self.enqueuer.stop()

  def get_sample(self, ptr):
    '''TODO: 加载一份数据,包括raw_data和label,并预处理.
    Args:
      ptr: 对self.dataset_size的索引, ptr的传入由enqueue来控制
    '''
    # >>> #
    name = self.im_names[ptr]
    im = np.array(Image.open(osp.join(self.im_dir, name)))
    label = parse_im_name(name, 'cam')
    im, mirrored = self.pre_process_im(im)
    # <<< #
    return im, label, name

  def next_batch(self):
    """负责数据的读取和打包
    """
    if self.ep_done:
      np.random.shuffle(self.ids)
      self.enqueuer.start_ep()
      self.ptr = 0

    self.ep_done = False
    samples = []
    for _ in range(self.batch_size):
      if self.ptr >= self.dataset_size:
        self.ep_done = True
        break
      else:
        self.ptr += 1
        sample = self.enqueuer.queue.get()
        samples.append(sample)
    if self.ptr >= self.dataset_size:
      self.ep_done = True

    # TODO: 将返回的n份数据整理成batch
    # >>> #
    im_list, labels, im_names = list(zip(*samples))
    ims = np.stack(im_list)
    labels = np.array(labels)
    im_names = np.array(im_names)
    # <<< #
    return ims, labels, im_names, self.ep_done


def createDataset(data_dir, part='trainval', batch_size=32, **kwargs):
  '''TODO 根据part选择返回哪部分的dataset'''
  # >>> #
  assert part in ('trainval', 'test', 'query', 'gallery')
  im_dir = osp.join(data_dir, 'images')
  pickle_file = osp.join(data_dir, 'dataset_split.pkl')
  split = load_pickle(pickle_file)
  im_names = split['{}_im_names'.format(part)]

  if part == 'trainval':
    ret = Dataset(im_dir, im_names, batch_size, len(im_names), **kwargs)
  if part == 'trainval':
    ret = Dataset(im_dir, im_names, batch_size, len(im_names), **kwargs)
  # >>> #
  return ret
