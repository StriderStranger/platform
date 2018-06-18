# 基本工具
# Content:
#   1. 模型,变量gpu化工具
#   2. 变量记录工具
#   3. 训练Log记录工具
#
# Copyright (c) 2018 @WiederSeele.
# =============================================


import torch
from torch.autograd import Variable


# 模型,变量gpu化工具

class TransferVarTensor(object):
  """Return a copy of the input Variable or Tensor on specified device."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, var_or_tensor):
    return var_or_tensor.cpu() if self.device_id == -1 \
      else var_or_tensor.cuda(self.device_id)

class TransferModulesOptims(object):
  """Transfer optimizers/modules to cpu or specified gpu."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, modules_and_or_optims):
    for item in modules_and_or_optims:
      if isinstance(item, torch.optim.Optimizer):
        transfer_optim_state(item.state, device_id=self.device_id)
      elif isinstance(item, torch.nn.Module):
        if self.device_id == -1:
          item.cpu()
        else:
          item.cuda(device=self.device_id)
      elif item is not None:
        print('[Warning] Invalid type {}'.format(item.__class__.__name__))

def set_devices(sys_device_ids):
  """
  It sets some GPUs to be visible and returns some wrappers to transferring 
  Variables/Tensors and Modules/Optimizers.
  Args:
    sys_device_ids: a tuple; which GPUs to use
      e.g.  sys_device_ids = (), only use cpu
            sys_device_ids = (3,), use the 4th gpu
            sys_device_ids = (0, 1, 2, 3,), use first 4 gpus
            sys_device_ids = (0, 2, 4,), use the 1st, 3rd and 5th gpus
  Returns:
    TVT: a `TransferVarTensor` callable
    TMO: a `TransferModulesOptims` callable
  """
  import os
  visible_devices = ''
  for i in sys_device_ids:
    visible_devices += '{}, '.format(i)
  os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
  device_id = 0 if len(sys_device_ids) > 0 else -1
  TVT = TransferVarTensor(device_id)
  TMO = TransferModulesOptims(device_id)
  return TVT, TMO

def transfer_optim_state(state, device_id=-1):
  """Transfer an optimizer.state to cpu or specified gpu, which means 
  transferring tensors of the optimizer.state to specified device. 
  The modification is in place for the state.
  Args:
    state: An torch.optim.Optimizer.state
    device_id: gpu id, or -1 which means transferring to cpu
  """
  for key, val in list(state.items()):
    if isinstance(val, dict):
      transfer_optim_state(val, device_id=device_id)
    elif isinstance(val, Variable):
      raise RuntimeError("Oops, state[{}] is a Variable!".format(key))
    elif isinstance(val, torch.nn.Parameter):
      raise RuntimeError("Oops, state[{}] is a Parameter!".format(key))
    else:
      try:
        if device_id == -1:
          state[key] = val.cpu()
        else:
          state[key] = val.cuda(device=device_id)
      except:
        pass



def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
  """Decay exponentially in the later phase of training. All parameters in the 
  optimizer share the same learning rate.
  
  Args:
    optimizer: a pytorch `Optimizer` object
    base_lr: starting learning rate
    ep: current epoch, ep >= 1
    total_ep: total number of epochs to train
    start_decay_at_ep: start decaying at the BEGINNING of this epoch
  
  Example:
    base_lr = 2e-4
    total_ep = 300
    start_decay_at_ep = 201
    It means the learning rate starts at 2e-4 and begins decaying after 200 
    epochs. And training stops after 300 epochs.
  
  NOTE: 
    It is meant to be called at the BEGINNING of an epoch.
  """
  assert ep >= 1, "Current epoch number should be >= 1"

  if ep < start_decay_at_ep:
    return

  for g in optimizer.param_groups:
    g['lr'] = (base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                    / (total_ep + 1 - start_decay_at_ep))))
  print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))



# 变量记录工具

def load_pickle(path):
  assert osp.exists(path)
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  return ret

def save_pickle(obj, path):
  dir = osp.dirname(osp.abspath(path))
  if not osp.exists(dir):
    os.makedirs(dir)
  with open(path, 'wb') as f:
    pickle.dump(obj, f)

def load_ckpt(modules_optims, ckpt_file, load_to_cpu=True, verbose=True):
  map_location = (lambda storage, loc: storage) if load_to_cpu else None
  ckpt = torch.load(ckpt_file, map_location=map_location)
  for m, sd in zip(modules_optims, ckpt['state_dicts']):
    m.load_state_dict(sd)
  if verbose:
    print('Resume from ckpt {}, \nepoch {}'.format(
      ckpt_file, ckpt['ep']))
  return ckpt['ep']

def save_ckpt(modules_optims, ep, scores, ckpt_file):
  state_dicts = [m.state_dict() for m in modules_optims]
  ckpt = dict(state_dicts=state_dicts, ep=ep)
  torch.save(ckpt, ckpt_file)


# 训练Log记录工具

class AverageMeter(object):
  """ Computes and stores the average and current value """
  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = float(self.sum) / (self.count + 1e-20)

class Log(object):
  def __init__(self, AMlist):
    """ 根据AMlist的要求定制AverageMeter
        Args:
          AMlist : 一个指标名称列表   eg. ['g_prec','g_d_ap','loss'] """
    self.allMeters = {}
    for item in AMlist:
      self.allMeters[item] = AverageMeter()
    
  def update(self, UPdict):
    """ 根据UPdict来更新对应的AveMeter
        Args:
          UPdict : 一个数据字典   eg. {'g_prec':0.21, 'loss':0.114} """
    for k, value in UPdict.items():
      self.allMeters[k].update(value)

  def printlog(self, ep, cost):
    """ TODO: 自定义打印每次ep的输出内容
        Args:
          ep : 当前的ep次数
          cost : time.time() - ep_st """
    time_log = 'Ep {}, {:.2f}s'.format(ep + 1, cost)
    AMlog = ''
    for k, value in self.allMeters.items():
      AMlog += ', {} {:.4f}'.format(k, value.avg)
    log = time_log + AMlog
    print(log)