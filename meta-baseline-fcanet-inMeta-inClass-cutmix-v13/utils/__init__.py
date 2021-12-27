import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from . import few_shot


_log_path = None
_log_path_cutmix = None

def set_log_path(path):
    global _log_path
    _log_path = path

def set_log_path_cutmix(path):
    global _log_path_cutmix
    _log_path_cutmix = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def log_cutmix(obj, filename='log.txt'):
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

# 我写的，用loss计算acc
def compute_acc_loss(loss,label):
    ret = (torch.argmax(loss, dim=1) == label).float()
    return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(params, name, lr, beta1,step_size,gamma,scheduler_name,weight_decay=None, milestones=None,):
    if weight_decay is None:
        weight_decay = 0.
    if name == 'sgd':
        optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    elif name == 'adam-stand':
        # optimizer = Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        optimizer = Adam(params, lr=lr, betas=(beta1, 0.9), eps=1e-08, weight_decay=0, amsgrad=False)

    if scheduler_name =="StepLR":
        lr_scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        print("scheduler is StepLR")
    else:
        if milestones:
            lr_scheduler = MultiStepLR(optimizer, milestones,gamma=gamma)
            print("scheduler is MultiStepLR")
        else:
            lr_scheduler = None
    
    return optimizer, lr_scheduler


def visualize_dataset(dataset, name, writer, n_samples=16):
    demo = []
    for i in np.random.choice(len(dataset), n_samples):
        demo.append(dataset.convert_raw(dataset[i][0]))
    writer.add_images('visualize_' + name, torch.stack(demo))
    writer.flush()


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def rand_bbox(size, lam):
    # 对于classifier,size是[128,3,80,80]
    # 对于meta,size是[[2, 75, 3, 80, 80]
    # 为了减少修改，还是让meta适应rand_bbox
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_criterion(input,target,beta,cutmix_prob,model,criterion):
    r = np.random.rand(1)
    if beta > 0 and r < cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        # compute output
        output = model(input)
        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
    else:
        # compute output
        output = model(input)
        loss = criterion(output, target)
    
    err1, err5 = utils.accuracy(output.data, target, topk=(1, 5))
    
    return loss,err1, err5

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res
    

def cutmix_meta(support, query,label,beta,cutmix_prob,model,criterion):
    r = np.random.rand(1)
    if beta > 0 and r < cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(query.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(query.size(), lam)
        query[:, :, bbx1:bbx2, bby1:bby2] = query[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (query.size()[-1] * query.size()[-2]))
        # compute output
        output =  model(support, query)
        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
    else:
        # compute output
        output = model(input)
        loss = criterion(output, target)
    
    err1, err5 = utils.accuracy(output.data, target, topk=(1, 5))
    
    return loss,err1, err5
