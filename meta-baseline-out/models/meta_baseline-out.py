import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register
import pdb

@register('meta-baseline-out')
class MetaBaselineOut(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos-DN4',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)
        
        pdb.set_trace()
        
        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'
        elif self.method == 'cos-DN4':
            # 特征逐个匹配，按照相似度排序
            # 取所有特征作为loss,但要记录每个特征的相似度
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'

        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        return logits
        


