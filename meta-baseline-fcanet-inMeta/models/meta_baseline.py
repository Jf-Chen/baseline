import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register
from .layer import MultiSpectralAttentionLayer
import pdb

@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method
        #----------------------------------------------------------#
        
        
        # 原本的 c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        # self.att = MultiSpectralAttentionLayer(plane * 4, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
        reduction = 16
        freq_sel_method = 'top16'
        c2wh = dict([(64,42),(160,21),(320,10),(640,5)])
        planes=640 # 插在哪一层后面就是多少维
        
        self.att = MultiSpectralAttentionLayer(channel = planes, dct_h=c2wh[planes], dct_w=c2wh[planes],  reduction=reduction, freq_sel_method = freq_sel_method)
        #--------------------------end-----------------------------#

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
        
        #------------------
        x_shot_att=self.att(x_shot)
        x_query_att=self.att(x_query)
        #--------------------
        pdb.set_trace()
        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'
        elif self.method == 'BSNet':
            # 采样BSNet度量，不过还没写
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        return logits

