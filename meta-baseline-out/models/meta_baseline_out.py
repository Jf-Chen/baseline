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
        # n_way: 5 n_shot: 1 n_query: 15 train_batches: 200 ep_per_batch: 4
        # x_shot  [4, 5, 1, 3, 80, 80]   
        # x_query [4, 75, 3, 80, 80]
        #-------------------------------------------------------#
        
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]
        # shot_shape  [4, 5, 1]
        # query_shape [4, 75]
        # img_shape   [3, 80, 80]
        #-------------------------------------------------------#

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        # x_shot  [20, 640, 5, 5]) 
        # x_query [300, 640, 5, 5] 
        #-------------------------------------------------------#
        
        
        
        
        
        if self.method == 'cos':
            x_shot = x_shot.view(*shot_shape, -1)
            x_query = x_query.view(*query_shape, -1)
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
            logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        elif self.method == 'sqr':
            x_shot = x_shot.view(*shot_shape, -1)
            x_query = x_query.view(*query_shape, -1)
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'
            logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        elif self.method == 'cos-DN4':
            # 特征逐个匹配，按照相似度排序
            # 取所有特征作为loss,但要记录每个特征的相似度
            # 问题：正则化通常是给640还是25？——当前是640就给640
            ep_per_batch= 4
            width=5
            height=5
            dim=640
            resolution=width*height
            x_shot=x_shot.view(*shot_shape,dim,resolution) # [4, 5, 1, 640, 25]
            x_query=x_query.view(*query_shape,dim,resolution) # [4, 75, 640, 25]
            x_shot=F.normalize(x_shot,p=2,dim=-2)
            x_queryt=F.normalize(x_query,p=2,dim=-2)
            x_shot_class_level=x_shot.view([ep_per_batch,n_way*n_shot,dim,resolution]) # [4, 5, 640, 25]
            x_query_class_level=x_query # [4, 75, 640, 25]
            x_shot_dn=x_shot_class_level.permute([0,1,3,2]) # [4, 5, 25, 640]
            x_query_dn=x_query_class_level # [4, 75, 640, 25]

            x_shot_sq=torch.unsqueeze(x_shot_dn,1) # [4, 1, 5, 25, 640]
            x_query_sq=torch.unsqueeze(x_query_dn,2) # [4, 75, 1, 640, 25]
            # 需要 torch.norm
            innerproduct_matrix =x_shot_sq@x_query_sq # [4, 75, 5, 25, 25]
            neighbor_k = 1
            topk_value, topk_index = torch.topk(innerproduct_matrix, neighbor_k, -1) # [4, 75, 5, 25, 1],[4, 75, 5, 25, 1]
            # topk_index ,每个query的descriptor和一个support image最相似的descriptor_index
            descriptor_similarity=topk_value.squeeze(-1) # [4, 75, 5, 25]
            # 按照DN4的做法是取前5，然后平均
            simi_topk_value,simi_topk_index=torch.topk(descriptor_similarity,5,-1)
            simi_topk_sum=torch.sum(simi_topk_value,-1) # [4, 75, 5]
            #-------------end-------------------------#
            logits = simi_topk_sum
        
        return logits
        # logits [4, 75, 5]


