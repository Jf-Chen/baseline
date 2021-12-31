import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register
from .layer import MultiSpectralAttentionLayer
import pdb

@register('meta-baseline-dense')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={},temp=10., temp_learnable=True,
                    method='M2L_cos_dn4', neighbor_k=5, batch_size = 2, shot_num = 5, num_classes =5):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method
        #----------------------------------------------------------#
        self.neighbor_k = neighbor_k
        self.batch_size = batch_size
        self.shot_num = shot_num
        self.num_classes = num_classes
        
        
        self.Norm_layer = nn.BatchNorm1d(num_classes * 3, affine=True) # 只用了pix level 的dn4和cos
        self.FC_layer = nn.Conv1d(1, 1, kernel_size=3, stride=1, dilation=5, bias=False)
        
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        #                        groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

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
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0)) # [320, 640, 5, 5]
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        
        dimension=x_tot.size()[1] # 也许是512
        h = x_tot.size()[2] # 也许是7
        w = x_tot.size()[3]
        
        # x_shot = x_shot.view(*shot_shape,dimension,h ,w) # [4,5,1,640,5,5]
        # x_query = x_query.view(*query_shape, dimension,h ,w)
        
        #------------------
        x_shot_att=x_shot
        x_query_att=x_query
        
        x_shot_aft = x_shot_att.view(*shot_shape,dimension,h ,w) # [4,5,1,640,5,5]
        x_query_aft = x_query_att.view(*query_shape, dimension,h ,w) # [b,q_num,c,h,w]
        
        # avgpool
        # x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        x_shot_pool = x_shot_aft.view(x_shot_aft.shape[0], x_shot_aft.shape[1],x_shot_aft.shape[2],x_shot_aft.shape[3], -1).mean(dim=4) # [4, 5, 5, 640]
        x_query_pool = x_query_aft.view(x_query_aft.shape[0], x_query_aft.shape[1], x_query_aft.shape[2],-1).mean(dim=3) # [b,q_num,c,h*w]->[b,q_num,c] [4, 75, 640]
        #--------------------
          
        # print("x_shot",x_shot.size(),"x_query",x_query.size(),"x_shot_att",x_shot_att.size(),"x_query_att",x_query_att.size())
        if self.method == 'cos':
            x_shot_mean = x_shot_pool.mean(dim=-2)
            x_shot_F = F.normalize(x_shot_mean, dim=-1)
            x_query_F = F.normalize(x_query_pool, dim=-1)
            metric = 'dot'
            logits = utils.compute_logits(
                x_query_F, x_shot_F, metric=metric, temp=self.temp)
            
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'
            logits = utils.compute_logits(
                x_query_F, x_shot_F, metric=metric, temp=self.temp)
        
        elif self.method == "dense_match":
            b = query_shape[0]
            q_num=query_shape[1]
            way  = shot_shape[1]
            shot = shot_shape[2]
            
            self.batch_size = b
            self.shot_num = shot
            self.num_classes = way
            
            input1 =  x_query_aft.contiguous().permute(0,1,3,4,2) # [b,q_num,c,h,w]->[b,q_num,h,w,c]
            input2 = x_shot_aft.contiguous().permute(0,1,2,4,5,3)# [4,5,1,640,5,5]->[4,5,1,5,5,640] [b,way,shot,c,h,w] -> [b*shot*way,h*w,dimension]
            
            Similarity,Q_S_List = self.dense_match_similarity(input1 ,input2,self.neighbor_k)# query,support
            logits = Similarity
        
        return logits # [batch,q_num,way,]
        
#=================================  dense_match  ==================================#
    def dense_match_similarity(self,input1,input2,neighbor_k): # 仅计算local和global cos的sim
        # 经过校准，输入必须是 dim在最后一维
        # input1 [b,q_num,h,w,dimension]
        # input2 [b,shot,way,h,w,dimension]
        
        Similarity_list = []
        Q_S_List = []
        
        b,q_num,h,w,c = input1.size()
        _,shot,way,_,_,_ = input2.size()
        
        input1_batch = input1.contiguous().view(b, q_num,h*w,c )
        input2_batch = input2.contiguous().view(b,way*shot,h*w,c)
        #  input1_batch  [b,q_num,h*w,dimension]
        #  input2_batch  [b,way*shot,h*w,dimension]
        
        for i in range(b):
            input1 = input1_batch[i] # [q_num,h*w,dimension]
            input2 = input2_batch[i] # [way*shot,h*w,dimension]

            # L2 Normalization
            input1_norm = torch.norm(input1, 2, 2, True)
            input2_norm = torch.norm(input2, 2, 2, True)
            
            input1_after_norm=input1/input1_norm
            input2_after_norm=input2/input2_norm
            
            query = input1_after_norm.contiguous().view(q_num,h,w,c)
            support = input2_after_norm.contiguous().view(way,shot,h,w,c)
            
            support_set= support.contiguous().view(way,shot*h*w,c)
            query_set= query.contiguous().view(q_num,h*w,c)
            
            # query从support找到最相似的
            
            support_ex =  support_set.unsqueeze(0).expand(q_num,way,shot*h*w,c)
            query_ex = query_set.unsqueeze(1).expand(q_num,way,h*w,c)
            support_per = support_ex.permute(0,1,3,2) #[q_num,way,c,shot*h*w]
            
            innerproduct_matrix_local = torch.matmul(query_ex,support_per) #[q_num,way,h*w,shot*h*w]
            topk_value_local, topk_index_local = torch.topk(innerproduct_matrix_local, neighbor_k, 3)
            # [q_num,way,h*w,neighbor_k] [q_num,way,h*w,neighbor_k]
            
            # proto当然由最相似的组成，但是权重要看有多少个和他相似
            
            # support选中的local有25个，作为原型
            
            # 权重，query某个local和整个support way的相似度 占 query所有local和整个support way相似度的 占比
            #weight = torch.sum(innerproduct_matrix_local,3) #[q_num,way,h*w]
            #all_weight = torch.sum(weight,2)
            #weight_local = weight./
            # 先不加weight
            
            # ================ DN4的做法 =====================# 
            sim_local=torch.sum( topk_value_local,3)/neighbor_k # [q_num,way,h*w]
            sim_way = torch.sum(sim_local,2) # [q_num,way]
            
            Similarity_list.append(sim_way)
            
        Similarity=torch.stack(Similarity_list) #[4,75,5]
        
        return Similarity,Q_S_List
            
            
            
            
            