import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register
from .layer import MultiSpectralAttentionLayer

import pdb

@register('meta-baseline-att')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method
        #----------------------------------------------------------#
        
        self.encoder_name = encoder
        
        
        # 原本的 c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        # self.att = MultiSpectralAttentionLayer(plane * 4, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
        reduction = 16
        freq_sel_method = 'top16'
        c2wh = dict([(64,42),(160,21),(320,10),(640,5)])
        planes=640 # 插在哪一层后面就是多少维
        
        # self.att = MultiSpectralAttentionLayer(channel = planes, dct_h=c2wh[planes], dct_w=c2wh[planes],  reduction=reduction, freq_sel_method = freq_sel_method)
        # att期望的输入是 n,c,h,w 也就是同一个类别
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
        x_shot_att=x_shot#x_shot_att=self.att(x_shot)
        x_query_att=x_query#x_query_att=self.att(x_query)
        
        x_shot_aft = x_shot_att.view(*shot_shape,dimension,h ,w) # [4,5,1,640,5,5]
        x_query_aft = x_query_att.view(*query_shape, dimension,h ,w)
        
        # avgpool
        # x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        x_shot_pool = x_shot_aft.view(x_shot_aft.shape[0], x_shot_aft.shape[1],x_shot_aft.shape[2],x_shot_aft.shape[3], -1).mean(dim=4)
        x_query_pool = x_query_aft.view(x_query_aft.shape[0], x_query_aft.shape[1], x_query_aft.shape[2],-1).mean(dim=3)
        #--------------------
        
        # print("x_shot",x_shot.size(),"x_query",x_query.size(),"x_shot_att",x_shot_att.size(),"x_query_att",x_query_att.size())
        if self.method == 'cos':
            
            x_shot_mean = x_shot_pool.mean(dim=-2)
            x_shot_F = F.normalize(x_shot_mean, dim=-1)
            x_query_F = F.normalize(x_query_pool, dim=-1)
            metric = 'dot'
            
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'
        
        elif self.method =='dbrn_bias':
            b = x_shot_aft.size()[0];
            way = x_shot_aft.size()[1]
            shot  = x_shot_aft.size()[2]
            c = x_shot_aft.size()[3]
            h = x_shot_aft.size()[4]
            w = x_shot_aft.size()[5]
            support = x_shot_aft.view(b,way,shot,c,h*w)
            support  = support.permute(0,1,2,4,3) # [b,way,shot,h*w,c]
            x_shot_mean = get_bias_proto(support,h,w) # [b,way,c]
            x_shot_F = F.normalize(x_shot_mean, dim=-1)
            x_query_F = F.normalize(x_query_pool, dim=-1)
            metric = 'dot'
            logits = utils.compute_logits(
                x_query_F, x_shot_F, metric=metric, temp=self.temp)
            return logits
            
        logits = utils.compute_logits(
                x_query_F, x_shot_F, metric=metric, temp=self.temp)
        return logits

# ======================= 借鉴dbrn的原型生成方式 ==================== #
def get_bias_proto(support): # [b,way,shot,h*w,c] -> [b,way,c]
    b = support.size()[0]
    way = support.size()[1]
    shot  = support.size()[2]
    hw = support.size()[3]
    c = support.size()[4]
    
    Pe = support.mean(dim=2) # [b,way,h*w,c]
    
    Pe_w =  Pe.view(b*way,h*w,c)
    sim = torch.nn.functional.cosine_similarity(Pe_w,Pe_w,dim =-1) # [b*way,h*w]
    # 求出来全是1 写的不对
    bias = sim.view(b,way,h*w)
    
    bias_index  = torch.argsort(bias,descending=True) # 降序的bias的index # (b,way,h*w)
    # torch.argsort
    
    # 整理,获得sim,每张support对伪Proto,每个块的相似度
    Pe_sq=torch.unsqueeze(Pe,dim = 2) # Pe_sq torch.Size([4, 5, 1, 25, 640]) 
    Pe_expand = Pe_sq.expand(b,way,shot,h*w,c) # Pe_expand torch.Size([4, 5, 5, 25, 640])
    support_w = support.reshape(b*way*shot,h*w,c)  # support_w torch.Size([100, 25, 640])
    Pe_expand_w =  Pe_expand.reshape(b*way*shot,h*w,c) # Pe_expand_w torch.Size([100, 25, 640])
    sim_expand = torch.nn.functional.cosine_similarity(Pe_expand_w,support_w,dim =-1) #  sim_expand torch.Size([100, 25])
    sim = sim_expand.reshape(b,way,shot,h*w) # sim torch.Size([4, 5, 5, 25])
    sim = F.normalize(sim, dim=-1)+1.
    
    support_new = torch.ones([b,way,shot,h*w,c],device=torch.device('cuda:0'))
    
    device = torch.device('cuda:0') 
    for i in range(b):
        for j in range(way):
            curr_index = bias_index[i,j,:]
            for k in range(shot):
                mask = torch.ones(h*w,device=torch.device('cuda:0'))
                curr_way_Pe_bias = bias[i,j,:] #[h*w]
                curr_way_index = bias_index[i,j,:] #[h*w]
                
                curr_shot_sim = sim[i,j,k,:] #[h*w]
                
                curr_img = support[i,j,k,:,:] # [h*w,c]
                for s in range(h*w):
                    block = curr_way_index[s] #块按Pe中重要顺序
                    # 取出和该shot余下块中和该Pe块bias最高的块
                    masked_sim = curr_shot_sim.mul(mask)
                    max_index = torch.argmax(masked_sim, dim=0)
                    
                    mask[max_index] = 0
                    support_new[i,j,k,block,:]= support_new[i,j,k,max_index,:] # 填充到block位置
                    
    
    support_shot_mean = support_new.mean(dim=3)
    proto =  support_shot_mean.mean(dim=2)
    return proto
    
    # ====改成矩阵形式
    #mask = torch.ones(b,way,shot,h*w,device=torch.device('cuda:0')) #[b,way,shot,h*w]
    #for s in range(h*w):
    #    block = bias_index[:,:,s] # [b,way]
    #    masked_sim = sim.mul(mask) # [b,way, shot, h*w] * [b,way,shot,h*w]
    #    max_index = torch.argmax(masked_sim, dim=-1) # [b,way, shot]
    #    max_index_sq = torch.unsqueeze(max_idnex,-1) # [b,way, shot,1]
    #    
    # 不知道如何赋值
    
    
    
    

    