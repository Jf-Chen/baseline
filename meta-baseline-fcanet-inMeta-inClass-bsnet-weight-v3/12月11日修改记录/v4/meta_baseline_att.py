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
                 temp=10., temp_learnable=True, neighbor_k=5 ):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method
        #----------------------------------------------------------#
        self.neighbor_k = neighbor_k
        self.encoder_name = encoder
        # 初始化为0和1
        self.r_dn4 = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.r_cos = nn.Parameter(torch.ones(1),requires_grad=True)
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
        x_query_aft = x_query_att.view(*query_shape, dimension,h ,w)
        
        # avgpool
        # x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        x_shot_pool = x_shot_aft.view(x_shot_aft.shape[0], x_shot_aft.shape[1],x_shot_aft.shape[2],x_shot_aft.shape[3], -1).mean(dim=4) # [4,5,1,640]
        x_query_pool = x_query_aft.view(x_query_aft.shape[0], x_query_aft.shape[1], x_query_aft.shape[2],-1).mean(dim=3)
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
        #==================================================================#
        elif self.method == 'dn4cos':
            # 采样欧式+cos度量，需要输出[logits_dn4,logits_cos]
            # x_shot_aft [4, 5, 5, 640, 5, 5] 
            # x_query_aft [4, 75, 640, 5, 5]
            logits_cos , logits_dn4 = compute_dn4_cos_mix(x_shot_aft,x_query_aft,self.neighbor_k)
            logits = self.r_cos * logits_cos + self.r_dn4 * logits_dn4 
        #==================================================================#
        elif self.method == 'dn4':
            logits_dn4,logits = compute_cos_mix(x_shot_aft,x_query_aft,self.neighbor_k)
            logits = logits_dn4
        
        return logits

#========================== mix dn4 & cos  ==========================#
# 
# 混合dn4和cos,目标是输出两个logits
def compute_dn4_cos_mix(base,query,neighbor_k):
    # base [b,way,shot,c,h,w]
    # query [b,q_num,c,h,w]
    
    ## 余弦相似度(好像和proto一样)
    base_mean=base.contiguous().view(base.shape[0], base.shape[1],base.shape[2],base.shape[3], -1).mean(dim=4)# [b,way,shot,c]
    base_mean_proto=base_mean.mean(dim=2) # [b,way,c]
    query_mean=query.contiguous().view(query.shape[0], query.shape[1], query.shape[2],-1).mean(dim=3)
    
    # 加入正则化
    logits_cos = torch.bmm(F.normalize(query_mean, dim=-1), F.normalize(base_mean_proto, dim=-1).permute(0, 2, 1))
    
    # query_mean [4, 75, 640] base_mean [4, 5, 640]  
    # logits_cos [4, 75, 5]

    ## DN4相似度
    base_temp_1=base.view(base.shape[0], base.shape[1],base.shape[2],base.shape[3], -1)# [ep,way,shot,dim,h*w] [4, 5, 5, 640, 25]
    base_temp_2=base_temp_1.permute(0,1,2,4,3) # [ep,way,shot,h*w,dim]
    base_mix = base_temp_2.contiguous().view(base_temp_2.shape[0],base_temp_2.shape[1],base_temp_2.shape[2]*base_temp_2.shape[3],base_temp_2.shape[4])# [ep,way,shot*h*w,dim] [4,5,5*25,640]
    #  base_mix torch.Size 
    # 问题在这里,mix的过程错了，应当是[4,5,5*25,640]
    
    
    query_temp = query.view(query.shape[0], query.shape[1], query.shape[2],-1) # [4,75,5*5,640,] # [ep,q,h*w,dim]
    query_mix = query_temp.permute(0,1,3,2)
    
    # base_mix torch.Size([4, 5, 25, 640])
    # query_mix torch.Size([4, 75, 25, 640])
    
    batch=query_mix.size()[0]
    num_q=query_mix.size()[1]
    
    logits_dn4=[]
    
    for i in range(batch):
        Similarity_list = []
        for j in range(num_q):
            query_sam = query_mix[i,j,:,:] # [25, 640]
            query_sam_norm = torch.norm(query_sam, 2, 1, True)   
            query_sam = query_sam/query_sam_norm
            
            
            
            if torch.cuda.is_available():
                inner_sim = torch.zeros(1, base_mix.shape[1]).cuda()
            
            for k in range(base_mix.shape[1]):
                # 需要转置，但也许要在正则化前/后
                support_set_sam = base_mix[i,k,:,:]
                support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
                support_set_sam = support_set_sam/support_set_sam_norm
                support_set_sam_t = torch.transpose(support_set_sam,0,1)
                innerproduct_matrix = query_sam@support_set_sam_t
                
                topk_value, topk_index = torch.topk(innerproduct_matrix, neighbor_k, 1)
                inner_sim[0, k] = torch.sum(topk_value)/neighbor_k # 除以5试试
                
            Similarity_list.append(inner_sim)
        
        Similarity_list = torch.cat(Similarity_list, 0)  # 需要[batch,other]
        Similarity_list_per=torch.unsqueeze(Similarity_list,0)
        logits_dn4.append(Similarity_list_per)
    
    logits_dn4 = torch.cat(logits_dn4,0)
    ## logits 是 [4,75,5], 这个最后也要返回[4,75,5]

    
    # 只输出一个logits
    #logits_dn4_norm = torch.nn.functional.normalize(logits_dn4,p=2,dim=-1)
    #logits_cos_norm = torch.nn.functional.normalize(logits_cos,p=2,dim=-1)
    #logits = logits_dn4_norm + logits_cos_norm
    
    return logits_cos , logits_dn4



#========================== mix proto & cos ==========================#
# 
# 混合proto和dn4
def compute_cos_proto_mix(base,query,neighbor_k,r):
    # base [b,way,shot,c,h,w]
    # query [b,q_num,c,h,w]
    
    ## proto相似度
    base_mean=base.contiguous().view(base.shape[0], base.shape[1]*base.shape[2],base.shape[3], -1).mean(dim=3)# [b,way,shot,c]
    query_mean=query.contiguous().view(query.shape[0], query.shape[1], query.shape[2],-1).mean(dim=3)
    logits = torch.bmm(F.normalize(query_mean, dim=-1), F.normalize(base_mean, dim=-1).permute(0, 2, 1))

    ## dn4相似度
    base_temp_1=base.view(base.shape[0], base.shape[1],base.shape[2],base.shape[3], -1)# [ep,way*shot,dim,h*w]
    base_temp_2=base_temp_1 # [ep,way,shot,dim,h*w]
    base_mix = base_temp_2.contiguous().view(base_temp_2.shape[0],base_temp_2.shape[1],base_temp_2.shape[2]*base_temp_2.shape[3],base_temp_2.shape[4])# [ep,way,shot*h*w,dim] [4,5,1*25,640]
    
    query_temp = query.view(query.shape[0], query.shape[1], query.shape[2],-1) # [4,75,5*5,640,] # [ep,q,h*w,dim]
    query_mix = query_temp.permute(0,1,3,2)
    
    # base_mix torch.Size([4, 5, 25, 640])
    # query_mix torch.Size([4, 75, 25, 640])
    
    batch=query_mix.size()[0]
    num_q=query_mix.size()[1]
    
    logits_dn4=[]
    
    for i in range(batch):
        Similarity_list = []
        for j in range(num_q):
            query_sam = query_mix[i,j,:,:] # [25, 640]
            query_sam_norm = torch.norm(query_sam, 2, 1, True)   
            query_sam = query_sam/query_sam_norm
            
            
            
            if torch.cuda.is_available():
                inner_sim = torch.zeros(1, base_mix.shape[1]).cuda()
            
            for k in range(base_mix.shape[1]):
                support_set_sam = base_mix[i,k,:,:]
                support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
                support_set_sam = support_set_sam/support_set_sam_norm
                innerproduct_matrix = query_sam@support_set_sam
                
                topk_value, topk_index = torch.topk(innerproduct_matrix, neighbor_k, 1)
                inner_sim[0, k] = torch.sum(topk_value)/neighbor_k # 除以5试试
                
            Similarity_list.append(inner_sim)
        
        Similarity_list = torch.cat(Similarity_list, 0)  # 需要[batch,other]
        Similarity_list_per=torch.unsqueeze(Similarity_list,0)
        logits_dn4.append(Similarity_list_per)
    
    logits_dn4 = torch.cat(logits_dn4,0)
    ## logits 是 [4,75,5], 这个最后也要返回[4,75,5]
    result=r[0]*logits+r[1]*logits_dn4 # 但是二者的数值差距过大,详见debug
    
    
    return result



