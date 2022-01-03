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
            
            Similarity,Q_S_List = self.hyper_dense_both_similarity(input1 ,input2,self.neighbor_k)# query,support
            logits = Similarity
        
        return logits # [batch,q_num,way,]
        
#=========================  同时调整query和support,超参-》求和-》norm  ==============================#
    def hyper_dense_both_similarity(self,input1,input2,neighbor_k): # 仅计算local和global cos的sim
        # 经过校准，输入必须是 dim在最后一维
        # input1 [b,q_num,h,w,dimension]
        # input2 [b,shot,way,h,w,dimension]
        neighbor_k = 0.5
        
        Similarity_list = []
        Q_S_List = []
        
        b,q_num,h,w,c = input1.size()
        _,way,shot,_,_,_ = input2.size()
        
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
            
            # ================ 以support为基准，调整权重 =====================#
            # ================ 计算support set 自身权重和query到 support set的权重 =====================#
            # 只有这个bmm作为相似度，让人疑惑
            support_inner_matrix = torch.bmm(support_set,support_set.permute(0,2,1)) 
            
            # ======超参
            support_inner_matrix =  torch.pow(support_inner_matrix,neighbor_k)
            
            # [way,shot*h*w,shot*h*w]
            support_sum = support_inner_matrix.sum(dim=2) #[way,shot*h*w]
            support_sum_all = support_sum.sum(dim=1) + 1e-08# [way]
            support_sum_all_sq= support_sum_all.unsqueeze(dim=1) #[way,1]
            support_weight = (shot*h*w)*support_sum / support_sum_all_sq #[way,shot*h*w]
            
            # ======正则化
            support_weight_norm = torch.norm(support_weight,p=2,dim=1,keepdim=True)
            support_weight = support_weight/support_weight_norm
            
            support_set_proto = torch.mul(support_weight.unsqueeze(dim=2) ,support_set) # [way,shot*h*w,c]
            proto_pool = support_set_proto.sum(dim=1)/shot # [way,c]
            
            # 计算query
            # 以support 为基准
            # 要得到[q_num,way,h*w,shot*h*w]
            query_sq = query_set.unsqueeze(dim=1).expand(-1,way,-1,-1).contiguous().view(q_num*way,h*w,c)
            support_sq =  support_set.unsqueeze(dim=0).expand(q_num,-1,-1,-1).contiguous().view(q_num*way,shot*h*w,c)
            inter_matrix = torch.bmm(query_sq,support_sq.permute(0,2,1)) # [q_num*way,h*w,shot*h*w]
            
            # ======超参
            inter_matrix =  torch.pow(inter_matrix,neighbor_k)
            
            inter_sum =  inter_matrix.sum(dim= 2) # [q_num*way,h*w]
            inter_sum_all = inter_sum.sum(dim=1).unsqueeze(dim=1) # [q_num*way,1]
            query_weight = (h*w)* inter_sum / inter_sum_all # [q_num*way,h*w]
            query_weight_view = query_weight.contiguous().view(q_num,way,h*w)
            
            # ======正则化
            query_weight_norm = torch.norm(query_weight_view,p=2,dim=2,keepdim=True)
            query_weight_view = query_weight_view/query_weight_norm
            
            sim = torch.zeros(q_num,way).cuda()
            
            
            for j in range(way):
                query_way_weight =  query_weight_view[:,j,:].contiguous().view(q_num,h*w)
                query_way = torch.mul(query_set.contiguous().view(q_num*h*w,c),
                                query_way_weight.contiguous().view(q_num*h*w).unsqueeze(dim=1)
                                ).contiguous().view(q_num,h*w,c) #[q_num,h*w,c]
                query_way_pool = query_way.sum(dim=1) #[q_num,c]
                proto_way = proto_pool[j,:] #[c]
                sim_way = torch.mm(query_way_pool,proto_way.unsqueeze(dim=1)).view(q_num) # [q_num]
                sim[:,j]= sim_way
            
            Similarity_list.append(sim)
            
            
            
            
            
            
        Similarity=torch.stack(Similarity_list) #[4,75,5]
        
        # 这里在test时返回了[4,75,1],有点问题
        
        
        return Similarity,Q_S_List
        

        
#=================================  同时调整query和support  ==================================#
    def dense_both_similarity(self,input1,input2,neighbor_k): # 仅计算local和global cos的sim
        # 经过校准，输入必须是 dim在最后一维
        # input1 [b,q_num,h,w,dimension]
        # input2 [b,shot,way,h,w,dimension]
        neighbor_k = 0.1
        
        Similarity_list = []
        Q_S_List = []
        
        b,q_num,h,w,c = input1.size()
        _,way,shot,_,_,_ = input2.size()
        
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
            
            # ================ 以support为基准，调整权重 =====================#
            # ================ 计算support set 自身权重和query到 support set的权重 =====================#
            # 只有这个bmm作为相似度，让人疑惑
            support_inner_matrix = torch.bmm(support_set,support_set.permute(0,2,1)) 
            
            
            
            # [way,shot*h*w,shot*h*w]
            support_sum = support_inner_matrix.sum(dim=2) #[way,shot*h*w]
            support_sum_all = support_sum.sum(dim=1) + 1e-08# [way]
            support_sum_all_sq= support_sum_all.unsqueeze(dim=1) #[way,1]
            support_weight = (shot*h*w)*support_sum / support_sum_all_sq #[way,shot*h*w]
            
            # 超参
            support_weight =  torch.pow(support_weight,neighbor_k)
            
            
            #### 正则化
            support_weight_norm = torch.norm(support_weight,p=2,dim=1,keepdim=True)
            support_weight = support_weight/support_weight_norm
            
            support_set_proto = torch.mul(support_weight.unsqueeze(dim=2) ,support_set) # [way,shot*h*w,c]
            proto_pool = support_set_proto.sum(dim=1)/shot # [way,c]
            
            # 计算query
            # 以support 为基准
            # 要得到[q_num,way,h*w,shot*h*w]
            query_sq = query_set.unsqueeze(dim=1).expand(-1,way,-1,-1).contiguous().view(q_num*way,h*w,c)
            support_sq =  support_set.unsqueeze(dim=0).expand(q_num,-1,-1,-1).contiguous().view(q_num*way,shot*h*w,c)
            inter_matrix = torch.bmm(query_sq,support_sq.permute(0,2,1)) # [q_num*way,h*w,shot*h*w]
            
            
            
            
            
            inter_sum =  inter_matrix.sum(dim= 2) # [q_num*way,h*w]
            inter_sum_all = inter_sum.sum(dim=1).unsqueeze(dim=1) # [q_num*way,1]
            query_weight = (h*w)* inter_sum / inter_sum_all # [q_num*way,h*w]
            query_weight_view = query_weight.contiguous().view(q_num,way,h*w)
            
            # 超参
            query_weight_view =  torch.pow(query_weight_view,neighbor_k)
            
            # 正则化
            query_weight_norm = torch.norm(query_weight_view,p=2,dim=2,keepdim=True)
            query_weight_view = query_weight_view/query_weight_norm
            
            sim = torch.zeros(q_num,way).cuda()
            
            
            for j in range(way):
                query_way_weight =  query_weight_view[:,j,:].contiguous().view(q_num,h*w)
                query_way = torch.mul(query_set.contiguous().view(q_num*h*w,c),
                                query_way_weight.contiguous().view(q_num*h*w).unsqueeze(dim=1)
                                ).contiguous().view(q_num,h*w,c) #[q_num,h*w,c]
                query_way_pool = query_way.sum(dim=1) #[q_num,c]
                proto_way = proto_pool[j,:] #[c]
                sim_way = torch.mm(query_way_pool,proto_way.unsqueeze(dim=1)).view(q_num) # [q_num]
                sim[:,j]= sim_way
            
            Similarity_list.append(sim)
            
            
            
            
            
            
        Similarity=torch.stack(Similarity_list) #[4,75,5]
        
        # 这里在test时返回了[4,75,1],有点问题
        
        
        return Similarity,Q_S_List
        
        
#=================================  只调整query  ==================================#
    def dense_query_only_similarity(self,input1,input2,neighbor_k): # 仅计算local和global cos的sim
        # 经过校准，输入必须是 dim在最后一维
        # input1 [b,q_num,h,w,dimension]
        # input2 [b,shot,way,h,w,dimension]
        
        Similarity_list = []
        Q_S_List = []
        
        b,q_num,h,w,c = input1.size()
        _,way,shot,_,_,_ = input2.size()
        
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
            
            # ================ 计算query的自相关，调整权重 =====================#
            query_inner_matrix = torch.bmm(query_set,query_set.permute(0,2,1))#[q_num,h*w,h*w]
            query_sum = query_inner_matrix.sum(dim=2) #[q_num,h*w]
            query_sum_all = query_sum.sum(dim=1)
            query_sum_all_sq = query_sum_all.unsqueeze(dim=1)#[q_num,1]
            query_weight = (h*w)*query_sum/query_sum_all_sq #[q_num,h*w]
            
            query_after_weight = torch.mul(query_weight.unsqueeze(dim=2),query_set)
            query_pool = query_after_weight.sum(dim=1) #[q_num,c]
            
            proto_pool = support_set.sum(dim=1)/shot # [way,c]
            sim = torch.mm(query_pool,proto_pool.permute(1,0)) # [query,way] 
            Similarity_list.append(sim)
            
            
            
            
            
            
        Similarity=torch.stack(Similarity_list) #[4,75,5]
        
        # 这里在test时返回了[4,75,1],有点问题
        
        
        return Similarity,Q_S_List
            
        


#=================================  dense_match  ==================================#
    def dense_match_similarity(self,input1,input2,neighbor_k): # 仅计算local和global cos的sim
        # 经过校准，输入必须是 dim在最后一维
        # input1 [b,q_num,h,w,dimension]
        # input2 [b,shot,way,h,w,dimension]
        
        Similarity_list = []
        Q_S_List = []
        
        b,q_num,h,w,c = input1.size()
        _,way,shot,_,_,_ = input2.size()
        
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
            
            # ================ 计算support set 自身权重和query到 support set的权重 =====================#
            # 只有这个bmm作为相似度，让人疑惑
            support_inner_matrix = torch.bmm(support_set,support_set.permute(0,2,1)) 
            # [way,shot*h*w,shot*h*w]
            support_sum = support_inner_matrix.sum(dim=2) #[way,shot*h*w]
            support_sum_all = support_sum.sum(dim=1) + 1e-08# [way]
            support_sum_all_sq= support_sum_all .unsqueeze(dim=1) #[way,1]
            support_weight = support_sum / support_sum_all_sq #[way,shot*h*w]
            
            support_set_proto = torch.mul(support_weight.unsqueeze(dim=2) ,support_set) # [way,shot*h*w,c]
            proto_pool = support_set_proto.sum(dim=1)/shot # [way,c]
            
            # 计算query
            # 先不算，看看成果
            query_pool = query_set.sum(dim=1)
            sim = torch.mm(query_pool,proto_pool.permute(1,0)) # [query,way] 
            Similarity_list.append(sim)
            
            
            
        Similarity=torch.stack(Similarity_list) #[4,75,5]
        
        # 这里在test时返回了[4,75,1],有点问题
        
        
        return Similarity,Q_S_List
            
            
            
            
            