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
        
        fusion_num =2 
        if method == 'M2L_cos_dn4':
            fusion_num = 3
        elif method  == 'M2L_cos_match':
            fusion_num = 2
        
        self.Norm_layer = nn.BatchNorm1d(num_classes * fusion_num, affine=True) # 只用了pix level 的dn4和cos
        self.FC_layer = nn.Conv1d(1, 1, kernel_size = fusion_num, stride=1, dilation=5, bias=False)
        
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
        elif self.method == "M2L_cos_match":
            
            
            b = query_shape[0]
            q_num=query_shape[1]
            way  = shot_shape[1]
            shot = shot_shape[2]
            
            self.batch_size = b
            self.shot_num = shot
            self.num_classes = way
            
            input1 =  x_query_aft.contiguous().permute(0,1,3,4,2) # [b,q_num,c,h,w]->[b,q_num,h,w,c]
            input2 = x_shot_aft.contiguous().permute(0,1,2,4,5,3)# [4,5,1,640,5,5]->[4,5,1,5,5,640] [b,way,shot,c,h,w] -> [b*shot*way,h*w,dimension]
            
            Similarity,Q_S_List = self.match_similarity(input1 ,input2)# query,support
            logits = Similarity
        
        return logits # [batch,q_num,way]
        
#=================================  M2L_cos_match  ==================================#
    def match_similarity(self,input1_batch,input2_batch): # 仅计算local和global cos的sim
        # 经过校准，输入必须是 dim在最后一维
        # input1_batch [b*q_num,h*w,dimension]
        # input2_batch [b*shot*way,h*w,dimension]
        
        Similarity_list = []
        Q_S_List = []
        
        input1_batch = input1_batch.contiguous().view(self.batch_size, -1, input1_batch.size(1),
                                                      input1_batch.size(2))
        input2_batch = input2_batch.contiguous().view(self.batch_size, -1, input2_batch.size(1),
                                                      input2_batch.size(2))
        #  input1_batch  [b,q_num,h*w,dimension]
        #  input2_batch  [b,way*shot,h*w,dimension]
        for i in range(self.batch_size):
            input1 = input1_batch[i] # [q_num,h*w,dimension]
            input2 = input2_batch[i] # [way*shot,h*w,dimension]

            # L2 Normalization
            input1_norm = torch.norm(input1, 2, 2, True)
            input2_norm = torch.norm(input2, 2, 2, True)
            
            
            # ========================== global cos with proto ======================================# 
            ## 余弦相似度(好像和proto一样)
            dimension = input2.size()[2]
            HW = input2.size()[1]
            query = input1/input1_norm 
            
            base =  (input2/input2_norm).contiguous().view(self.num_classes,self.shot_num,HW,dimension) #[way,shot,h*w,dimension]
            
            base_mean=base.mean(dim=2) # [way,shot,c]
            base_mean_proto=base_mean.mean(dim=1) # [way,c]
            query_mean=query.mean(dim=1) #[q_num,c]
            
            logits_cos = torch.mm(query_mean, base_mean_proto.permute(1, 0)) # bmm 3d, mm 2d
            
            # logits_cos [75, 5]
            
            
            
            # ==========================local cos ,query to proto ======================================# 
            ######## 以query为锚 
            way = base.size()[0]
            q_num = query.size()[0]
            shot = base.size()[1]

            # Calculate the mean and covariance of the support set
            support_set = input2.contiguous().view(-1,self.shot_num * input2.size(1), input2.size(2)) # support_set [way,shot*h*w,d]
            
            # Calculate pixel and part -level similarity
            query_norm = input1 / input1_norm # # [q_num,h*w,d]
            support_norm = input2 / input2_norm # [way*shot,h*w,d]
            
            assert (torch.min(input1_norm) > 0) # 不懂这句话的含义，但有时会触发，代表什么情况？
            assert (torch.min(input2_norm) > 0)
            
            support_norm_l = support_norm.contiguous().view(-1,self.shot_num * support_norm.size(1),support_norm.size(2)) # [way,shot*h*w,d]

            # local level cosine similarity between a query set and a support set
            innerproduct_matrix_l = torch.matmul(query_norm.unsqueeze(1),support_norm_l.permute(0, 2, 1))
            # [q_num,1,h*w,d] * [way,d,h*w*shot] -> [q_num,way,h*w,h*w*shot]
            # ****注释：innerproduct_matrix_l 还可以理解，视为 query与way support 在 query的hw位置 与support中任意shot的任意位置 的相似度 ****#
            
            
            # choose the top-k nearest neighbors
            topk_value_l, topk_index_l = torch.topk(innerproduct_matrix_l, self.neighbor_k, 3)
            inner_sim_l = torch.sum(torch.sum(topk_value_l, 3), 2) # [75, 5]
            
            sim = topk_value_l.mean(dim=-1).mean(dim=-1) # [q_num, way]
            
            ######## 以proto为锚  每个shot都从query中挑选相似度的，组成相似度
            
            
            # 暂且只组合这两种
            wass_sim_soft2 = torch.cat((logits_cos,  sim), 1)
            wass_sim_soft2 = self.Norm_layer(wass_sim_soft2).unsqueeze(1)
            wass_sim_soft = self.FC_layer(wass_sim_soft2).squeeze(1) # [75, 5]

            Similarity_list.append(wass_sim_soft)
        
        Similarity=torch.stack(Similarity_list) #[4,75,5]
        
        return Similarity,Q_S_List








#================================= MML_cos_dn4 ==================================#
# 这里面必须要把dim放在最后一维，否则含义有误
# 需要重新调整
    def cal_MML_similarity(self, input1_batch, input2_batch):
        # 经过校准，输入必须是 dim在最后一维
        # input1_batch [b*q_num,h*w,dimension]
        # input2_batch [b*shot*way,h*w,dimension]
        
        Similarity_list = []
        Q_S_List = []
        input1_batch = input1_batch.contiguous().view(self.batch_size, -1, input1_batch.size(1),
                                                      input1_batch.size(2))
        input2_batch = input2_batch.contiguous().view(self.batch_size, -1, input2_batch.size(1),
                                                      input2_batch.size(2))
        #  input1_batch  [b,q_num,h*w,dimension]
        #  input2_batch  [b,way*shot,h*w,dimension]

        for i in range(self.batch_size):
            input1 = input1_batch[i] # [q_num,h*w,dimension]
            input2 = input2_batch[i] # [way*shot,h*w,dimension]

            # L2 Normalization
            input1_norm = torch.norm(input1, 2, 2, True) #  [q_num,h*w,1]
            input2_norm = torch.norm(input2, 2, 2, True) # [way*shot,h*w,1]
            
            
            # ================================================================# 
            ## 余弦相似度(好像和proto一样)
            dimension = input2.size()[2]
            HW = input2.size()[1]
            query = input1/input1_norm  # [q_num,h*w,dimension]
            
            base =  (input2/input2_norm).contiguous().view(self.num_classes,self.shot_num,HW,dimension) #[way,shot,h*w,dimension]
            
            base_mean=base.mean(dim=2) # [way,shot,c]
            
            print(base_mean.size())
            pdb.set_trace()
            
            base_mean_proto=base_mean.mean(dim=1) # [way,c]
            query_mean=query.mean(dim=1) #[q_num,c]
            
            logits_cos = torch.mm(query_mean, base_mean_proto.permute(1, 0)) # bmm 3d, mm 2d
            
            # logits_cos [75, 5]
            
            # ================================================================# 
            

            # Calculate the mean and covariance of the all the query images
            query_mean, query_cov = self.cal_covariance_matrix_Batch(
                input1)

            # Calculate the mean and covariance of the support set
            support_set = input2.contiguous().view(-1,
                                                   self.shot_num * input2.size(1), input2.size(2)) # support_set [way,shot*h*w,d]
            s_mean, s_cov = self.cal_covariance_matrix_Batch(support_set)

            # Find the remaining support set
            support_set_remain = self.support_remaining(support_set)
            s_remain_mean, s_remain_cov = self.cal_covariance_matrix_Batch(
                support_set_remain)

            # Calculate the Wasserstein Distance
            was_dis = -self.wasserstein_distance_raw_Batch(query_mean, query_cov, s_mean, s_cov)
            # Calculate pixel and part -level similarity
            query_norm = input1 / input1_norm # # [q_num,h*w,d]
            support_norm = input2 / input2_norm # [way*shot,h*w,d]
            # assert (torch.min(input1_norm) > 0) 不懂这句话的含义，但有时会触发，代表什么情况？
            # assert (torch.min(input2_norm) > 0)

            support_norm_p = support_norm.permute(0, 2, 1) # [way*shot,d,h*w]
            support_norm_p = support_norm_p.contiguous().view(-1,
                                                              self.shot_num * support_norm.size(2),
                                                              support_norm.size(1)) # [way,shot*d,h*w]

            support_norm_l = support_norm.contiguous().view(-1,
                                                            self.shot_num * support_norm.size(1),
                                                            support_norm.size(2)) # [way,shot*h*w,d]

            # local level cosine similarity between a query set and a support set
            innerproduct_matrix_l = torch.matmul(query_norm.unsqueeze(1),
                                                 support_norm_l.permute(0, 2, 1))
            # [q_num,1,h*w,d] * [way,d,h*w*shot] -> [q_num,way,h*w,h*w*shot]
            # ****注释：innerproduct_matrix_l 还可以理解，视为 query与way support 在 query的hw位置 与support中任意shot的任意位置 的相似度 ****#
            
            
            # 验证尺寸
            print("innerproduct_matrix_l",innerproduct_matrix_l.size())
            pdb.set_trace()
            
            
            innerproduct_matrix_p = torch.matmul(query_norm.permute(0, 2, 1).unsqueeze(1),
                                                 support_norm_p.permute(0, 2, 1))
            # [q_num,1,d,h*w] * [way,h*w,shot*d] -> [q_num,way,d,shot*d]
            # **** 注释： 这个就理解为 视为 query与way support 在 query的c通道 与support中任意shot的任意通道 的相似度 **** #

            # choose the top-k nearest neighbors
            topk_value_l, topk_index_l = torch.topk(innerproduct_matrix_l, self.neighbor_k, 3)
            inner_sim_l = torch.sum(torch.sum(topk_value_l, 3), 2) # [75, 5]

            topk_value_p, topk_index_p = torch.topk(innerproduct_matrix_p, self.neighbor_k, 3)
            inner_sim_p = torch.sum(torch.sum(topk_value_p, 3), 2) # [75, 5]

            # Using Fusion Layer to fuse three parts
            # wass_sim_soft2 = torch.cat((was_dis, inner_sim_l, inner_sim_p), 1)
            wass_sim_soft2 = torch.cat((logits_cos,  inner_sim_l, inner_sim_p), 1)
            wass_sim_soft2 = self.Norm_layer(wass_sim_soft2).unsqueeze(1)
            wass_sim_soft = self.FC_layer(wass_sim_soft2).squeeze(1) # [75, 5]

            Similarity_list.append(wass_sim_soft)

            """
            # Store the mean and covariance
            parser = argparse.ArgumentParser()
            Q_S = parser.parse_args()
            Q_S.query_mean = query_mean
            Q_S.query_cov = query_cov
            Q_S.s_mean = s_mean
            Q_S.s_cov = s_cov
            Q_S.s_remain_mean = s_remain_mean
            Q_S.s_remain_cov = s_remain_cov
            Q_S_List.append(Q_S)
            """
        
        Similarity=torch.stack(Similarity_list) #[4,75,5]
        return Similarity,Q_S_List

    def wasserstein_distance_raw_Batch(self, mean1, cov1, mean2, cov2):

        mean_diff = mean1 - mean2.squeeze(1)
        cov_diff = cov1.unsqueeze(1) - cov2
        l2_norm_mean = torch.div(torch.norm(mean_diff, p=2, dim=2), mean1.size(2))
        l2_norm_cova = torch.div(torch.norm(cov_diff, p=2, dim=(2, 3)), mean1.size(2) * mean1.size(2))

        return l2_norm_mean + l2_norm_cova

    def cal_covariance_matrix_Batch(self, feature):
        n_local_descriptor = torch.tensor(feature.size(1)).cuda()
        feature_mean = torch.mean(feature, 1, True)
        feature = feature - feature_mean
        cov_matrix = torch.matmul(feature.permute(0, 2, 1), feature)
        cov_matrix = torch.div(cov_matrix, n_local_descriptor - 1)
        cov_matrix = cov_matrix + 0.01 * torch.eye(cov_matrix.size(1)).cuda()

        return feature_mean, cov_matrix

    def support_remaining(self, S):

        S_new = []
        for ii in range(S.size(0)):
            indices = [j for j in range(S.size(0))]
            indices.pop(ii)
            indices = torch.tensor(indices).cuda()

            S_clone = S.clone()
            S_remain = torch.index_select(S_clone, 0, indices)
            S_remain = S_remain.contiguous().view(-1, S_remain.size(2))
            S_new.append(S_remain.unsqueeze(0))

        S_new = torch.cat(S_new, 0)
        return S_new