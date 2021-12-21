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
        elif self.method == 'M2L_cos_dn4':
            b = query_shape[0]
            q_num=query_shape[1]
            way  = shot_shape[1]
            shot = shot_shape[2]
            
            self.batch_size = b
            self.shot_num = shot
            self.num_classes = way
            
            input1_batch =  x_query_aft.contiguous().view(b*q_num,dimension,h*w)
            input2_batch = x_shot_aft.contiguous().view(b*shot*way,dimension,h*w) # [b,way,shot,c,h,w] -> [b*shot*way,dimension,h*w]
            
            Similarity_list, Q_S_List = self.cal_MML_similarity(input1_batch,input2_batch) # query,support
            
            # 还不知道输出尺寸
            print(Similarity_list.size())
            
            print(type(Q_S_List))
            
            logits = Similarity_list # 
        
        
        return logits # [batch,q_num,way,]
        
#================================= MML_cos_dn4 ==================================#

    def cal_MML_similarity(self, input1_batch, input2_batch):
        # input1_batch [b*q_num,dimension,h*w]
        # input2_batch [b*shot*way,dimension,h*w]
        
        Similarity_list = []
        Q_S_List = []
        input1_batch = input1_batch.contiguous().view(self.batch_size, -1, input1_batch.size(1),
                                                      input1_batch.size(2))
        input2_batch = input2_batch.contiguous().view(self.batch_size, -1, input2_batch.size(1),
                                                      input2_batch.size(2))
        #  input1_batch  [b,q_num,dimension,h*w]
        #  input2_batch  [b,way*shot,dimension,h*w]

        for i in range(self.batch_size):
            input1 = input1_batch[i] # [q_num,dimension,h*w]
            input2 = input2_batch[i] # [way*shot,dimension,h*w]

            # L2 Normalization
            input1_norm = torch.norm(input1, 2, 2, True)
            input2_norm = torch.norm(input2, 2, 2, True)
            
            
            # ================================================================# 
            ## 余弦相似度(好像和proto一样)
            query = input1_norm 
            base =  input2_norm.contiguous().view(-1,self.shot_num,) #[way,shot,dimension,h*w]
            
            base_mean=base.mean(dim=3) # [way,shot,c]
            base_mean_proto=base_mean.mean(dim=1) # [way,c]
            query_mean=query.mean(dim=2) #[q_num,c]
            
            logits_cos = torch.bmm(query_mean, base_mean_proto.permute(0, 1))
            
            # logits_cos [75, 5]
            
            # ================================================================# 
            

            # Calculate the mean and covariance of the all the query images
            query_mean, query_cov = self.cal_covariance_matrix_Batch(
                input1)

            # Calculate the mean and covariance of the support set
            support_set = input2.contiguous().view(-1,
                                                   self.shot_num * input2.size(1), input2.size(2))
            s_mean, s_cov = self.cal_covariance_matrix_Batch(support_set)

            # Find the remaining support set
            support_set_remain = self.support_remaining(support_set)
            s_remain_mean, s_remain_cov = self.cal_covariance_matrix_Batch(
                support_set_remain)

            # Calculate the Wasserstein Distance
            was_dis = -self.wasserstein_distance_raw_Batch(query_mean, query_cov, s_mean, s_cov)
            # Calculate pixel and part -level similarity
            query_norm = input1 / input1_norm
            support_norm = input2 / input2_norm
            assert (torch.min(input1_norm) > 0)
            assert (torch.min(input2_norm) > 0)

            support_norm_p = support_norm.permute(0, 2, 1)
            support_norm_p = support_norm_p.contiguous().view(-1,
                                                              self.shot_num * support_norm.size(2),
                                                              support_norm.size(1))

            support_norm_l = support_norm.contiguous().view(-1,
                                                            self.shot_num * support_norm.size(1),
                                                            support_norm.size(2))

            # local level cosine similarity between a query set and a support set
            innerproduct_matrix_l = torch.matmul(query_norm.unsqueeze(1),
                                                 support_norm_l.permute(0, 2, 1))
            innerproduct_matrix_p = torch.matmul(query_norm.permute(0, 2, 1).unsqueeze(1),
                                                 support_norm_p.permute(0, 2, 1))

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
        return Similarity_list, Q_S_List

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