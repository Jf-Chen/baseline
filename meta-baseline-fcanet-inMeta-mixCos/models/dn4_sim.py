import torch
import torch.nn as nn
from torch.nn import init
import functools
import pdb
import math
import sys



#========================== Define an image-to-class layer ==========================#
class ImgtoClass_Metric(nn.Module):
    def __init__(self, neighbor_k=3):
        super(ImgtoClass_Metric, self).__init__()
        self.neighbor_k = neighbor_k


    # Calculate the k-Nearest Neighbor of each local descriptor 
    def cal_cosinesimilarity(self, input1, input2):
        B, C, h, w = input1.size()
        Similarity_list = []
        
        B_s,C_s,h_s,w_s=input2.size()
        S = []
		for i in range(len(input2)):
			support_set_sam = self.features(input2[i])# b,c,h,w
			B, C, h, w = support_set_sam.size()
			support_set_sam = support_set_sam.permute(1, 0, 2, 3)# c,b,h,w
			support_set_sam = support_set_sam.contiguous().view(C, -1)
			S.append(support_set_sam) # c,hw*b
        input2=S

        for i in range(B):
            query_sam = input1[i] # C, h, w 
            query_sam = query_sam.view(C, -1) # C, hw 
            query_sam = torch.transpose(query_sam, 0, 1) # hw,c
            query_sam_norm = torch.norm(query_sam, 2, 1, True) # hw,c
            query_sam = query_sam/query_sam_norm # hw,c

            if torch.cuda.is_available():
                inner_sim = torch.zeros(1, len(input2)).cuda()

            for j in range(len(input2)):
                support_set_sam = input2[j] # C,h,w
                support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True) # 1, h,w
                support_set_sam = support_set_sam/support_set_sam_norm # C,h,w

                # cosine similarity between a query sample and a support category
                innerproduct_matrix = query_sam@support_set_sam

                # choose the top-k nearest neighbors
                topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
                inner_sim[0, j] = torch.sum(topk_value)

            Similarity_list.append(inner_sim)

        Similarity_list = torch.cat(Similarity_list, 0)    

        return Similarity_list 


    def forward(self, x1, x2):

        Similarity_list = self.cal_cosinesimilarity(x1, x2)

        return Similarity_list


#========================== mix  ==========================#
    # 
    # 混合proto和dn4
    def compute_cos_mix(self,base,query):
        # base [b,way,shot,c,h,w]
        # query [b,q_num,c,h,w]
        
        ## proto相似度
        base_mean=base.contiguous().view(base.shape[0], base.shape[1]*base.shape[2],base.shape[3], -1).mean(dim=3)# [b,way,shot,c]
        query_mean=query.query.contiguous().view(query.shape[0], query.shape[1], query.shape[2],-1).mean(dim=3)
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
        
        Similarity_list = []
        
        for i in range(batch):
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
                    
                    topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
                    inner_sim[0, j] = torch.sum(topk_value)
                    
            Similarity_list.append(inner_sim)
            
        Similarity_list = torch.cat(Similarity_list, 0)  # 需要[batch,other]
        
        