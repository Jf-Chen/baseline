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
