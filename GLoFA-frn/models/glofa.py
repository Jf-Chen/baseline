# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2021-04-06 19:31:25
"""

import torch
from torch import nn
from torch.nn import functional as F

from networks.set_function import SetFunction

class MyModel(nn.Module):
    def __init__(self, args, network):
        super(MyModel, self).__init__()
        self.args = args
        self.encoder = network
        if (args.network_name == 'resnet' && args.avg_pool==True):
            dimension = 640
        elif (args.network_name == 'resnet' && args.avg_pool==False):
            dimension = 640*25
        self.f_task = SetFunction(args, input_dimension=dimension, output_dimension=dimension)
        self.f_class = SetFunction(args, input_dimension=dimension, output_dimension=dimension)
        self.h = SetFunction(args, input_dimension=dimension, output_dimension=2)
    
    def forward(self, images):
        embeddings = self.encoder(images) 
        embeddings = embeddings.view(self.args.N * (self.args.K + self.args.Q), -1)

        support_embeddings = embeddings[:self.args.N * self.args.K, :]
        query_embeddings = embeddings[self.args.N * self.args.K:, :]

        mask_task = self.f_task(support_embeddings, level='task').unsqueeze(0)
        mask_class = self.f_class(support_embeddings, level='class').unsqueeze(0)

        alpha = self.h(support_embeddings, level='balance').squeeze(0)
        [alpha_task, alpha_class] = alpha

        masked_support_embeddings = support_embeddings.view(self.args.K, self.args.N, -1) * \
            (1 + mask_task * alpha_task) * (1 + mask_class * alpha_class)
        prototypes = torch.mean(masked_support_embeddings.view(self.args.K, self.args.N, -1), dim=0)
        prototypes = F.normalize(prototypes, dim=1, p=2)

        masked_query_embeddings = query_embeddings.unsqueeze(0).expand(self.args.N, -1, -1) * \
            (1 + mask_task * alpha_task) * (1 + mask_class.transpose(0, 1) * alpha_class)

        logits = torch.bmm(masked_query_embeddings, prototypes.t().unsqueeze(0).expand(self.args.N, -1, -1)) / self.args.tau
        x = torch.arange(self.args.N).long().cuda(self.args.devices[0])
        collapsed_logits = logits[x, :, x].t()

        return collapsed_logits
    
    def get_network_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j
    
    def get_other_params(self):
        modules = [self.f_task, self.f_class, self.h]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j
    
    
    def get_recon_dist(self,query,support,alpha,beta,Woodbury=True):
    # query: way*query_shot*resolution, d
    # support: way, shot*resolution , d
    # Woodbury: whether to use the Woodbury Identity as the implementation or not

        # correspond to kr/d in the paper
        reg = support.size(1)/support.size(2)
        
        # correspond to lambda in the paper
        lam = reg*alpha.exp()+1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0,2,1) # way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper
            
            sts = st.matmul(support) # way, d, d
            m_inv = (sts+torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse() # way, d, d
            hat = m_inv.matmul(sts) # way, d, d
        
        else:
            # correspond to Equation 8 in the paper
            
            sst = support.matmul(st) # way, shot*resolution, shot*resolution
            m_inv = (sst+torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse() # way, shot*resolution, shot*resolutionsf 
            hat = st.matmul(m_inv).matmul(support) # way, d, d

        Q_bar = query.matmul(hat).mul(rho) # way, way*query_shot*resolution, d

        dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way
        
        return dist
