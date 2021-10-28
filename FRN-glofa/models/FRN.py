import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbones import Conv_4,ResNet
from models.set_function import SetFunction



class FRN(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False,is_pretraining=False,num_cat=None,is_se=False,):
        
        super().__init__()
        

        if resnet:
            num_channel = 640
            self.feature_extractor = ResNet.resnet12()

        else:
            num_channel = 64
            self.feature_extractor = Conv_4.BackBone(num_channel)

        self.shots = shots
        self.way = way
        self.resnet = resnet

        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel
        
        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
        
        # H*W=5*5=25, resolution of feature map, correspond to r in the paper
        self.resolution = 25 

        # correpond to [alpha, beta] in the paper
        # if is during pre-training, we fix them to 0
        self.r = nn.Parameter(torch.zeros(2),requires_grad=not is_pretraining)
        #----------改成glofa形式---------
        dimension=640
        self.f_task = SetFunction(train_way=way,train_shot=shots[0], input_dimension=dimension, output_dimension=dimension,resolution=self.resolution)
        self.f_class = SetFunction(train_way=way,train_shot=shots[0], input_dimension=dimension, output_dimension=dimension,resolution=self.resolution)
        
        #------------end----------------
        
        
        if is_pretraining:
            # number of categories during pre-training
            self.num_cat = num_cat
            # category matrix, correspond to matrix M of section 3.6 in the paper
            self.cat_mat = nn.Parameter(torch.randn(self.num_cat,self.resolution,self.d),requires_grad=True)  

        #---------插入senet，更改了网络，需要重新进行预训练---------
        
    

    def get_feature_map(self,inp):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        
        if self.resnet:
            feature_map = feature_map/np.sqrt(640)
        
        return feature_map.view(batch_size,self.d,-1).permute(0,2,1).contiguous() # N,HW,C
        # return feature_map.view(batch_size,self.d,-1).contiguous()

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
        
        
    def get_recon_dist_glofa(self,query,support,alpha,beta,Woodbury=True):
    

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
            
            
        #---------唯一的变化就是Q_bar----------# 
        # 原本的query是          [(batch_size-train_way*train_shot)*resolution,d]
        # 现在的query是[train_way,(batch_size-train_way*train_shot)*resolution,d]
        # 需要做的是直接相乘
        # 需要输出dist=[1875, 5]
        # query [5,1875,640], hat [5,640,640]
        Q_bar = query.matmul(hat).mul(rho) # way, way*query_shot*resolution, d

        dist = (Q_bar-query).pow(2).sum(2).permute(1,0) # [1875, 5] way*query_shot*resolution, way
        #--------------end------------------#
        
        return dist


    
    def get_neg_l2_dist(self,inp,way,shot,query_shot,return_support=False):
        
        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]

        feature_map = self.get_feature_map(inp) # [100, 25, 640]
        
        
        
        # support = feature_map[:way*shot].view(way, shot*resolution , d) #[5, 125, 640]
        # query = feature_map[way*shot:].view(way*query_shot*resolution, d) #[1875, 640]
        
        #-------在这里插入glofa-------------
        train_way=way
        train_shot=shot
        support_embeddings = feature_map[:train_way*train_shot] # [25,25,640]
        query_embeddings = feature_map[train_way*train_shot:] # [75,25,640]
        mask_task=self.f_task(support_embeddings,level='task')
        mask_class=self.f_class(support_embeddings, level='class')
        
        
        # print("train_way: ",train_way,"train_shot",train_shot,"support_embeddings: ",support_embeddings.size(),"query_embeddings:",query_embeddings.size(),"mask_task:",mask_task.size(),"mask_class:",mask_class.size())
        # inp [75,3,84,84]
        # train_way:  5 train_shot 5 
        # support_embeddings:  torch.Size([25, 25, 640]) 
        # query_embeddings: torch.Size([50, 25, 640]) 
        # mask_task: torch.Size([1, 25, 640]) 
        # mask_class: torch.Size([5, 25, 640])
        
        masked_support_embeddings = support_embeddings.view(train_way,train_shot, resolution,-1) * \
            (1 + mask_task ) * (1 + mask_class )# [5, 5, 25, 640]
        masked_query_embeddings =query_embeddings.unsqueeze(0).expand(train_way, -1, -1,-1) * \
            (1 + mask_task) * (1 + mask_class.unsqueeze(0).transpose(0, 1) ) # [5, 75, 25, 640]
        
        support=masked_support_embeddings.view(train_way,train_shot*resolution,-1) # [5,125,640]
        query=masked_query_embeddings.view(train_way,-1,d) # [5,1875,640]
        """
        for i in range(0,train_way,1):
            query=masked_support_embeddings[i,:]
            query=query.view(way*query_shot*resolution, d)
            support=masked_support_embeddings.view(train_way,train_shot*resolution,d)
            recon_dist = self.get_recon_dist_glofa(query=masked_query_embeddings,support=masked_support_embeddings,alpha=alpha,beta=beta) # 这可能影响r的迭代
        """
        
        
        
        #----------end-----------------------------#
        recon_dist = self.get_recon_dist_glofa(query=query,support=support,alpha=alpha,beta=beta) # [3750, 10]
        neg_l2_dist = recon_dist.neg().view(way*query_shot,resolution,way).mean(1) # [150, 10]
        """
        recon_dist = self.get_recon_dist_glofa(query=query,support=support,alpha=alpha,beta=beta) # way*query_shot*resolution, way # [3750, 10]
        neg_l2_dist = recon_dist.neg().view(way*query_shot,resolution,way).mean(1) # way*query_shot, way # [150, 10]
        """
        # 其实这里是两步操作，第一步得到[way*query_shot,resolution,way],代表每个通道的特征图的相似度
        # 第二部计算每张图的相似性均值，也就是|Q_bar-Q|/r的操作
        
        
        if return_support:
            return neg_l2_dist, support
        else:
            return neg_l2_dist


    def meta_test(self,inp,way,shot,query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot)

        _,max_index = torch.max(neg_l2_dist,1)

        return max_index
    

    def forward_pretrain(self,inp):

        feature_map = self.get_feature_map(inp)
        batch_size = feature_map.size(0)

        feature_map = feature_map.view(batch_size*self.resolution,self.d)
        
        alpha = self.r[0]
        beta = self.r[1]
        
        recon_dist = self.get_recon_dist_glofa(query=feature_map,support=self.cat_mat,alpha=alpha,beta=beta) # way*query_shot*resolution, way

        neg_l2_dist = recon_dist.neg().view(batch_size,self.resolution,self.num_cat).mean(1) # batch_size,num_cat
        
        logits = neg_l2_dist*self.scale
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction


    def forward(self,inp):
        
        
        
        
        
        neg_l2_dist, support = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)
        
            
        logits = neg_l2_dist*self.scale
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction, support
