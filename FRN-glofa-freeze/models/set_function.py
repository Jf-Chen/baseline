# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-08-07 14:07:17
"""

import torch
from torch import nn
from torch.nn import functional as F

# 看作是MLP
# 为了防止出bug,应该在forward的输入中添加train_shot和query_shot,而不是初始化后就一成不变
class SetFunction(nn.Module):
    def __init__(self, train_way,train_shot, resolution,input_dimension, output_dimension):
        super(SetFunction, self).__init__()# nn.Module.__init()__
        
        self.resolution=resolution
        self.train_way=train_way
        self.train_shot=train_shot
        self.resolution=resolution
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.psi = nn.Sequential(
            # nn.Linear() y=x*A'+b，A是weight，
            nn.Linear(input_dimension, input_dimension  * 2), 
            nn.ReLU(),
            nn.Linear(input_dimension * 2, input_dimension * 2),
            nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(input_dimension * 3, input_dimension * 2),
            nn.ReLU(),
            nn.Linear(input_dimension * 2, output_dimension),
        )

    def forward(self, support_embeddings, level,train_way,train_shot, resolution):
        if level == 'task':
            
            psi_output = self.psi(support_embeddings) #[
            
            rho_input = torch.cat([psi_output, support_embeddings], dim=2)
            rho_input = torch.sum(rho_input, dim=0, keepdim=True)
            
            rho_output = torch.nn.functional.relu6(self.rho(rho_input)) / 6
            
            
            return rho_output
        elif level == 'class':
            psi_output = self.psi(support_embeddings) 
            rho_input = torch.cat([psi_output, support_embeddings], dim=2)
            rho_input = rho_input.view(train_way, train_shot,resolution, -1)
            rho_input = torch.sum(rho_input, dim=1)
            
            
            
            rho_output = torch.nn.functional.relu6(self.rho(rho_input)) / 6 
            
            
            
            return rho_output
        # level = balance 不要了