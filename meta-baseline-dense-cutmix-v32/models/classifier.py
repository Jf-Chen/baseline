import math

import torch
import torch.nn as nn

import models
import utils
from .models import register
import pdb


@register('classifier')
class Classifier(nn.Module):
    
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)

    def forward(self, x):
        x = self.encoder(x) # [128,3,84,84]->[128,640,5,5]
        x = self.classifier(x) # [128,640,5,5]->[128,1,5,5]
        return x


@register('linear-classifier')
class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


@register('nn-classifier')
class NNClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, metric='cos', temp=None):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if temp is None:
            if metric == 'cos':
                temp = nn.Parameter(torch.tensor(10.))
            else:
                temp = 1.0
        self.metric = metric
        self.temp = temp

    def forward(self, x):
        return utils.compute_logits(x, self.proto, self.metric, self.temp)

@register('avgpool-linear-classifier')
class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        # 因为resnet中没池化
        
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2) #[batch,c,h,w]->[batch,c,hw]->[batch,c]
        return self.linear(x)
        
        
@register('dense-linear-classifier')
class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        
        # linear_dim =1
        #for i in range(len(in_dim)):
        #    linear_dim = linear_dim * in_dim[i]
        linear_dim  =  in_dim[0]

        self.n_classes = n_classes
        # self.linear = nn.Linear(linear_dim, n_classes*feature_h*feature_w)
        self.linear = nn.Linear(linear_dim, n_classes)
        # 不是64,而是[64,5,5]

    def forward(self, x):
        # x [b,c,h,w]
        feature = x.permute(0,2,3,1) # [b,h,w,c]
        logits = self.linear(feature) # [b,h,w,64]
        result  = logits.contiguous().view(-1,logits.size()[3])
        
        """
        batch  = x.shape[0]
        y =  x.contiguous().view(x.shape[0],-1)
        z = self.linear(y)
        logits = z.contiguous().view(batch,-1,self.feature_h,self.feathre_w)# [b,n_classes,h,w]
        
        logits = logits.permute(0,2,3,1).reshape(-1,self.n_classes*self.feature_h*self.feature_w)
        """
        
        return result  # [128*5*5,64]



