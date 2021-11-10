import math

import torch
import torch.nn as nn

import models
import utils
from .models import register


@register('classifier')
class Classifier(nn.Module):
    
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim 
        classifier_args['resolution'] = 25
        self.classifier = models.make(classifier, **classifier_args)
        # encoder == resnet12 
        # classifier == linear-classifier
        # classifier_args {'n_classes': 64, 'in_dim': 512}
        
    def forward(self, x):
        # x [128, 3, 80, 80]
        x2 = self.encoder(x) # x2 [128, 512, 5, 5]
        x3 = self.classifier(x2) # x3 形状不匹配
        return x3


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

#---------------以下是我添加的--------------#
@register('linear-classifier_without_avgpool')
class LinearClassifier_without_avgpool(nn.Module):

    def __init__(self, resolution,in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(resolution*in_dim, n_classes)

    def forward(self, x):
        x1=x
        x2=x.view(x.size()[0],-1)
        x3=self.linear(x2)
        return self.linear(x3)
#----------------end------------------------#
