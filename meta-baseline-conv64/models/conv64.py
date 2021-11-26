import torch.nn as nn
import torch
from torch.nn import init
import functools
from .models import register

##############################################################################
# Classes: FourLayer_64F
##############################################################################

# Model: FourLayer_64F 
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer  
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21


class FourLayer_64F(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d,out_dim=):
		super(FourLayer_64F, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.features = nn.Sequential(                              # 3*84*84
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*42*42

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                                # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                                # 64*21*21
		)
		self.out_dim = 64
		



	def forward(self, x):

		
		y = self.features(x)
		return y
		
@register('conv-64')
def conv_64():
		return FourLayer_64F()
