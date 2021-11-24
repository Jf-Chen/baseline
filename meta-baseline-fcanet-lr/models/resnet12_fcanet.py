import torch.nn as nn
from .layer import MultiSpectralAttentionLayer

from .models import register


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)
        
        self.relu = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        #----------------------------------------------------------#
        # out = self.att(out)
        #--------------------------end-----------------------------#
        
        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out


class ResNet12_FcaNet(nn.Module):

    def __init__(self, channels, c2wh, planes, reduction, freq_sel_method):
        super().__init__()

        self.inplanes = 3

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])
        
        # 网络的卷积层的最后一层加入注意力机制
        #----------------------------------------------------------#
        reduction = 16
        freq_sel_method = 'top16'
        
        # 原本的 c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        c2wh = dict([(64,42),(160,21),(320,10),(640,5)])
        
        
        planes=640 # 插在哪一层后面就是多少维
        # self.att = MultiSpectralAttentionLayer(plane * 4, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
        self.att = MultiSpectralAttentionLayer(channel = planes, dct_h=c2wh[planes], dct_w=c2wh[planes],  reduction=reduction, freq_sel_method = freq_sel_method)
        #--------------------------end-----------------------------#

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
        
        # print("x",x.size())
        
        
        
        x = self.layer1(x)
        # print("x1",x.size())
        
        x = self.layer2(x)
        # print("x2",x.size())
        
        x = self.layer3(x)
        # print("x3",x.size())
        
        x = self.layer4(x)
        # print("x4",x.size())
        
        x = self.att(x)
        # print("xatt",x.size())
        
        
        # x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        
        # x=self.att(x)
        
        return x


@register('resnet12-fcanet')
def resnet12():
    # 需要修改对应的参数
    # [64, 128, 256, 512] 
    reduction = 16
    freq_sel_method = 'top16'
    c2wh = dict([(64,42),(160,21),(320,10),(640,5)])
    planes=640 # 插在哪一层后面就是多少维
    channels=[64, 160, 320, 640]
    return ResNet12_FcaNet(channels, c2wh, planes, reduction, freq_sel_method)


@register('resnet12-wide-fcanet')
def resnet12_wide_fcanet():
    reduction = 16
    freq_sel_method = 'top16'
    c2wh = dict([(64,42),(160,21),(320,10),(640,5)])
    planes=640 # 插在哪一层后面就是多少维
    channels=[64, 160, 320, 640]
    return ResNet12_FcaNet(channels, c2wh, planes, reduction, freq_sel_method)

