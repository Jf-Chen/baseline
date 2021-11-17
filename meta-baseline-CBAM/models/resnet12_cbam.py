import torch.nn as nn

from .models import register
from .attention import ChannelAttention,SpatialAttention


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

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out


class ResNet12_with_CBAM(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.inplanes = 3
        
        # 网络的第一层加入注意力机制
        # self.ca = ChannelAttention(channels[0])
        # self.sa = SpatialAttention()

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])
        
        # 网络的卷积层的最后一层加入注意力机制
        self.ca1 = ChannelAttention(channels[3])
        self.sa1 = SpatialAttention()
        
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
        
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        # x = self.maxpool(x)
        
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        
        
        # x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        
        return x


@register('resnet12-cbam')
def resnet12_cbam():
    return ResNet12_with_CBAM([64, 128, 256, 512])


@register('resnet12-wide-cbam')
def resnet12_wide_cbam():
    return ResNet12_with_CBAM([64, 160, 320, 640])

