""" ResNet45
- Convolutional Block
- Res Block
"""
import math

import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,  stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.relu1 =nn.ReLU()
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):

    def __init__(self, ch_in, block, layers, rgb):
        self.inplanes = ch_in
        super(ResNet, self).__init__()
        first_ch=3 if rgb else 1
        self.conv1 = nn.Conv2d(first_ch, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        channels = [ch_in, ch_in * 2, ch_in * 4, ch_in * 8, ch_in * 16]
        """
        if isinstance(activation, nn.ReLU):
            self.relu = nn.ReLU(inplace=True)
        elif isinstance(activation, nn.Tanh):
            self.relu = nn.Tanh()
        elif isinstance(activation, nn.GELU):
            self.relu = nn.GELU()
        elif isinstance(activation, nn.LeakyReLU()):
            self.relu = nn.LeakyReLU()
        """
        ## Downsampling을 1번째, 3번째 layer에서만 수행한다.
        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=1)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block,  channels[3], layers[3], stride=1)
        self.layer5 = self._make_layer(block, channels[4], layers[4],  stride=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def resnet45(ch_in,rgb):
    return ResNet(ch_in, BasicBlock, [3, 4, 6, 6, 3], rgb)