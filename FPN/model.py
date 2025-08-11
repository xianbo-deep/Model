import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, out_channels = 256):
        super().__init__()
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])
        c1_channels, c2_channels, c3_channels, c4_channels = 256,512,1024,2048

        self.conv1 = nn.Conv2d(c1_channels,out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(c2_channels, out_channels, kernel_size=1,stride=1, padding=0)
        self.conv3 = nn.Conv2d(c3_channels,out_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(c4_channels,out_channels, kernel_size=1, stride=1, padding=0)

        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        y2,y3,y4,y5 = self.backbone(x)
        y2 = self.conv1(y2)
        y3 = self.conv2(y3)
        y4 = self.conv3(y4)
        y5 = self.conv4(y5)

        # 上采样2倍
        y5 = F.interpolate(y5, scale_factor=2, mode='nearest')
        # 融合
        y4 = y5 + y4
        # 输出
        p4 = self.conv_out(y4)

        # 上采样2倍
        y4 = F.interpolate(y4, scale_factor=2, mode='nearest')
        # 融合
        y3 = y3 + y4
        # 输出
        p3 = self.conv_out(y3)

        # 上采样2倍
        y3 = F.interpolate(y3, scale_factor=2, mode='nearest')
        # 融合
        y2 = y3 + y2
        # 输出
        p2 = self.conv_out(y2)

        return p2,p3,p4



class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_channels,out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion , kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self,x):
        skip = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            skip = self.downsample(skip)
        return self.relu(x + skip)


class ResNet(nn.Module):
    def __init__(self,block,block_num):
        super().__init__()
        self.in_channels = 64
        self.layer1 = self._make_layer(block,64,block_num[0])
        self.layer2 = self._make_layer(block,128,block_num[1])
        self.layer3 = self._make_layer(block,256,block_num[2])
        self.layer4 = self._make_layer(block,512,block_num[3])
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_layer(self,block,channel,block_num,stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels,channel,stride=stride,downsample=downsample))

        self.in_channels = channel * block.expansion

        for _ in range(1,block_num):
            # 后续block的stride都为1
            layers.append(block(self.in_channels,channel,))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        skip1 = x
        x = self.layer2(x)
        skip2 = x
        x = self.layer3(x)
        skip3 = x
        x = self.layer4(x)
        return skip1,skip2,skip3,x