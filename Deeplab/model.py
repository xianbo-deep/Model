import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=12,
                      dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=24,dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=36,dilation=18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, stride=1,)
            ,nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        b,c,h,w = x.shape
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        # 进行插值，统一尺寸
        x5 = F.interpolate(x5,size=(h,w), mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.head(x)
        return x

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self,in_channels,out_channels,stride = 1,downsample = None,dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=dilation,dilation=dilation)
        self.conv3 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels * self.expansion,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self,x):
        skip = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            skip = self.downsample(skip)
        return self.relu(x + skip)

class ResNet(nn.Module):
    def __init__(self,block,block_num):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,64,block_num[0])
        self.layer2 = self._make_layer(block,128,block_num[1],stride=2)
        self.layer3 = self._make_layer(block,256,block_num[2],stride=1,dilation=2)
        self.layer4 = self._make_layer(block,512,block_num[3],stride=1,dilation=4)


    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _make_layer(self,block,channel,block_num,stride=1,dilation=1):
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
            layers.append(block(self.in_channels,channel,dilation=dilation))
        return nn.Sequential(*layers)


class DeepLab(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet(BottleNeck,[3, 4, 23, 3])
        self.aspp = ASPP(in_channels=2048,out_channels=256)
        self.classifier = nn.Conv2d(in_channels=256, out_channels=1,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.classifier(x)
        # 上采样回原图,这里就是上采样8倍
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        return x