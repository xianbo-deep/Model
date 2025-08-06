import torch.nn as nn
import torch


# downsample主要是调整shortcut连接的维度，防止网络输出和输入的维度不匹配
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        # 残差连接
        return self.relu(x + identity)




class Bottleneck(nn.Module):
    # 每个残差块的输出通道数是输入中间通道数的几倍
    expansion = 4
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.relu(x + identity)



class ResNet(nn.Module):
    # include_top表示是否包含完整的分类网络结构,即是否保留最后的线性层
    def __init__(self,block,block_num,num_classes=10,include_top = True):
        super().__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        # 开始构建残差块
        # channel 为block的中间通道数
        self.layer1 = self._make_layer(block,64,block_num[0])
        self.layer2 = self._make_layer(block,128,block_num[1],stride=2)
        self.layer3 = self._make_layer(block,256,block_num[2],stride=2)
        self.layer4 = self._make_layer(block,512,block_num[3],stride=2)

        # 是否保留最后的线性层
        if self.include_top is True:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512*block.expansion,num_classes)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top is True:
            x = self.avgpool(x)
            x = torch.flatten(x,1)
            x = self.fc(x)

        return x




    def _make_layer(self,block,channel,block_num,stride=1):
        downsample = None

        # 判断维度是否发生变化,此情况只会在每个layer的block1发生
        if stride != 1 or self.in_channel != channel*block.expansion:
            # 发生变化，需要进行维度缩放
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,channel*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channel*block.expansion),
            )

        layers = []
        layers.append(block(self.in_channel,channel,stride,downsample=downsample))
        # 下一个block的输入维度发生变化
        self.in_channel = channel*block.expansion

        for _ in range(1,block_num):
            layers.append(block(self.in_channel,channel))

        return nn.Sequential(*layers)



def resnet18(num_classes=10,include_top = True):
    return ResNet(BasicBlock,[2,2,2,2],num_classes,include_top)

def resnet34(num_classes=10,include_top = True):
    return ResNet(BasicBlock,[3,4,6,3],num_classes,include_top)

def resnet50(num_classes=10,include_top = True):
    return ResNet(Bottleneck,[3,4,6,3],num_classes,include_top)

def resnet101(num_classes=10,include_top = True):
    return ResNet(Bottleneck,[3,4,23,3],num_classes,include_top)

def resnet152(num_classes=10,include_top = True):
    return ResNet(Bottleneck,[3,8,36,3],num_classes,include_top)