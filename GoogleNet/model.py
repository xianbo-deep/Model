'''
1. 提出了并联结构，使用Inception模块
具体来说就是用不同维度的卷积核进行卷积，最后输出的时候拼接起来
2. 使用了1*1卷积进行降维，降低参数大小
3. 使用辅助分类器，训练的时候把自身的损失乘以0.3加入总损失
4. 使用全局平均池化
'''
import torch.nn as nn
import torch


# 基本卷积模块


class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, **kwargs)
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)

        return x






# Inception模块
'''
ch1x1:1x1卷积的输出通道数
ch3x3red:1x1卷积的中间通道数，3x3卷积的输入通道数
ch3x3:3x3卷积的输出通道数
ch5x5red:1x1卷积的中间输出通道数，5x5卷积的输入通道数
ch5x5:5x5卷积的输出通道数
pool_proj:3x3池化+1x1卷积的输出通道数
'''
class Inception(nn.Module):
    def __init__(self, in_channel, ch1x1,ch3x3red,ch3x3,ch5x5red,ch5x5, pool_proj):
        super().__init__()
        self.branch1 = BasicConv2d(in_channel, ch1x1,kernel_size =1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, ch3x3red,kernel_size =1),
            BasicConv2d(ch3x3red, ch3x3,kernel_size =3,padding = 1),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, ch5x5red,kernel_size =1),
            BasicConv2d(ch5x5red, ch5x5,kernel_size =5,padding = 2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3,1,1,ceil_mode=True),
            BasicConv2d(in_channel, pool_proj,kernel_size =1),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        outputs = torch.cat((x1, x2, x3, x4), dim=1)
        return outputs


# 辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            BasicConv2d(in_channel, 128,kernel_size =1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,num_classes)
        )

    def forward(self, x):
        return self.layer(x)



class GoogLeNet(nn.Module):
    def __init__(self,aux_logits=True,num_classes=102,init_weights=True):
        super().__init__()
        self.aux_logits = aux_logits
        # Stage 1
        self.conv1= BasicConv2d(3,64,kernel_size =7,stride =2,padding = 3)
        self.maxpool1= nn.MaxPool2d(3,2,ceil_mode=True) # ceil_mode=True表示计算为小数时，向上取整
        # Stage 2
        self.conv2 = BasicConv2d(64,64,kernel_size =1)
        self.conv3 = BasicConv2d(64,192,kernel_size =3,padding = 1)
        self.maxpool2 = nn.MaxPool2d(3,2,ceil_mode=True)
        # Stage 3
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # Stage 4
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Stage 5
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)


        # 辅助分类器
        if self.aux_logits:
            self.aux1 = InceptionAux(512,num_classes)
            self.aux2 = InceptionAux(528,num_classes)


        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1024,num_classes)
        if init_weights:
            self._initialize_weights()



    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.aux_logits and self.training else None

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        aux2 = self.aux2(x) if self.aux_logits and self.training else None

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming初始化，更适合ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 对于线性层使用正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)