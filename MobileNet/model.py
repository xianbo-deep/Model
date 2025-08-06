import torch
import torch.nn as nn


# 深度可分离卷积模块
def Depthiwise_Separable(in_channel,out_channel,stride,padding):
    return nn.Sequential(
        # 每个通道单独使用一个卷积核的精髓 并不改变输出通道数 groups设为输入通道数 代表一个输入通道为1组
        nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=stride,padding=padding,groups=in_channel,bias=False),
        nn.BatchNorm2d(in_channel),
        # 使用ReLU6作为激活函数，防止量化部署的时候造成精度损失
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self,num_classes=10,width = 1 ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.layers = nn.Sequential(
            Depthiwise_Separable(width*32,width*64,1,1),
            Depthiwise_Separable(width*64,width*128,2,1),
            Depthiwise_Separable(width*128,width*128,1,1),
            Depthiwise_Separable(width*128,width*256,2,1),
            Depthiwise_Separable(width*256,width*256,1,1),
            Depthiwise_Separable(width*256,width*512,2,1),
            Depthiwise_Separable(width*512,width*512,1,1),
            Depthiwise_Separable(width * 512, width * 512, 1, 1),
            Depthiwise_Separable(width * 512, width * 512, 1, 1),
            Depthiwise_Separable(width * 512, width * 512, 1, 1),
            Depthiwise_Separable(width * 512, width * 512, 1, 1),
            Depthiwise_Separable(width*512,width*1024,2,1),
            Depthiwise_Separable(width*1024,width*1024,2,4),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=7,stride=1),
            nn.Flatten(),
            nn.Linear(width*1024,num_classes),
        )



    def forward(self,x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.classifier(x)
        return x




