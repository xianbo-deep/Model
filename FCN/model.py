import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = VGG()
        self.classifier = nn.Sequential(
            nn.Conv2d(512,4096,7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096,4096,1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096,num_classes,1),
        )
        # 跳跃连接做的变换，确保通道数相同
        self.skip3 = nn.Conv2d(256,num_classes,1)
        self.skip4 = nn.Conv2d(512,num_classes,1)
        # 上采样
        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        # 记录输出
        skip1 = x
        x = self.backbone.layer4(x)
        # 记录输出
        skip2 = x
        x = self.backbone.layer5(x)
        x = self.classifier(x)

        # 进行上采样
        x = self.upsample2x(x)
        # 保证通道数相同
        skip2 = self.skip4(skip2)
        # 融合
        x = x + skip2

        # 进行上采样
        x = self.upsample2x(x)
        skip1 = self.skip3(skip1)
        x = x + skip1

        # 进行八倍上采样，还原回原图大小
        output = self.upsample8x(x)

        return output


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),  # 经过一次池化层图像尺寸减半


        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )


        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )


