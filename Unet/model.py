import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_channels,out_channels,max_pool=True):
        super(DownBlock, self).__init__()
        self.max_pool = max_pool
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if self.max_pool:
            self.pool = nn.MaxPool2d(kernel_size=2)


    def forward(self, x):
        x = self.conv(x)
        skip = x
        if self.pool:
             x = self.pool(x)
        return x,skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv1(x)

        x = torch.cat((x, skip), dim=1)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        return x





class Unet(nn.Module):
    def __init__(self,in_channels = 1,out_channels = 64,num_classes = 2):
        super().__init__()
        self.down1 = DownBlock(in_channels, out_channels)
        self.down2 = DownBlock(out_channels, out_channels * 2)
        self.down3 = DownBlock(out_channels * 2, out_channels * 4)
        self.down4 = DownBlock(out_channels * 4, out_channels * 8)
        self.down5 = DownBlock(out_channels * 8, out_channels * 16,max_pool=False)




        self.up1 = UpBlock(out_channels * 16, out_channels * 8)
        self.up2 = UpBlock(out_channels * 8, out_channels * 4)
        self.up3 = UpBlock(out_channels * 4, out_channels * 2)
        self.up4 = UpBlock(out_channels * 2, out_channels)

        self.out = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    # 对skip进行裁剪
    def crop_tensor(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        # 如果原始张量的尺寸为10，而delta为2，那么"delta:tensor_size - delta"将截取从索引2到索引8的部分，长度为6，以使得截取后的张量尺寸变为6。
        return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


    def forward(self, x):
        x1,skip1 = self.down1(x)
        x2,skip2 = self.down2(x1)
        x3,skip3 = self.down3(x2)
        x4,skip4 = self.down4(x3)
        x5,_ = self.down5(x4)

        skip1 = self.crop_tensor(skip1,x1)
        skip2 = self.crop_tensor(skip2,x2)
        skip3 = self.crop_tensor(skip3,x3)
        skip4 = self.crop_tensor(skip4,x4)


        x = self.up1(x5,skip4)
        x = self.up2(x,skip3)
        x = self.up3(x,skip2)
        x = self.up4(x,skip1)


        x = self.out(x)

        return x
