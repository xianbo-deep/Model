import torch.nn as nn

class VGG(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),  # 经过一次池化层图像尺寸减半

            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),


            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),


            nn.Conv2d(256,512,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),


            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),


            nn.Flatten(),
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classes),
        )

    def forward(self, x):
        return self.layers(x)