import torch.nn as nn

class Lenet(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层里面不用指定图像尺寸，指定输入输出通道数即可
        self.model = nn.Sequential(
            nn.Conv2d(1,6,5,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(16*4*4,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )


    def forward(self,x):
        return self.model(x)