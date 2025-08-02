import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # 3通道RGB图像
            nn.Conv2d(3,96,11,4,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96,256,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )


    def forward(self, x):
        return self.layers(x)