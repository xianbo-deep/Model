import torch
import torch.nn as nn

class FPN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
