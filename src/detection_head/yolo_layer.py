import torch
import torch.nn as nn

class YoloLayer(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLayer, self).__init__()
        self.S = S
        self.B = B
        self.C = C

    def forward(self, x):
        N = x.size(0)
        x = x.view(N, self.S, self.S, self.B * 5 + self.C)
        return x
