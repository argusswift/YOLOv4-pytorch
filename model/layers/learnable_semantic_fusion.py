import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# fusion features
class FusionLayer(nn.Module):
    def __init__(self, nums=6):
        super(FusionLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(nums))
        self.nums = nums
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        for i in range(self.nums):
            out = (
                self.weights[i] * x[i]
                if i == 0
                else out + self.weights[i] * x[i]
            )
        return out
