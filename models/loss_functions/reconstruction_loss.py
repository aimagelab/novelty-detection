import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):

    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, x, x_r):

        L = torch.pow((x - x_r), 2)

        while L.dim() > 1:
            L = torch.sum(L, dim=-1)

        return torch.mean(L)
