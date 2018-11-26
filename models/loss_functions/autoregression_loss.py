import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressionLoss(nn.Module):

    def __init__(self, cpd_channels):
        super(AutoregressionLoss, self).__init__()

        self.cpd_channels = cpd_channels

        self.eps = np.finfo(float).eps
        self.pi = np.pi

    def forward(self, z, z_dist):

        z_d = z.detach()

        # Apply softmax
        z_dist = F.softmax(z_dist, dim=1)

        # Flatten out codes and distributions
        z_d = z_d.view(len(z_d), -1).contiguous()
        z_dist = z_dist.view(len(z_d), self.cpd_channels, -1).contiguous()

        # Log (regularized), pick the right ones
        z_dist = torch.clamp(z_dist, self.eps, 1 - self.eps)
        log_z_dist = torch.log(z_dist)
        index = torch.clamp(torch.unsqueeze(z_d, dim=1) * self.cpd_channels, min=0,
                            max=(self.cpd_channels - 1)).long()
        selected = torch.gather(log_z_dist, dim=1, index=index)
        selected = torch.squeeze(selected, dim=1)

        # Sum and mean
        S = torch.sum(selected, dim=-1)
        nll = - torch.mean(S)

        return nll
