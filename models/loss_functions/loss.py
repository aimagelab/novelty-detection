import torch.nn as nn
from models.loss_functions.reconstruction_loss import ReconstructionLoss
from models.loss_functions.autoregression_loss import AutoregressionLoss


class Loss(nn.Module):

    def __init__(self, cpd_channels: int):
        super(Loss, self).__init__()

        self.cpd_channels = cpd_channels

        self.reconstruction_loss_fn = ReconstructionLoss()
        self.autoregression_loss_fn = AutoregressionLoss(cpd_channels)

        self.reconstruction_loss = None
        self.autoregression_loss = None
        self.total_loss = None

    def forward(self, x, x_r, z, z_dist):

        rec_loss = self.reconstruction_loss_fn(x, x_r)
        arg_loss = self.autoregression_loss_fn(z, z_dist)

        self.reconstruction_loss = rec_loss.item()
        self.autoregression_loss = arg_loss.item()
        self.total_loss = self.reconstruction_loss + self.autoregression_loss

        return self.total_loss
