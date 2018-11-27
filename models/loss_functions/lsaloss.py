import torch
import torch.nn as nn

from models.loss_functions.autoregression_loss import AutoregressionLoss
from models.loss_functions.reconstruction_loss import ReconstructionLoss


class LSALoss(nn.Module):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """
    def __init__(self, cpd_channels, lam=1):
        # type: (int, float) -> None
        """
        Class constructor.

        :param cpd_channels: number of bins in which the multinomial works.
        :param lam: weight of the autoregression loss.
        """
        super(LSALoss, self).__init__()

        self.cpd_channels = cpd_channels
        self.lam = lam

        # Set up loss modules
        self.reconstruction_loss_fn = ReconstructionLoss()
        self.autoregression_loss_fn = AutoregressionLoss(cpd_channels)

        # Numerical variables
        self.reconstruction_loss = None
        self.autoregression_loss = None
        self.total_loss = None

    def forward(self, x, x_r, z, z_dist):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        # Compute pytorch loss
        rec_loss = self.reconstruction_loss_fn(x, x_r)
        arg_loss = self.autoregression_loss_fn(z, z_dist)
        tot_loss = rec_loss + self.lam * arg_loss

        # Store numerical
        self.reconstruction_loss = rec_loss.item()
        self.autoregression_loss = arg_loss.item()
        self.total_loss = tot_loss.item()

        return tot_loss
