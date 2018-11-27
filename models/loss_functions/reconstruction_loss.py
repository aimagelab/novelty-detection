import torch

from models.base import BaseModule


class ReconstructionLoss(BaseModule):
    """
    Implements the reconstruction loss.
    """
    def __init__(self):
        # type: () -> None
        """
        Class constructor.
        """
        super(ReconstructionLoss, self).__init__()

    def forward(self, x, x_r):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :return: the mean reconstruction loss (averaged along the batch axis).
        """
        L = torch.pow((x - x_r), 2)

        while L.dim() > 1:
            L = torch.sum(L, dim=-1)

        return torch.mean(L)
