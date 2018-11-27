import torch
import torch.nn as nn

from models.base import BaseModule


class MaskedConv3d(BaseModule, nn.Conv3d):
    """
    Implements a Masked Convolution 3D.
    This is a 3D Convolution that cannot access future frames.
    """
    def __init__(self, *args, **kwargs):
        super(MaskedConv3d, self).__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kT, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kT // 2 + 1:] = 0

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Performs the forward pass.

        :param x: the input tensor.
        :return: the output tensor as result of the convolution.
        """
        self.weight.data *= self.mask
        return super(MaskedConv3d, self).forward(x)
