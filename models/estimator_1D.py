from typing import List

import torch
import torch.nn as nn

from models.base import BaseModule


class MaskedFullyConnection(BaseModule, nn.Linear):
    """
    Implements a Masked Fully Connection layer (MFC, Eq. 6).
    This is the autoregressive layer employed for the estimation of
    densities of image feature vectors.
    """
    def __init__(self, mask_type, in_channels, out_channels, *args, **kwargs):
        """
        Class constructor.

        :param mask_type: type of autoregressive layer, either `A` or `B`.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        """
        self.mask_type = mask_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(MaskedFullyConnection, self).__init__(*args, **kwargs)

        assert mask_type in ['A', 'B']
        self.register_buffer('mask', self.weight.data.clone())

        # Build mask
        self.mask.fill_(0)
        for f in range(0 if mask_type == 'B' else 1, self.out_features // self.out_channels):
            start_row = f*self.out_channels
            end_row = (f+1)*self.out_channels
            start_col = 0
            end_col = f*self.in_channels if mask_type == 'A' else (f+1)*self.in_channels
            if start_col != end_col:
                self.mask[start_row:end_row, start_col:end_col] = 1

        self.weight.mask = self.mask

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input tensor.
        :return: the output of a MFC manipulation.
        """

        # Reshape
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(len(x), -1)

        # Mask weights and call fully connection
        self.weight.data *= self.mask
        o = super(MaskedFullyConnection, self).forward(x)

        # Reshape again
        o = o.view(len(o), -1, self.out_channels)
        o = torch.transpose(o, 1, 2).contiguous()

        return o

    def __repr__(self):
        # type: () -> str
        """
        String representation.
        """
        return self.__class__.__name__ + '(' \
               + 'mask_type=' + str(self.mask_type) \
               + ', in_features=' + str(self.in_features // self.in_channels) \
               + ', out_features=' + str(self.out_features // self.out_channels)\
               + ', in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', n_params=' + str(self.n_parameters) + ')'


class Estimator1D(BaseModule):
    """
    Implements an estimator for 1-dimensional vectors.
    1-dimensional vectors arise from the encoding of images.
    This module is employed in MNIST and CIFAR10 LSA models.
    Takes as input a latent vector and outputs cpds for each variable.
    """
    def __init__(self, code_length, fm_list, cpd_channels):
        # type: (int, List[int], int) -> None
        """
        Class constructor.

        :param code_length: the dimensionality of latent vectors.
        :param fm_list: list of channels for each MFC layer.
        :param cpd_channels: number of bins in which the multinomial works.
        """
        super(Estimator1D, self).__init__()

        self.code_length = code_length
        self.fm_list = fm_list
        self.cpd_channels = cpd_channels

        activation_fn = nn.LeakyReLU()

        # Add autoregressive layers
        layers_list = []
        mask_type = 'A'
        fm_in = 1
        for l in range(0, len(fm_list)):

            fm_out = fm_list[l]
            layers_list.append(
                MaskedFullyConnection(mask_type=mask_type,
                                      in_features=fm_in * code_length,
                                      out_features=fm_out * code_length,
                                      in_channels=fm_in, out_channels=fm_out)
            )
            layers_list.append(activation_fn)

            mask_type = 'B'
            fm_in = fm_list[l]

        # Add final layer providing cpd params
        layers_list.append(
            MaskedFullyConnection(mask_type=mask_type,
                                  in_features=fm_in * code_length,
                                  out_features=cpd_channels * code_length,
                                  in_channels=fm_in,
                                  out_channels=cpd_channels))

        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of CPD estimates.
        """
        h = torch.unsqueeze(x, dim=1)  # add singleton channel dim
        h = self.layers(h)
        o = h

        return o
