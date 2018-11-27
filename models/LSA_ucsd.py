from typing import Tuple

import torch
import torch.nn as nn

from models.base import BaseModule
from models.blocks_3d import DownsampleBlock
from models.blocks_3d import UpsampleBlock
from models.estimator_2D import Estimator2D
from models.layers.tsc import TemporallySharedFullyConnection


class Encoder(BaseModule):
    """
    UCSD Ped2 model encoder.
    """
    def __init__(self, input_shape, code_length):
        # type: (Tuple[int, int, int, int], int) -> None
        """
        Class constructor:

        :param input_shape: the shape of UCSD Ped2 samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        c, t, h, w = input_shape

        activation_fn = nn.LeakyReLU()

        # Convolutional network
        self.conv = nn.Sequential(
            DownsampleBlock(channel_in=c, channel_out=8, activation_fn=activation_fn, stride=(1, 2, 2)),
            DownsampleBlock(channel_in=8, channel_out=12, activation_fn=activation_fn, stride=(2, 1, 1)),
            DownsampleBlock(channel_in=12, channel_out=18, activation_fn=activation_fn, stride=(1, 2, 2)),
            DownsampleBlock(channel_in=18, channel_out=27, activation_fn=activation_fn, stride=(2, 1, 1)),
            DownsampleBlock(channel_in=27, channel_out=40, activation_fn=activation_fn, stride=(1, 2, 2))
        )

        self.deepest_shape = (40, t // 4, h // 8, w // 8)

        # FC network
        dc, dt, dh, dw = self.deepest_shape
        self.tdl = nn.Sequential(
            TemporallySharedFullyConnection(in_features=dc * dh * dw, out_features=code_length),
            nn.Sigmoid()
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input batch of patches.
        :return: the batch of latent vectors.
        """
        h = x
        h = self.conv(h)

        # Reshape for fully connected sub-network (flatten)
        c, t, height, width = self.deepest_shape
        h = torch.transpose(h, 1, 2).contiguous()
        h = h.view(-1, t, (c * height * width))
        o = self.tdl(h)

        return o


class Decoder(BaseModule):
    """
    UCSD Ped2 model decoder.
    """
    def __init__(self, code_length, deepest_shape, output_shape):
        # type: (int, Tuple[int, int, int, int], Tuple[int, int, int, int]) -> None
        """
        Class constructor.

        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of UCSD Ped2 samples.
        """
        super(Decoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        dc, dt, dh, dw = deepest_shape

        activation_fn = nn.LeakyReLU()

        # FC network
        self.tdl = nn.Sequential(
            TemporallySharedFullyConnection(in_features=code_length, out_features=(dc * dh * dw)),
            activation_fn
        )

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=dc, channel_out=40, activation_fn=activation_fn,
                          stride=(1, 2, 2), output_padding=(0, 1, 1)),
            UpsampleBlock(channel_in=40, channel_out=27, activation_fn=activation_fn,
                          stride=(1, 2, 2), output_padding=(0, 1, 1)),
            UpsampleBlock(channel_in=27, channel_out=18, activation_fn=activation_fn,
                          stride=(2, 1, 1), output_padding=(1, 0, 0)),
            UpsampleBlock(channel_in=18, channel_out=12, activation_fn=activation_fn,
                          stride=(1, 2, 2), output_padding=(0, 1, 1)),
            UpsampleBlock(channel_in=12, channel_out=8, activation_fn=activation_fn,
                          stride=(2, 1, 1), output_padding=(1, 0, 0)),
            nn.Conv3d(in_channels=8, out_channels=output_shape[0], kernel_size=1, bias=False)
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x
        h = self.tdl(h)

        # Reshape to encoder's deepest convolutional shape
        h = torch.transpose(h, 1, 2).contiguous()
        h = h.view(len(h), *self.deepest_shape)

        h = self.conv(h)
        o = h

        return o


class LSAUCSD(BaseModule):
    """
    LSA model for UCSD Ped2 video anomaly detection.
    """
    def __init__(self, input_shape, code_length, cpd_channels):
        # type: (Tuple[int, int, int, int], int, int) -> None
        """
        Class constructor.

        :param input_shape: the shape of UCSD Ped2 samples.
        :param code_length: the dimensionality of latent vectors.
        :param cpd_channels: number of bins in which the multinomial works.
        """
        super(LSAUCSD, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.cpd_channels = cpd_channels

        # Build encoder
        self.encoder = Encoder(
            input_shape=input_shape,
            code_length=code_length
        )

        # Build decoder
        self.decoder = Decoder(
            code_length=code_length,
            deepest_shape=self.encoder.deepest_shape,
            output_shape=input_shape
        )

        # Build estimator
        self.estimator = Estimator2D(
            code_length=code_length,
            fm_list=[4, 4, 4, 4],
            cpd_channels=cpd_channels
        )

    def forward(self, x):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        Forward propagation.

        :param x: the input batch of patches.
        :return: a tuple of torch.Tensors holding reconstructions, latent vectors and CPD estimates.
        """
        h = x

        # Produce representations
        z = self.encoder(h)

        # Estimate CPDs with autoregression
        z_dist = self.estimator(z)

        # Reconstruct x
        x_r = self.decoder(z)
        x_r = x_r.view(-1, *self.input_shape)

        return x_r, z, z_dist
