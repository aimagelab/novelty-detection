from functools import reduce
from operator import mul

import torch
import torch.nn as nn

from models.base import BaseModule
from models.blocks_2d import DownsampleBlock
from models.blocks_2d import UpsampleBlock
from models.estimator_1D import Estimator1D


class Encoder(BaseModule):
    def __init__(self, input_shape: tuple, code_length: int):
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        c, h, w = input_shape

        activation_fn = nn.LeakyReLU()

        self.conv = nn.Sequential(
            DownsampleBlock(channel_in=c, channel_out=32, activation_fn=activation_fn),
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation_fn),
        )
        self.deepest_shape = (64, h // 4, w // 4)

        fc_layers = [
            nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=code_length),
            nn.Sigmoid()
        ]

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.FloatTensor):

        h = x
        h = self.conv(h)
        h = h.view(len(h), -1)
        o = self.fc(h)

        return o


class ANDDecoder(BaseModule):
    def __init__(self, code_length: int, deepest_shape: tuple, output_shape: tuple):
        super(ANDDecoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        activation_fn = nn.LeakyReLU()

        self.fc = nn.Sequential(
            nn.Linear(in_features=code_length, out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=reduce(mul, deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, deepest_shape)),
            activation_fn
        )

        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=activation_fn),
            UpsampleBlock(channel_in=32, channel_out=16, activation_fn=activation_fn),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.FloatTensor):

        h = x
        h = self.fc(h)
        h = h.view(len(h), *self.deepest_shape)
        h = self.conv(h)
        o = h

        return o


class LSAMNIST(BaseModule):

    def __init__(self,  input_shape: tuple, code_length: int, cpd_channels: int):
        super(LSAMNIST, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.cpd_channels = cpd_channels

        self.encoder = Encoder(
            input_shape=input_shape,
            code_length=code_length
        )

        self.decoder = ANDDecoder(
            code_length=code_length,
            deepest_shape=self.encoder.deepest_shape,
            output_shape=input_shape
        )

        self.estimator = Estimator1D(
            code_length=code_length,
            fm_list=[32, 32, 32, 32],
            cpd_channels=cpd_channels
        )

    def forward(self, x: torch.FloatTensor):

        h = x

        z = self.encoder(h)

        z_dist = self.estimator(z)

        x_r = self.decoder(z)
        x_r = x_r.view(-1, *self.input_shape)

        return x_r, z, z_dist
