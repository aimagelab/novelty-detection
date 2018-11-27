import torch
import torch.nn as nn

from models.base import BaseModule
from models.blocks_3d import DownsampleBlock
from models.blocks_3d import UpsampleBlock
from models.estimator_2D import Estimator2D
from models.layers.tsc import TemporallySharedFullyConnection


class Encoder(BaseModule):

    def __init__(self, input_shape, code_length):
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        c, t, h, w = input_shape

        activation_fn = nn.LeakyReLU()

        self.conv = nn.Sequential(
            DownsampleBlock(channel_in=c, channel_out=8, activation_fn=activation_fn, stride=(1, 2, 2)),
            DownsampleBlock(channel_in=8, channel_out=16, activation_fn=activation_fn, stride=(1, 2, 2)),
            DownsampleBlock(channel_in=16, channel_out=32, activation_fn=activation_fn, stride=(2, 2, 2)),
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation_fn, stride=(1, 2, 2)),
            DownsampleBlock(channel_in=64, channel_out=64, activation_fn=activation_fn, stride=(2, 2, 2))
        )

        self.deepest_shape = (64, t // 4, h // 32, w // 32)

        dc, dt, dh, dw = self.deepest_shape
        self.tdl = nn.Sequential(
            TemporallySharedFullyConnection(in_features=(dc * dh * dw), out_features=512),
            nn.Tanh(),
            TemporallySharedFullyConnection(in_features=512, out_features=code_length),
            nn.Sigmoid()
        )

    def forward(self, x):

        h = x
        h = self.conv(h)

        # reshape for tdd
        c, t, height, width = self.deepest_shape
        h = torch.transpose(h, 1, 2).contiguous()
        h = h.view(-1, t, (c * height * width))
        o = self.tdl(h)

        return o


class ANDDecoder(BaseModule):

    def __init__(self, code_length, deepest_shape, output_shape):

        super(ANDDecoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        dc, dt, dh, dw = deepest_shape

        activation_fn = nn.LeakyReLU()

        self.tdl = nn.Sequential(
            TemporallySharedFullyConnection(in_features=code_length, out_features=512),
            nn.Tanh(),
            TemporallySharedFullyConnection(in_features=512, out_features=(dc * dh * dw)),
            activation_fn
        )

        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=dc, channel_out=64,
                          activation_fn=activation_fn, stride=(2, 2, 2), output_padding=(1, 1, 1)),
            UpsampleBlock(channel_in=64, channel_out=32,
                          activation_fn=activation_fn, stride=(1, 2, 2), output_padding=(0, 1, 1)),
            UpsampleBlock(channel_in=32, channel_out=16,
                          activation_fn=activation_fn, stride=(2, 2, 2), output_padding=(1, 1, 1)),
            UpsampleBlock(channel_in=16, channel_out=8,
                          activation_fn=activation_fn, stride=(1, 2, 2), output_padding=(0, 1, 1)),
            UpsampleBlock(channel_in=8, channel_out=8,
                          activation_fn=activation_fn, stride=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.Conv3d(in_channels=8, out_channels=output_shape[0], kernel_size=1)
        )

    def forward(self, x):

        h = x
        h = self.tdl(h)

        # reshape
        h = torch.transpose(h, 1, 2).contiguous()
        h = h.view(len(h), *self.deepest_shape)

        h = self.conv(h)
        o = h

        return o


class LSAShanghaiTech(BaseModule):

    def __init__(self, input_shape, code_length, cpd_channels):

        super(LSAShanghaiTech, self).__init__()

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

        self.estimator = Estimator2D(
            code_length=code_length,
            fm_list=[4, 4],
            cpd_channels=cpd_channels
        )

    def forward(self, x):

        h = x

        z = self.encoder(h)

        z_dist = self.estimator(z)

        x_r = self.decoder(z)
        x_r = x_r.view(-1, *self.input_shape)

        return x_r, z, z_dist
