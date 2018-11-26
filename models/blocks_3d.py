import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.mconv3d import MaskedConv3d

from model.base import BaseModule

DAVID_MODE = 'david'
BERGA_MODE = 'berga'
PLAIN_MODE = 'plain'
OLD_MODE = 'old'
OP_MODES = [DAVID_MODE, BERGA_MODE, PLAIN_MODE, OLD_MODE]


def resolve_ops(mode_id: str) -> tuple:
    map_id = {
        DAVID_MODE: (ResidualBlock, UpsampleDavidBlock, DownsampleDavidBlock),
        BERGA_MODE: (ResidualBlock, UpsampleBergaBlock, DownsampleBergaBlock),
        PLAIN_MODE: (PlainBlock, UpsamplePlainBlock, DownsamplePlainBlock),
        OLD_MODE: (None, DecoderBlock, EncoderBlock)
    }
    assert mode_id in map_id.keys()
    return map_id[mode_id]


def residual_op(x: torch.FloatTensor, functions: list, bns: list, activation_fn: nn.Module):
    """ Implements a global residual operation. """
    f1, f2, f3 = functions
    bn1, bn2, bn3 = bns

    assert len(functions) == len(bns) == 3
    assert f1 is not None and f2 is not None
    assert not (f3 is None and bn3 is not None)

    # a branch
    ha = x
    ha = f1(ha)
    if bn1 is not None:
        ha = bn1(ha)
    ha = activation_fn(ha)

    ha = f2(ha)
    if bn2 is not None:
        ha = bn2(ha)

    # b branch
    hb = x
    if f3 is not None:
        hb = f3(hb)
    if bn3 is not None:
        hb = bn3(hb)

    # residual connection
    out = ha + hb
    return activation_fn(out)


def plain_op(x: torch.FloatTensor, functions: list, bns: list, activation_fn: nn.Module):
    f1, f2 = functions
    bn1, bn2 = bns

    assert len(functions) == len(bns) == 2
    assert f1 is not None and f2 is not None

    h = x
    h = f1(h)
    if bn1 is not None:
        h = bn1(h)
    h = activation_fn(h)
    h = f2(h)
    if bn2 is not None:
        h = bn2(h)
    h = activation_fn(h)
    out = h
    return out


class BaseBlock(BaseModule):
    """ Base class for all blocks. """

    def __init__(self, channel_in: int, channel_out: int, activation_fn: nn.Module,
                 use_bn: bool = True, use_bias: bool = True):
        super(BaseBlock, self).__init__()

        # assert not (use_bn and use_bias), 'Using bias=True with batch_normalization is forbidden.'

        self._channel_in = channel_in
        self._channel_out = channel_out
        self._activation_fn = activation_fn
        self._use_bn = use_bn
        self._bias = use_bias

    def get_bn(self):
        return nn.BatchNorm3d(num_features=self._channel_out) if self._use_bn else None

    def forward(self, x: torch.FloatTensor):
        raise NotImplementedError


class DownsampleDavidBlock(BaseBlock):
    def __init__(self, channel_in: int, channel_out: int, activation_fn: nn.Module, stride: tuple,
                 use_bn: bool = True, use_bias: bool = True):
        super(DownsampleDavidBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)
        self.stride = stride

        # convolution to halve the dimensions
        self.conv1a = MaskedConv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                                   padding=1, stride=stride, bias=use_bias)
        self.conv1b = MaskedConv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                   padding=1, stride=1, bias=use_bias)
        self.conv2a = nn.Conv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=1,
                                padding=0, stride=stride, bias=use_bias)

        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x: torch.FloatTensor):
        return residual_op(x,
                           functions=[self.conv1a, self.conv1b, self.conv2a],
                           bns=[self.bn1a, self.bn1b, self.bn2a],
                           activation_fn=self._activation_fn)


class UpsampleDavidBlock(BaseBlock):
    def __init__(self, channel_in: int, channel_out: int, activation_fn: nn.Module, stride: tuple,
                 output_padding: tuple, use_bn: bool = True, use_bias: bool = True):
        super(UpsampleDavidBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)
        self.stride = stride
        self.output_padding = output_padding

        # convolution to halve the dimensions
        self.conv1a = nn.ConvTranspose3d(channel_in, channel_out, kernel_size=5,
                                         padding=2, stride=stride, output_padding=output_padding, bias=use_bias)
        self.conv1b = nn.Conv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                padding=1, stride=1, bias=use_bias)
        self.conv2a = nn.ConvTranspose3d(channel_in, channel_out, kernel_size=5,
                                         padding=2, stride=stride, output_padding=output_padding, bias=use_bias)

        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x: torch.FloatTensor):
        return residual_op(x,
                           functions=[self.conv1a, self.conv1b, self.conv2a],
                           bns=[self.bn1a, self.bn1b, self.bn2a],
                           activation_fn=self._activation_fn)


class DownsampleBergaBlock(BaseBlock):
    def __init__(self, channel_in: int, channel_out: int, activation_fn: nn.Module, stride: tuple,
                 use_bn: bool = True, use_bias: bool = True):
        super(DownsampleBergaBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)
        self.stride = stride

        # convolution to halve the dimensions
        self.convd = MaskedConv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                                  padding=1, stride=stride, bias=use_bias)
        self.conv1 = MaskedConv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                  padding=1, stride=1, bias=use_bias)
        self.conv2 = MaskedConv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                  padding=1, stride=1, bias=use_bias)

        self.bnd = self.get_bn()
        self.bn1 = self.get_bn()
        self.bn2 = self.get_bn()

    def forward(self, x: torch.FloatTensor):
        # first downsampling conv
        ha = self.convd(x)
        if self._use_bn:
            ha = self.bnd(ha)
        ha = self._activation_fn(ha)

        return residual_op(ha,
                           functions=[self.conv1, self.conv2, None],
                           bns=[self.bn1, self.bn2, None],
                           activation_fn=self._activation_fn)


class UpsampleBergaBlock(BaseBlock):
    def __init__(self, channel_in: int, channel_out: int, activation_fn: nn.Module, stride: tuple,
                 output_padding: tuple, use_bn: bool = True, use_bias: bool = True):
        super(UpsampleBergaBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)
        self.stride = stride
        self.output_padding = output_padding

        # convolution to halve the dimensions
        self.convd = nn.ConvTranspose3d(channel_in, channel_out, kernel_size=5,
                                        padding=2, stride=stride, output_padding=output_padding, bias=use_bias)
        self.conv1 = nn.Conv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                               padding=1, stride=1, bias=use_bias)
        self.conv2 = nn.Conv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                               padding=1, stride=1, bias=use_bias)

        self.bnd = self.get_bn()
        self.bn1 = self.get_bn()
        self.bn2 = self.get_bn()

    def forward(self, x: torch.FloatTensor):
        # first downsampling conv
        ha = self.convd(x)
        if self._use_bn:
            ha = self.bnd(ha)
        ha = self._activation_fn(ha)

        return residual_op(ha,
                           functions=[self.conv1, self.conv2, None],
                           bns=[self.bn1, self.bn2, None],
                           activation_fn=self._activation_fn)


class ResidualBlock(BaseBlock):
    def __init__(self, channel_in: int, channel_out: int, activation_fn: nn.Module, conv_type: str,
                 use_bn: bool = True, use_bias: bool = True):
        super(ResidualBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        assert conv_type in ['masked', 'normal']
        ConvClass = MaskedConv3d if conv_type == 'masked' else nn.Conv3d

        self.conv1 = ConvClass(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                               padding=1, stride=1, bias=use_bias)
        self.conv2 = ConvClass(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                               padding=1, stride=1, bias=use_bias)

        self.bn1 = self.get_bn()
        self.bn2 = self.get_bn()

    def forward(self, x: torch.FloatTensor):
        return residual_op(x,
                           functions=[self.conv1, self.conv2, None],
                           bns=[self.bn1, self.bn2, None],
                           activation_fn=self._activation_fn)


class PlainBlock(BaseBlock):
    def __init__(self, channel_in: int, channel_out: int, activation_fn: nn.Module, conv_type: str,
                 use_bn: bool = True, use_bias: bool = True):
        super(PlainBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        assert conv_type in ['masked', 'normal']
        ConvClass = MaskedConv3d if conv_type == 'masked' else nn.Conv3d

        self.conv1 = ConvClass(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                               padding=1, stride=1, bias=use_bias)
        self.conv2 = ConvClass(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                               padding=1, stride=1, bias=use_bias)

        self.bn1 = self.get_bn()
        self.bn2 = self.get_bn()

    def forward(self, x: torch.FloatTensor):
        return plain_op(x,
                        functions=[self.conv1, self.conv2],
                        bns=[self.bn1, self.bn2],
                        activation_fn=self._activation_fn)


class DownsamplePlainBlock(BaseBlock):
    def __init__(self, channel_in: int, channel_out: int, activation_fn: nn.Module, stride: tuple,
                 use_bn: bool = True, use_bias: bool = True):
        super(DownsamplePlainBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)
        self.stride = stride

        self.conv1 = MaskedConv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                                  padding=1, stride=stride, bias=use_bias)
        self.conv2 = MaskedConv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                  padding=1, stride=1, bias=use_bias)

        self.bn1 = self.get_bn()
        self.bn2 = self.get_bn()

    def forward(self, x: torch.FloatTensor):
        return plain_op(x,
                        functions=[self.conv1, self.conv2],
                        bns=[self.bn1, self.bn2],
                        activation_fn=self._activation_fn)


class UpsamplePlainBlock(BaseBlock):
    def __init__(self, channel_in: int, channel_out: int, activation_fn: nn.Module, stride: tuple,
                 output_padding: tuple, use_bn: bool = True, use_bias: bool = True):
        super(UpsamplePlainBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)
        self.stride = stride
        self.output_padding = output_padding

        self.conv1 = nn.ConvTranspose3d(channel_in, channel_out, kernel_size=5,
                                        padding=2, stride=stride, output_padding=output_padding, bias=use_bias)
        self.conv2 = nn.Conv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                               padding=1, stride=1, bias=use_bias)

        self.bn1 = self.get_bn()
        self.bn2 = self.get_bn()

    def forward(self, x: torch.FloatTensor):
        return plain_op(x,
                        functions=[self.conv1, self.conv2],
                        bns=[self.bn1, self.bn2],
                        activation_fn=self._activation_fn)


class EncoderBlock(BaseModule):

    def __init__(self, channel_in: int, channel_out: int, stride: tuple):
        super(EncoderBlock, self).__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.conv1a = MaskedConv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                                   padding=1, stride=stride)
        self.conv1b = nn.Conv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=1,
                                padding=0, stride=stride)
        self.conv2 = MaskedConv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                  padding=1, stride=1)
        self.conv3 = MaskedConv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                  padding=1, stride=1)

        self.bn1a = nn.BatchNorm3d(num_features=channel_out, momentum=0.9)
        self.bn1b = nn.BatchNorm3d(num_features=channel_out, momentum=0.9)
        self.bn2 = nn.BatchNorm3d(num_features=channel_out, momentum=0.9)
        self.bn3 = nn.BatchNorm3d(num_features=channel_out, momentum=0.9)

    def forward(self, x: torch.FloatTensor):

        # a branch
        ha = x
        ha = self.conv1a(ha)
        ha = self.bn1a(ha)
        ha = F.leaky_relu(ha)
        ha = self.conv2(ha)
        ha = self.bn2(ha)
        ha = F.leaky_relu(ha)
        ha = self.conv3(ha)
        ha = self.bn3(ha)

        # b branch
        hb = x
        hb = self.conv1b(hb)
        hb = self.bn1b(hb)

        # residual connection
        out = ha + hb

        return out


class DecoderBlock(BaseModule):

    def __init__(self, channel_in: int, channel_out: int, stride: tuple, output_padding: tuple):

        super(DecoderBlock, self).__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.conv1a = nn.ConvTranspose3d(channel_in, channel_out, kernel_size=5, padding=2, stride=stride,
                                         output_padding=output_padding)
        self.conv1b = nn.ConvTranspose3d(channel_in, channel_out, kernel_size=5, padding=2, stride=stride,
                                         output_padding=output_padding)
        self.conv2 = nn.Conv3d(channel_out, channel_out, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv3d(channel_out, channel_out, kernel_size=3, padding=1, stride=1)

        self.bn1a = nn.BatchNorm3d(channel_out, momentum=0.9)
        self.bn1b = nn.BatchNorm3d(channel_out, momentum=0.9)
        self.bn2 = nn.BatchNorm3d(channel_out, momentum=0.9)
        self.bn3 = nn.BatchNorm3d(channel_out, momentum=0.9)

    def forward(self, x: torch.FloatTensor):

        # a branch
        ha = x
        ha = self.conv1a(ha)
        ha = self.bn1a(ha)
        ha = F.leaky_relu(ha)
        ha = self.conv2(ha)
        ha = self.bn2(ha)
        ha = F.leaky_relu(ha)
        ha = self.conv3(ha)
        ha = self.bn3(ha)

        # b branch
        hb = x
        hb = self.conv1b(hb)
        hb = self.bn1b(hb)

        # residual connection
        out = ha + hb

        return out