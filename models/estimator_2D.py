import torch
import torch.nn as nn

from models.base import BaseModule
from models.utils import ListModule


class MaskedConv2d(BaseModule, nn.Conv2d):
    def __init__(self, mask_type: str, idx: int, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in ['A', 'B']
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kt, kd = self.weight.size()
        assert kt == 3
        self.mask.fill_(0)
        self.mask[:, :, :kt // 2, :] = 1
        if idx + (mask_type == 'B') > 0:
            self.mask[:, :, kt // 2, :idx + (mask_type == 'B')] = 1

        self.weight.mask = self.mask

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class MaskedStackedConvolution(BaseModule):

    def __init__(self, mask_type: str, code_length: int, in_channels: int, out_channels: int):
        super(MaskedStackedConvolution, self).__init__()

        self.mask_type = mask_type
        self.code_length = code_length
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = []
        for i in range(0, code_length):
            layers.append(
                MaskedConv2d(mask_type=mask_type,
                             idx=i,
                             in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=(3, code_length),
                             padding=(1, 0))
            )
        self.conv_layers = ListModule(*layers)

    def forward(self, x: torch.FloatTensor):
        out = []
        for i in range(0, self.code_length):
            out.append(self.conv_layers[i](x))
        out = torch.cat(out, dim=-1)

        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'mask_type=' + str(self.mask_type) \
               + ', code_length=' + str(self.code_length) \
               + ', in_channels=' + str(self.in_channels) \
               + ', out_features=' + str(self.out_channels) \
               + ', n_params=' + str(self.n_parameters) + ')'


class Estimator2D(BaseModule):

    def __init__(self, code_length: int, fm_list: list, cpd_channels: int):
        super(Estimator2D, self).__init__()

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
                MaskedStackedConvolution(mask_type=mask_type, code_length=code_length,
                                         in_channels=fm_in, out_channels=fm_out),
            )
            layers_list.append(activation_fn)

            mask_type = 'B'
            fm_in = fm_list[l]

        # Add final layer providing cpd params
        layers_list.append(
            MaskedStackedConvolution(mask_type=mask_type, code_length=code_length,
                                     in_channels=fm_in, out_channels=cpd_channels)
        )
        self.layers = nn.Sequential(*layers_list)

    def forward(self, x: torch.FloatTensor):
        h = torch.unsqueeze(x, dim=1)  # add singleton channel dim
        h = self.layers(h)
        o = h

        return o
