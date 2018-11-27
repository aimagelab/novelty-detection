import torch
import torch.nn as nn

from models.base import BaseModule
from models.utils import ListModule


class MaskedConv2d(BaseModule, nn.Conv2d):
    """
    Implements a Masked Convolution 2D.
    This is a 2D convolution with a masked kernel.
    """
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
        # type: (torch.Tensor) -> torch.Tensor
        """
        Performs the forward pass.

        :param x: the input tensor.
        :return: the output tensor as result of the convolution.
        """
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class MaskedStackedConvolution(BaseModule):
    """
    Implements a Masked Stacked Convolution layer (MSC, Eq. 7).
    This is the autoregressive layer employed for the estimation of
    densities of video feature vectors.
    """
    def __init__(self, mask_type, code_length, in_channels, out_channels):
        """
        Class constructor.

        :param mask_type: type of autoregressive layer, either `A` or `B`.
        :param code_length: the lentgh of each feature vector in the time series.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        """
        super(MaskedStackedConvolution, self).__init__()

        self.mask_type = mask_type
        self.code_length = code_length
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build a masked convolution for each element of the code
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

        # Merge all layers in a list module
        self.conv_layers = ListModule(*layers)

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input tensor.
        :return: the output of a MSC manipulation.
        """
        out = []
        for i in range(0, self.code_length):
            out.append(self.conv_layers[i](x))
        out = torch.cat(out, dim=-1)

        return out

    def __repr__(self):
        # type: () -> str
        """
        String representation.
        """
        return self.__class__.__name__ + '(' \
               + 'mask_type=' + str(self.mask_type) \
               + ', code_length=' + str(self.code_length) \
               + ', in_channels=' + str(self.in_channels) \
               + ', out_features=' + str(self.out_channels) \
               + ', n_params=' + str(self.n_parameters) + ')'


class Estimator2D(BaseModule):
    """
    Implements an estimator for 2-dimensional vectors.
    2-dimensional vectors arise from the encoding of video clips.
    This module is employed in UCSD Ped2 and ShanghaiTech LSA models.
    Takes as input a time series of latent vectors and outputs cpds for each variable.
    """
    def __init__(self, code_length, fm_list, cpd_channels):
        # type: (int, List[int], int) -> None
        """
        Class constructor.

        :param code_length: the dimensionality of latent vectors.
        :param fm_list: list of channels for each MFC layer.
        :param cpd_channels: number of bins in which the multinomial works.
        """
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
