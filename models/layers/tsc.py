import torch
import torch.nn as nn

from models.base import BaseModule


class TemporallySharedFullyConnection(BaseModule):
    """
    Implements a temporally-shared fully connection.
    Processes a time series of feature vectors and performs
    the same linear projection to all of them.
    """
    def __init__(self, in_features, out_features, bias=True):
        # type: (int, int, bool) -> None
        """
        Class constructor.

        :param in_features: number of input features.
        :param out_features: number of output features.
        :param bias: whether or not to add bias.
        """
        super(TemporallySharedFullyConnection, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # the layer to be applied at each timestep
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward function.

        :param x: layer input. Has shape=(batchsize, seq_len, in_features).
        :return: layer output. Has shape=(batchsize, seq_len, out_features)
        """
        b, t, d = x.size()

        output = []
        for i in range(0, t):
            # apply dense layer
            output.append(self.linear(x[:, i, :]))
        output = torch.stack(output, 1)

        return output
