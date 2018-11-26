import torch
import torch.nn as nn

from models.base import BaseModule


class TemporallySharedFullyConnection(BaseModule):

    def __init__(self, in_features: int, out_features: int, bias: bool=True):

        super(TemporallySharedFullyConnection, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # the layer to be applied at each timestep
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x: torch.FloatTensor):
        """
        Forward function.

        Parameters
        ----------
        x: Variable
            layer input. Has shape=(batchsize, seq_len, in_features)
        Returns
        -------
        output: Variable
            layer output. Has shape=(batchsize, seq_len, out_features)
        """
        b, t, d = x.size()

        output = []
        for i in range(0, t):
            # apply dense layer
            output.append(self.linear(x[:, i, :]))
        output = torch.stack(output, 1)

        return output
