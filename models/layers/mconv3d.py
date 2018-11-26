from model.base import BaseModule
import torch.nn as nn


class MaskedConv3d(BaseModule, nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv3d, self).__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kT, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kT // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv3d, self).forward(x)

    def __call__(self, *args, **kwargs):
        return super(MaskedConv3d, self).__call__(*args, **kwargs)
