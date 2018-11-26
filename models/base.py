from functools import reduce
from operator import mul

import torch
import torch.nn as nn


class BaseModule(nn.Module):

    def load_w(self, checkpoint_path: str):
        self.load_state_dict(torch.load(checkpoint_path))

    def __repr__(self):
        good_old = super(BaseModule, self).__repr__()
        addition = 'Total number of parameters: {:,}'.format(self.n_parameters)

        return good_old + '\n' + addition

    def __call__(self, *args, **kwargs):
        return super(BaseModule, self).__call__(*args, **kwargs)

    @property
    def n_parameters(self):
        n_parameters = 0
        for p in self.parameters():
            if hasattr(p, 'mask'):
                n_parameters += torch.sum(p.mask).item()
            else:
                n_parameters += reduce(mul, p.shape)
        return int(n_parameters)
