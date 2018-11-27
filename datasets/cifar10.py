from typing import Tuple
from typing import Union

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms

from datasets.base import OneClassDataset
from datasets.transforms import OCToFloatTensor2D
from datasets.transforms import ToFloat32
from datasets.transforms import ToFloatTensor2D


class CIFAR10(OneClassDataset):
    """
    Models CIFAR10 dataset for one class classification.
    """
    def __init__(self, path):
        # type: (str) -> None
        """
        Class constructor.

        :param path: The folder in which to download CIFAR10.
        """
        super(CIFAR10, self).__init__()

        self.path = path

        self.normal_class = None

        # Get train and test split
        self.train_split = datasets.CIFAR10(self.path, train=True, download=True, transform=None)
        self.test_split = datasets.CIFAR10(self.path, train=False, download=True, transform=None)

        # Shuffle training indexes to build a validation set (see val())
        train_idx = np.arange(len(self.train_split))
        np.random.shuffle(train_idx)
        self.shuffled_train_idx = train_idx

        # Transform zone
        self.val_transform = transforms.Compose([ToFloatTensor2D()])
        self.test_transform = transforms.Compose([ToFloat32(), OCToFloatTensor2D()])
        self.transform = None

        # Other utilities
        self.mode = None
        self.length = None
        self.val_idxs = None

    def val(self, normal_class):
        # type: (int) -> None
        """
        Sets CIFAR10 in validation mode.

        :param normal_class: the class to be considered normal.
        """
        self.normal_class = int(normal_class)

        # Update mode, indexes, length and transform
        self.mode = 'val'
        self.transform = self.val_transform
        self.val_idxs = self.shuffled_train_idx[int(0.9 * len(self.shuffled_train_idx)):]
        self.val_idxs = [idx for idx in self.val_idxs if self.train_split[idx][1] == self.normal_class]
        self.length = len(self.val_idxs)

    def test(self, normal_class):
        # type: (int) -> None
        """
        Sets CIFAR10 in test mode.

        :param normal_class: the class to be considered normal.
        """
        self.normal_class = int(normal_class)

        # Update mode, length and transform
        self.mode = 'test'
        self.transform = self.test_transform
        self.length = len(self.test_split)

    def __len__(self):
        # type: () -> int
        """
        Returns the number of examples.
        """
        return self.length

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]
        """
        Provides the i-th example.
        """
        assert self.normal_class is not None, 'Call test() first to select a normal class!'

        # Load the i-th example
        if self.mode == 'test':
            x, y = self.test_split[i]
            sample = x, int(y == self.normal_class)
        elif self.mode == 'val':
            x, _ = self.train_split[self.val_idxs[i]]
            sample = x, x
        else:
            raise ValueError

        # Apply transform
        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def test_classes(self):
        # type: () -> np.ndarray
        """
        Returns all test possible test sets (the 10 classes).
        """
        return np.arange(0, 10)

    @property
    def shape(self):
        # type: () -> Tuple[int, int, int]
        """
        Returns the shape of examples.
        """
        return 3, 32, 32

    def __repr__(self):
        return f'ONE-CLASS CIFAR10 (normal class = {self.normal_class})'
