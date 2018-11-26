import numpy as np
from torchvision import datasets
from torchvision import transforms

from datasets.base import DatasetBase
from datasets.transforms import OCToFloatTensor2D
from datasets.transforms import ToFloatTensor2D
from datasets.transforms import ToFloat32


class MNIST(DatasetBase):

    def __init__(self, path):

        super(MNIST, self).__init__()

        self.path = path

        self.normal_class = None

        self.train_split = datasets.MNIST(self.path, train=True, download=True, transform=None)
        self.test_split = datasets.MNIST(self.path, train=False, download=True, transform=None)

        # Split randomly the training set to build a validation set
        train_idx = np.arange(len(self.train_split))
        np.random.shuffle(train_idx)
        self.shuffled_train_idx = train_idx

        self.val_transform = transforms.Compose([ToFloatTensor2D()])
        self.test_transform = transforms.Compose([ToFloat32(), OCToFloatTensor2D()])
        self.transform = None

        self.mode = None
        self.length = None

    def val(self, normal_class):
        self.normal_class = int(normal_class)

        self.mode = 'val'
        self.transform = self.val_transform
        self.val_idxs = self.shuffled_train_idx[int(0.9 * len(self.shuffled_train_idx)):]
        self.val_idxs = [idx for idx in self.val_idxs if self.train_split[idx][1] == self.normal_class]
        self.length = len(self.val_idxs)

    def test(self, normal_class):
        self.normal_class = int(normal_class)

        self.mode = 'test'
        self.transform = self.test_transform
        self.length = len(self.test_split)

    def __len__(self):
        return self.length

    def __getitem__(self, i):

        assert self.normal_class is not None, 'Call test() first to select a normal class!'

        if self.mode == 'test':
            X, Y = self.test_split[i]
            X = np.uint8(X)[..., np.newaxis]
            sample = X, int(Y == self.normal_class)

        elif self.mode == 'val':
            X, _ = self.train_split[self.val_idxs[i]]
            X = np.uint8(X)[..., np.newaxis]
            sample = X, X
        else:
            raise ValueError

        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def all_tests(self):
        return list(np.arange(0, 10))

    @property
    def shape(self):
        return 1, 28, 28

    def __repr__(self):
        return f'ONE-CLASS MNIST (normal class = {self.normal_class})'
