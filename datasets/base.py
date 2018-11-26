from abc import ABCMeta
from abc import abstractmethod

from torch.utils.data import Dataset


class DatasetBase(Dataset):
    __metaclass__ = ABCMeta

    @abstractmethod
    def test(self, *args):
        pass

    @abstractmethod
    def val(self, *args):
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    @abstractmethod
    def all_tests(self):
        pass


    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, i):
        pass
