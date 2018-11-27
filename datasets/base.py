from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    """
    Base class for all datasets.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def test(self, *args):
        """
        Sets the dataset in test mode.
        """
        pass

    @property
    @abstractmethod
    def shape(self):
        """
        Returns the shape of examples.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Returns the number of examples.
        """
        pass

    @abstractmethod
    def __getitem__(self, i):
        """
        Provides the i-th example.
        """
        pass


class OneClassDataset(DatasetBase):
    """
    Base class for all one-class classification datasets.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def val(self, *args):
        """
        Sets the dataset in validation mode.
        """
        pass

    @property
    @abstractmethod
    def test_classes(self):
        """
        Returns all test possible test classes.
        """
        pass


class VideoAnomalyDetectionDataset(DatasetBase):
    """
    Base class for all video anomaly detection datasets.
    """
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def test_videos(self):
        """
        Returns all test video ids.
        """
        pass


    @abstractmethod
    def __len__(self):
        """
        Returns the number of examples.
        """
        pass

    @property
    def raw_shape(self):
        """
        Workaround!
        """
        return self.shape

    @abstractmethod
    def __getitem__(self, i):
        """
        Provides the i-th example.
        """
        pass

    @abstractmethod
    def load_test_sequence_gt(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads the groundtruth of a test video in memory.

        :param video_id: the id of the test video for which the groundtruth has to be loaded.
        :return: the groundtruth of the video in a np.ndarray, with shape (n_frames,).
        """
        pass

    @property
    @abstractmethod
    def collate_fn(self):
        """
        Returns a function that decides how to merge a list of examples in a batch.
        """
        pass
