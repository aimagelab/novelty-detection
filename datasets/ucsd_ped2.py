from glob import glob
from os.path import basename
from os.path import isdir
from os.path import join
from typing import List
from typing import Tuple

import numpy as np
import skimage.io as io
import torch
from skimage.transform import resize
from torchvision import transforms

from datasets.base import VideoAnomalyDetectionDataset
from datasets.transforms import ToCrops
from datasets.transforms import ToFloatTensor3D
from utils import concat_collate


class UCSDPed2(VideoAnomalyDetectionDataset):
    """
    Models UCSD Ped2 dataset for video anomaly detection.
    """
    def __init__(self, path):
        # type: (str) -> None
        """
        Class constructor.

        :param path: The folder in which UCSD is stored.
        """
        super(UCSDPed2, self).__init__()

        self.path = join(path, 'UCSDped2')

        # Test directory
        self.test_dir = join(self.path, 'Test')

        # Transform
        self.transform = transforms.Compose([ToFloatTensor3D(), ToCrops(self.raw_shape, self.crop_shape)])

        # Load all test ids
        self.test_ids = self.load_test_ids()

        # Other utilities
        self.cur_len = 0
        self.cur_video_id = None
        self.cur_video_frames = None
        self.cur_video_gt = None

    def load_test_ids(self):
        # type: () -> List[str]
        """
        Loads the set of all test video ids.

        :return: The list of test ids.
        """
        return sorted([basename(d) for d in glob(join(self.test_dir, '**'))
                       if isdir(d) and 'gt' not in basename(d)])

    def load_test_sequence_frames(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads a test video in memory.

        :param video_id: the id of the test video to be loaded
        :return: the video in a np.ndarray, with shape (n_frames, h, w, c).
        """
        c, t, h, w = self.raw_shape

        sequence_dir = join(self.test_dir, video_id)
        img_list = sorted(glob(join(sequence_dir, '*.tif')))
        test_clip = []
        for img_path in img_list:
            img = io.imread(img_path)
            img = resize(img, output_shape=(h, w), preserve_range=True)
            img = np.uint8(img)
            test_clip.append(img)
        test_clip = np.stack(test_clip)
        return test_clip

    def load_test_sequence_gt(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads the groundtruth of a test video in memory.

        :param video_id: the id of the test video for which the groundtruth has to be loaded.
        :return: the groundtruth of the video in a np.ndarray, with shape (n_frames,).
        """
        sequence_dir = join(self.test_dir, f'{video_id}_gt')
        img_list = sorted(glob(join(sequence_dir, '*.bmp')))
        clip_gt = []
        for img_path in img_list:
            img = io.imread(img_path) // 255
            clip_gt.append(np.max(img))  # if at least one pixel is 1, then anomaly
        clip_gt = np.stack(clip_gt)
        return clip_gt

    def test(self, video_id):
        # type: (str) -> None
        """
        Sets the dataset in test mode.

        :param video_id: the id of the video to test.
        """
        c, t, h, w = self.raw_shape

        self.cur_video_id = video_id
        self.cur_video_frames = self.load_test_sequence_frames(video_id)
        self.cur_video_gt = self.load_test_sequence_gt(video_id)

        self.cur_len = len(self.cur_video_frames) - t + 1

    @property
    def shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of examples being fed to the model.
        """
        return self.crop_shape

    @property
    def raw_shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of the raw examples (prior to patches).
        """
        return 1, 16, 256, 384

    @property
    def crop_shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of examples (patches).
        """
        return 1, 8, 32, 32

    @property
    def test_videos(self):
        # type: () -> List[str]
        """
        Returns all available test videos.
        """
        return self.test_ids

    def __len__(self):
        # type: () -> int
        """
        Returns the number of examples.
        """
        return int(self.cur_len)

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        """
        Provides the i-th example.
        """
        c, t, h, w = self.raw_shape

        clip = self.cur_video_frames[i:i+t]
        clip = np.expand_dims(clip, axis=-1)  # add channel dimension
        sample = clip, clip

        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def collate_fn(self):
        """
        Returns a function that decides how to merge a list of examples in a batch.
        """
        return concat_collate

    def __repr__(self):
        return f'UCSD Ped2 (video id = {self.cur_video_id})'
