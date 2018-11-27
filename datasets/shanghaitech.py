from glob import glob
from os.path import basename
from os.path import isdir
from os.path import join
from typing import List

import cv2
import numpy as np
import skimage.io as io
from skimage.transform import resize
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from datasets.base import VideoAnomalyDetectionDataset
from datasets.transforms import RemoveBackground
from datasets.transforms import ToFloatTensor3D


class SHANGHAITECH(VideoAnomalyDetectionDataset):

    def __init__(self, path):

        super(SHANGHAITECH, self).__init__()

        self.path = path
        self.test_dir = join(path, 'testing')

        # Transform
        self.transform = transforms.Compose([RemoveBackground(threshold=128), ToFloatTensor3D(normalize=True)])

        # Load all test ids
        self.test_ids = self.load_test_ids()

        # Other utilities
        self.cur_len = 0
        self.cur_video_id = None
        self.cur_video_frames = None
        self.cur_video_gt = None
        self.cur_background = None

    def load_test_ids(self):
        # type: () -> List[str]
        """
        Loads the set of all test video ids.

        :return: The list of test ids.
        """
        return sorted([basename(d) for d in glob(join(self.test_dir, 'frames', '**')) if isdir(d)])

    def load_test_sequence_frames(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads a test video in memory.

        :param video_id: the id of the test video to be loaded
        :return: the video in a np.ndarray, with shape (n_frames, h, w, c).
        """
        c, t, h, w = self.shape

        sequence_dir = join(self.test_dir,  'frames', video_id)
        img_list = sorted(glob(join(sequence_dir, '*.jpg')))
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
        clip_gt = np.load(join(self.test_dir,  'test_frame_mask', f'{video_id}.npy'))
        return clip_gt

    def test(self, video_id):
        # type: (str) -> None
        """
        Sets the dataset in test mode.

        :param video_id: the id of the video to test.
        """
        c, t, h, w = self.shape

        self.cur_video_id = video_id
        self.cur_video_frames = self.load_test_sequence_frames(video_id)
        self.cur_video_gt = self.load_test_sequence_gt(video_id)
        self.cur_background = self.create_background(self.cur_video_frames)

        self.cur_len = len(self.cur_video_frames) - t + 1

    @property
    def shape(self):
        return 3, 16, 256, 512

    @property
    def test_videos(self):
        return self.test_ids

    def __len__(self):
        return self.cur_len

    @staticmethod
    def create_background(video_frames):

        mog = cv2.createBackgroundSubtractorMOG2()
        for frame in video_frames:
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mog.apply(img)

        # Get background
        background = mog.getBackgroundImage()

        return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    def __getitem__(self, i):

        c, t, h, w = self.shape

        clip = self.cur_video_frames[i:i+t]

        sample = clip, clip, self.cur_background

        # Apply transform
        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def collate_fn(self):
        """
        Returns a function that decides how to merge a list of examples in a batch.
        """
        return default_collate

    def __repr__(self):
        return f'ShanghaiTech (video id = {self.cur_video_id})'

