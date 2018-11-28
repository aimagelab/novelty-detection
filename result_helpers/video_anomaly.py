from typing import Tuple

import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.base import VideoAnomalyDetectionDataset
from models.base import BaseModule
from models.loss_functions import LSALoss
from utils import normalize
from utils import novelty_score


class ResultsAccumulator:
    """
    Accumulates results in a buffer for a sliding window
    results computation. Employed to get frame-level scores
    from clip-level scores.
    ` In order to recover the anomaly score of each
    frame, we compute the mean score of all clips in which it
    appears`
    """
    def __init__(self, time_steps):
        # type: (int) -> None
        """
        Class constructor.

        :param time_steps: the number of frames each clip holds.
        """

        # This buffers rotate.
        self._buffer = np.zeros(shape=(time_steps,), dtype=np.float32)
        self._counts = np.zeros(shape=(time_steps,))

    def push(self, score):
        # type: (float) -> None
        """
        Pushes the score of a clip into the buffer.
        :param score: the score of a clip
        """

        # Update buffer and counts
        self._buffer += score
        self._counts += 1

    def get_next(self):
        # type: () -> float
        """
        Gets the next frame (the first in the buffer) score,
        computed as the mean of the clips in which it appeared,
        and rolls the buffers.

        :return: the averaged score of the frame exiting the buffer.
        """

        # Return first in buffer
        ret = self._buffer[0] / self._counts[0]

        # Roll time backwards
        self._buffer = np.roll(self._buffer, shift=-1)
        self._counts = np.roll(self._counts, shift=-1)

        # Zero out final frame (next to be filled)
        self._buffer[-1] = 0
        self._counts[-1] = 0

        return ret

    @property
    def results_left(self):
        # type: () -> np.int32
        """
        Returns the number of frames still in the buffer.
        """
        return np.sum(self._counts != 0).astype(np.int32)


class VideoAnomalyDetectionResultHelper(object):
    """
    Performs tests for video anomaly detection datasets (UCSD Ped2 or Shanghaitech).
    """

    def __init__(self, dataset, model, checkpoint, output_file):
        # type: (VideoAnomalyDetectionDataset, BaseModule, str, str) -> None
        """
        Class constructor.

        :param dataset: dataset class.
        :param model: pytorch model to evaluate.
        :param checkpoint: path of the checkpoint for the model.
        :param output_file: text file where to save results.
        """
        self.dataset = dataset
        self.model = model
        self.checkpoint = checkpoint
        self.output_file = output_file

        # Set up loss function
        self.loss = LSALoss(cpd_channels=100)

    @torch.no_grad()
    def test_video_anomaly_detection(self):
        # type: () -> None
        """
        Actually performs tests.
        """

        c, t, h, w = self.dataset.raw_shape

        # Load the checkpoint
        self.model.load_w(self.checkpoint)

        # Prepare a table to show results
        vad_table = self.empty_table

        # Set up container for novelty scores from all test videos
        global_llk = []
        global_rec = []
        global_ns = []
        global_y = []

        # Get accumulators
        results_accumulator_llk = ResultsAccumulator(time_steps=t)
        results_accumulator_rec = ResultsAccumulator(time_steps=t)

        # Start iteration over test videos
        for cl_idx, video_id in enumerate(self.dataset.test_videos):

            # Run the test
            self.dataset.test(video_id)
            loader = DataLoader(self.dataset, collate_fn=self.dataset.collate_fn)

            # Build score containers
            sample_llk = np.zeros(shape=(len(loader) + t - 1,))
            sample_rec = np.zeros(shape=(len(loader) + t - 1,))
            sample_y = self.dataset.load_test_sequence_gt(video_id)
            for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {self.dataset}'):
                x = x.to('cuda')

                x_r, z, z_dist = self.model(x)

                self.loss(x, x_r, z, z_dist)

                # Feed results accumulators
                results_accumulator_llk.push(self.loss.autoregression_loss)
                results_accumulator_rec.push(self.loss.reconstruction_loss)
                sample_llk[i] = results_accumulator_llk.get_next()
                sample_rec[i] = results_accumulator_rec.get_next()

            # Get last results
            while results_accumulator_llk.results_left != 0:
                index = (- results_accumulator_llk.results_left)
                sample_llk[index] = results_accumulator_llk.get_next()
                sample_rec[index] = results_accumulator_rec.get_next()

            min_llk, max_llk, min_rec, max_rec = self.compute_normalizing_coefficients(sample_llk, sample_rec)

            # Compute the normalized scores and novelty score
            sample_llk = normalize(sample_llk, min_llk, max_llk)
            sample_rec = normalize(sample_rec, min_rec, max_rec)
            sample_ns = novelty_score(sample_llk, sample_rec)

            # Update global scores (used for global metrics)
            global_llk.append(sample_llk)
            global_rec.append(sample_rec)
            global_ns.append(sample_ns)
            global_y.append(sample_y)

            try:
                # Compute AUROC for this video
                this_video_metrics = [
                    roc_auc_score(sample_y, sample_llk),  # likelihood metric
                    roc_auc_score(sample_y, sample_rec),  # reconstruction metric
                    roc_auc_score(sample_y, sample_ns)    # novelty score
                ]
                vad_table.add_row([video_id] + this_video_metrics)
            except ValueError:
                # This happens for sequences in which all frames are abnormal
                # Skipping this row in the table (the sequence will still count for global metrics)
                continue

        # Compute global AUROC and print table
        global_llk = np.concatenate(global_llk)
        global_rec = np.concatenate(global_rec)
        global_ns = np.concatenate(global_ns)
        global_y = np.concatenate(global_y)
        global_metrics = [
            roc_auc_score(global_y, global_llk),  # likelihood metric
            roc_auc_score(global_y, global_rec),  # reconstruction metric
            roc_auc_score(global_y, global_ns)    # novelty score
        ]
        vad_table.add_row(['avg'] + list(global_metrics))
        print(vad_table)

        # Save table
        with open(self.output_file, mode='w') as f:
            f.write(str(vad_table))

    @staticmethod
    def compute_normalizing_coefficients(sample_llk, sample_rec):
        # type: (np.ndarray, np.ndarray) -> Tuple[float, float, float, float]
        """
        Computes normalizing coefficients for the computationof the novelty score.

        :param sample_llk: array of log-likelihood scores.
        :param sample_rec: array of reconstruction scores.
        :return: a tuple of normalizing coefficients in the form (llk_min, llk_max, rec_min, rec_max).
        """
        return sample_llk.min(), sample_llk.max(), sample_rec.min(), sample_rec.max()

    @property
    def empty_table(self):
        # type: () -> PrettyTable
        """
        Sets up a nice ascii-art table to hold results.
        This table is suitable for the video anomaly detection setting.
        :return: table to be filled with auroc metrics.
        """
        table = PrettyTable()
        table.field_names = ['VIDEO-ID', 'AUROC-LLK', 'AUROC-REC', 'AUROC-NS']
        table.float_format = '0.3'
        return table
