import random

import numpy as np
import torch


def set_random_seed(seed):
    # type: (int) -> None
    """
    Sets random seeds.
    :param seed: the seed to be set for all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def novelty_score(sample_llk, sample_rec, min_llk, max_llk, min_rec, max_rec):
    # type: (np.ndarray, np.ndarray, float, float, float, float) -> np.ndarray
    """
    Computes the normalized novelty score given likelihood scores, reconstruction scores
    and normalization coefficients (Eq. 9-10).
    :param sample_llk: array of log-likelihood scores.
    :param sample_rec: array of reconstruction scores.
    :param min_llk: minimum log-likelihood score.
    :param max_llk: maximum log-likelihood score.
    :param min_rec: minimum reconstruction score.
    :param max_rec: maximum reconstruction score.
    :return: array of novelty scores.
    """

    # Normalize log-likelihood and reconstruction
    sample_llk = (sample_llk - min_llk) / (max_llk - min_llk)
    sample_rec = (sample_rec - min_rec) / (max_rec - min_rec)

    # Sum
    ns = sample_llk + sample_rec

    return ns
