"""

Normalization for the data.

"""

import numpy as np
import copy

def Z_score_normalization(sequence_data, M):
    """
    (value - mean) / standard deviation

    :param sequence_data: key: PATNO, value: data matrix

    :return: normalized_sequence_data
    """
    patients = sequence_data.keys()

    normalized_sequence_data = copy.deepcopy(sequence_data)

    for m in range(M):
        values = np.array([])  # initialize the vector of m-th feature
        for p in patients:
            values = np.concatenate((values, sequence_data[p][:, m]))
        # take the mean()
        feature_mean = values.mean()
        # take standard deviation
        feature_std = values.std()

        # update the data with Z score
        for p in patients:
            Zscore = (sequence_data[p][:, m] - feature_mean) / feature_std
            normalized_sequence_data[p][:, m] = Zscore

    return normalized_sequence_data


def minmax_normalization(sequence_data, M):
    """
        (value - min) / (max - min)

        :param sequence_data: key: PATNO, value: data matrix

        :return: (normalized) sequence_data
        """
    patients = sequence_data.keys()

    normalized_sequence_data = copy.deepcopy(sequence_data)

    for m in range(M):
        values = np.array([])  # initialize the vector of m-th feature
        for p in patients:
            values = np.concatenate((values, sequence_data[p][:, m]))
        # take the min()
        feature_min = values.min()
        # take the max()
        feature_max = values.max()

        # update the data with min-max normalization value
        for p in patients:
            norm_value = (sequence_data[p][:, m] - feature_min) / (feature_max - feature_min)
            normalized_sequence_data[p][:, m] = norm_value

    return normalized_sequence_data