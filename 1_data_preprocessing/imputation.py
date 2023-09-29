"""
Imputation for sequence data.
"""

import pandas as pd
import numpy as np

def interpolate_imputation(sequence_data, feature_median, feature_list):
    """
    Imputation based on pandas' interpolate method.

    :param sequence_data: key: PATNO, value: data matrix
    :param feature_median: median of each feature
    :param feature_list: list of features

    :return: imputed_data
    """

    M = len(feature_list)

    imputed_data = {}

    for p in sequence_data:
        data = pd.DataFrame(sequence_data[p])

        # address the issue that all column are nan
        for m in range(M):
            if data[m].isnull().all():
                data[m] = feature_median[feature_list[m]]

        data = data.interpolate(method='linear', axis=0, limit_direction='both')

        imputed_data[p] = data.values

    return imputed_data


def LOCF_FOCB_imputation(sequence_data, patient_feature_median, feature_list):
    """
    LOFC: last occurrence carry forward strategy
    FOCB: first occurrence carry backward strategy

    :param sequence_data: key: PATNO, value: data matrix
    :param patient_feature_median: median of each feature of each patient
    :param feature_list: list of features

    :return: imputed_data
    """

    imputed_data = {}

    for p in sequence_data:
        data = sequence_data[p]
        L, N = data.shape

        # build mask matrix (0: has missing value in the location, 1: other)
        data = (pd.DataFrame(data)).fillna(-1).values  # fill NaN as -1
        mask_idx = np.where(data == -1)  # first row: x-axis, second row: y-axis
        mask_matrix = np.ones((L, N), dtype='int')
        mask_matrix[mask_idx] = 0

        missing_num = len(mask_idx[0])
        for i in range(missing_num):
            row_idx = mask_idx[0][i]
            col_idx = mask_idx[1][i]

            if L == 1:  # only one visit
                data[row_idx, col_idx] = patient_feature_median[p][feature_list[col_idx]]

            else:       # multiple visit
                if row_idx == 0:  # first visit is NaN

                    if int(data[row_idx + 1, col_idx]) != -1:  # using FOCB
                        data[row_idx, col_idx] = data[row_idx + 1, col_idx]
                    else:  # using median of feature of patient
                        data[row_idx, col_idx] = patient_feature_median[p][feature_list[col_idx]]

                elif row_idx == L-1:  # last visit
                    if int(data[row_idx - 1, col_idx]) != -1:  # using LOCF
                        data[row_idx, col_idx] = data[row_idx - 1, col_idx]
                    else:  # using median of feature of patient
                        data[row_idx, col_idx] = patient_feature_median[p][feature_list[col_idx]]

                else:
                    if int(data[row_idx - 1, col_idx]) != -1:  # using LOCF
                        data[row_idx, col_idx] = data[row_idx - 1, col_idx]
                        continue
                    if int(data[row_idx + 1, col_idx]) != -1:  # using FOCB
                        data[row_idx, col_idx] = data[row_idx + 1, col_idx]
                        continue
                    data[row_idx, col_idx] = patient_feature_median[p][feature_list[col_idx]]

        imputed_data[p] = data
    return imputed_data

def LOCF_FOCB_imputation_exact(sequence_data, feature_list, N_hot_median):
    """
    LOFC: last occurrence carry forward strategy
    FOCB: first occurrence carry backward strategy

    Method:
            for each patient:
                for each feature:
                    run LOFC
                    run FOCB
                for each feature:
                    fill missing values with feature median

    :param sequence_data: key: PATNO, value: data matrix
    :param feature_list: list of features
    :param N_hot_median: map of N_hot feature median values

    :return: imputed_data
    """

    imputed_data = {}
    # sequence_data[3001][:, 0:5] = np.nan
    for p in sequence_data:
        data = sequence_data[p]
        L, N = data.shape

        # build mask matrix (0: has missing value in the location, 1: other)
        data = (pd.DataFrame(data)).fillna(-1).values  # fill NaN as -1
        mask_idx = np.where(data == -1)  # first row: x-axis, second row: y-axis
        mask_matrix = np.ones((L, N), dtype='int')
        mask_matrix[mask_idx] = 0
        # print(data)
        # print(mask_matrix)
        if L == 1:  # only one visit
            missing_num = len(mask_idx[0])
            for i in range(missing_num):
                row_idx = mask_idx[0][i]
                col_idx = mask_idx[1][i]
                data[row_idx, col_idx] = N_hot_median[feature_list[col_idx]]    #       patient_feature_median[p][feature_list[col_idx]]
        else:  # multiple visit
            ### run LOCF
            missing_num = len(mask_idx[0])
            for i in range(missing_num):
                row_idx = mask_idx[0][i]
                col_idx = mask_idx[1][i]
                if row_idx != 0 and int(data[row_idx - 1, col_idx]) != -1:
                    data[row_idx, col_idx] = data[row_idx - 1, col_idx]
            # update mask
            mask_idx = np.where(data == -1)  # first row: x-axis, second row: y-axis
            mask_matrix = np.ones((L, N), dtype='int')
            mask_matrix[mask_idx] = 0


            # ### run FOCB
            missing_num = len(mask_idx[0])
            for i in range(missing_num):
                row_idx = mask_idx[0][i]
                col_idx = mask_idx[1][i]
                if row_idx != L-1 and int(data[row_idx + 1, col_idx]) != -1:
                    data[row_idx, col_idx] = data[row_idx + 1, col_idx]

            # update mask
            mask_idx = np.where(data == -1)  # first row: x-axis, second row: y-axis
            mask_matrix = np.ones((L, N), dtype='int')
            mask_matrix[mask_idx] = 0
            missing_num = len(mask_idx[0])

            for i in range(missing_num):  # complete missing features
                row_idx = mask_idx[0][i]
                col_idx = mask_idx[1][i]
                data[row_idx, col_idx] = N_hot_median[feature_list[col_idx]]

            # update mask
            mask_idx = np.where(data == -1)  # first row: x-axis, second row: y-axis
            mask_matrix = np.ones((L, N), dtype='int')
            mask_matrix[mask_idx] = 0
            missing_num = len(mask_idx[0])
            if missing_num != 0:
                print(p, "--------- still have missing values!!!!")
        imputed_data[p] = data
    return imputed_data

