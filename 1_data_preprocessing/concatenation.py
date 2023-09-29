"""
Data concatenation.

Input:
      1. data
      2. patient included
      3. length of info of patient (number of visits)
      4. visit list
      5. feature list

Output:
      1. sequential data of patient
      2. median of variable
      3. median of variable of patient
"""

import pandas as pd
import numpy as np

def concatenate(data_df, patient_length, vist_id_map, feature_list):
    """
    :param data_df: DataFrame
    :param patient_length: Dictionary: key: PATNO, value: length of patient info (number of visit)
    :param vist_id_map: Dictionary: key: ENVENT_ID, value: int 0-16
    :param feature_list: List of feature used

    :return: patient_arrays, feature_median, patient_feature_median
    """

    M = len(feature_list)

    # initialize sequential data of each patient
    patient_arrays = {}  # key: PATNO, value: numpy ndarray (column: feature, row: visit)
    for p in patient_length:
        n = patient_length[p]
        patient_arrays[p] = np.full((n, M), np.nan)

    # create feature_id map
    feature_id_map = {}  # key: feature name, value: f_id (int)
    for f_id in range(len(feature_list)):
        feature_id_map[feature_list[f_id]] = f_id
    # print(feature_id_map)

    # update the data matrices of patients
    print(vist_id_map)
    data_df = data_df[data_df['Variable'].isin(feature_id_map)]  # exclude unused features
    for idx, row in data_df.iterrows():
        PATNO, EVENT_ID, Variable, Value = row['PATNO'], row['EVENT_ID'], row['Variable'], row['Value']

        # move SC moca to BL
        if EVENT_ID == 'SC':
            if Variable in ["visuospatial", "naming", "attention", "language", "delayed_recall", "MCAABSTR", "MCAVFNUM", "MCATOT"]:
                EVENT_ID = 'BL'
            else:
                continue


        v_id = vist_id_map[EVENT_ID]
        f_id = feature_id_map[Variable]
        patient_arrays[PATNO][v_id, f_id] = Value

    # compute feature median
    patient_null_column = []
    feature_median = {}  # key: feature, value: median of the feature
    patient_feature_median = {}  # key: patient, value: { feature : median of the feature of patient }
    for PATNO in patient_length:
        patient_feature_median[PATNO] = {}  # initialize
    for var in feature_list:
        # feature median
        temp_data = data_df[data_df['Variable'] == var]
        feature_median[var] = np.median(list(temp_data['Value'].astype('float').values))
        # patient median
        for p in patient_feature_median:
            patient_temp_data = temp_data[temp_data['PATNO'] == p]
            tmp_values = list(patient_temp_data['Value'].astype('float').values)
            if len(tmp_values) == 0:  # patient do not have information of this feature
                patient_feature_median[p][var] = feature_median[var]
                patient_null_column.append(p)
            else:
                patient_feature_median[p][var] = round(np.median(tmp_values))
    # patient_null_column = list(set(patient_null_column))
    # print(patient_null_column)
    # print(feature_median)
    # print(patient_feature_median)
    # print(patient_arrays[3401])
    return (patient_arrays, feature_median, patient_feature_median)