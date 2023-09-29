"""

Format longitudinal clinical data.

Input:
       patient_info.csv
       feature_dictionary.csv
       data.csv
       patient_visit_info.csv

Output:
       Feature matrices (column: feature, row: visit) of patients

       * Of note: features are continuous value

"""

import pandas as pd
import numpy as np
from concatenation import concatenate
from imputation import interpolate_imputation
from imputation import LOCF_FOCB_imputation
from normalization import Z_score_normalization
from normalization import minmax_normalization
import dict_key_convert
import json
import pickle as pkl

def format(version, imputation_method, excluded_dataset = ['cog_catg', 'upsit_booklet', 'pd_medication'
                                                           , 'vital_signs', 'pase_house', 'updrs4', 'pase_house'
                                                           #, 'alpha_syn','total_tau', 'abeta_42', 'p_tau181p'
                                                           ]):
    vist_list = ['BL'] + ["V%02d" % i for i in range(1, 17)]
    vid_list = {}
    for v in range(len(vist_list)):
        vid_list[vist_list[v]] = v

    patient_df = pd.read_csv("../"+version+"/patient_info.csv")
    entire_cohort = list(patient_df.PATNO.values)
    PD_cohort = list(patient_df[patient_df['ENROLL_CAT'] == 'PD'].PATNO.values)
    HC_cohort = list(patient_df[patient_df['ENROLL_CAT'] == 'HC'].PATNO.values)
    SWEDD_cohort = list(patient_df[patient_df['ENROLL_CAT'] == 'SWEDD'].PATNO.values)
    Study_cohort = PD_cohort+HC_cohort+SWEDD_cohort
    Other_cohort = [i for i in entire_cohort if i not in Study_cohort]
    patient_df = patient_df[patient_df['PATNO'].isin(PD_cohort + SWEDD_cohort + HC_cohort)]

    visit_info_df = pd.read_csv("../"+version+"/patient_visit_info.csv")
    visit_info_df = visit_info_df[visit_info_df['PATNO'].isin(Study_cohort)]


    feature_df = pd.read_csv("../"+version+"/feature_dictionary.csv")
    # need to check updrs3a, udprs4, schwab, and pase_house

    included_dataset = [d for d in set(feature_df['Source']) if d not in excluded_dataset]
    feature_df = feature_df[feature_df['Source'].isin(included_dataset)]
    feature_list = list(feature_df['Variable'].values)


    data_df = pd.read_csv("../"+version+"/data.csv")
    print(data_df.shape)
    data_df = data_df[data_df['PATNO'].isin(Study_cohort)]
    print(data_df.shape)
    data_df = data_df[data_df['EVENT_ID'].isin(vist_list + ['SC'])]
    print(data_df.shape)

    patient_length = {}
    for idx, row in visit_info_df.iterrows():
        PATNO, BL, max_visit = row['PATNO'], row['BL'], row['max_visit']
        if np.isnan(BL):  # check is there missing of BL visit, if so, exclude the patient
            print("!!!! Patient %s has no BL information!", PATNO)
        patient_length[PATNO] = vid_list[max_visit]+1

    ###############################################################################################
    # Uncomment this block to generate sequence data
    patient_arrays, feature_median, patient_feature_median = concatenate(data_df, patient_length, vid_list, feature_list)
    with open("../"+version+"/sequence_data.pkl", "wb") as wf:
        pkl.dump(patient_arrays, wf)
    with open("../"+version+"/feature_median.json", 'w') as wf:
        json.dump(feature_median, wf, indent=4)
    with open("../"+version+"/patient_feature_median.json", 'w') as wf:
        json.dump(patient_feature_median, wf, indent=4)
    with open("../"+version+"/used_features.txt", 'w') as wf:
        for var in feature_list:
            wf.write(var)
            wf.write('\n')
    ###############################################################################################


    # ###############################################################################################
    # # Uncomment the block to load existing sequence data
    with open("../"+version+"/sequence_data.pkl", 'rb') as rf:
        patient_arrays = pkl.load(rf)
    with open("../"+version+"/feature_median.json") as rf:
        feature_median = json.load(rf)
    with open("../"+version+"/patient_feature_median.json") as rf:
        patient_feature_median = json.load(rf)
        patient_feature_median = dict_key_convert.str2int(patient_feature_median)  # convert key from string to int
    feature_list = []
    with open("../"+version+"/used_features.txt") as rf:
        all_lines = rf.readlines()
        for line in all_lines:
            feature_list.append(line.strip())
    # ###############################################################################################

    # print(patient_arrays[3555])
    # print(feature_median)
    # print(patient_feature_median)
    # print(feature_list)


    ###############################################################################################
    # imputation
    # imputation_method = "interpolate"  # options: interpolate, LOCF&FOCB

    if imputation_method == "interpolate":
        print("Imputing data by pandas' interpolate.")
        patient_arrays_imp = interpolate_imputation(patient_arrays, feature_median, feature_list)

    if imputation_method == "LOCF&FOCB":
        print("Imputing data by LOCF&FOCB strategy.")
        patient_arrays_imp = LOCF_FOCB_imputation(patient_arrays, patient_feature_median, feature_list)

    with open("../"+version+"/sequence_data_%s_imputation.pkl" % imputation_method, "wb") as wf:
        pkl.dump(patient_arrays_imp, wf)
    ###############################################################################################


    ###############################################################################################
    # normalization
    # Z-score normalization
    # (value - mean) / standard deviation
    patient_arrays_imp_Zs = Z_score_normalization(patient_arrays_imp, len(feature_list))
    with open("../"+version+"/sequence_data_%s_imputation_Zscore.pkl" % imputation_method, "wb") as wf:
        pkl.dump(patient_arrays_imp_Zs, wf)

    # min-max normalization
    # (value - min) / (max - min)
    patient_arrays_imp_minmax = minmax_normalization(patient_arrays_imp, len(feature_list))
    with open("../"+version+"/sequence_data_%s_imputation_minmax.pkl" % imputation_method, "wb") as wf:
        pkl.dump(patient_arrays_imp_minmax, wf)
    ###############################################################################################



    # with open("../"+version+"/sequence_data_interpolate_imputation.pkl", 'rb') as rf:
    # with open("../"+version+"/sequence_data_interpolate_imputation_minmax.pkl", 'rb') as rf:
    # with open("../"+version+"/sequence_data_LOCF&FOCB_imputation_minmax.pkl", 'rb') as rf:
    #     patient_arrays_imp = pkl.load(rf)
    #     print(patient_arrays_imp[3000])
    #     print(len(patient_arrays_imp))
    #
    #     # To check is there NaN left
    #     for p in patient_arrays_imp:
    #         data = pd.DataFrame(patient_arrays_imp[p])
    #         for m in range(len(feature_list)):
    #             if data[m].isnull().any():
    #                 print(p, m)
    #                 print(data[m])