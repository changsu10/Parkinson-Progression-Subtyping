import pandas as pd
import numpy as np
import matplotlib as plt
import time


def extract_longitudinal_info(version):
    datatype = {'PATNO':int, 'diagnosis':int}
    patient_df = pd.read_csv("../"+version+"/patient_info.csv", dtype=datatype)
    datatype = {'PATNO':int}
    data_df = pd.read_csv("../"+version+"/data.csv", dtype=datatype)

    entire_cohort = list(patient_df.PATNO.values)
    PD_cohort = list(patient_df[patient_df['ENROLL_CAT'] == 'PD'].PATNO.values)
    HC_cohort = list(patient_df[patient_df['ENROLL_CAT'] == 'HC'].PATNO.values)
    SWEDD_cohort = list(patient_df[patient_df['ENROLL_CAT'] == 'SWEDD'].PATNO.values)
    Other_cohort = [i for i in entire_cohort if i not in PD_cohort+HC_cohort+SWEDD_cohort]

    print(len(entire_cohort),len(PD_cohort),len(HC_cohort),len(SWEDD_cohort),len(Other_cohort))


    info_df = pd.DataFrame(entire_cohort, columns=['PATNO'])

    vist_list = ['BL'] + ["V%02d" % i for i in range(1, 17)]

    info_df = info_df.reindex(columns=['PATNO']+vist_list+['max_visit'])

    info_df.set_index(['PATNO'], inplace = True)

    print(data_df.shape)
    data_df = data_df[data_df['PATNO'].isin(entire_cohort)]
    print(data_df.shape)
    data_df = data_df[data_df['EVENT_ID'].isin(vist_list)]
    print(data_df.shape)

    # data_df['Time'] = pd.to_datetime(data_df['Time'])
    print(data_df.info())


    i = 0
    for idx, row in data_df.iterrows():
        PATNO, EVENT_ID, Var_Type, Time = row['PATNO'], row['EVENT_ID'], row['Var_Type'], row['Time']
        i += 1
        timestamp = time.mktime(time.strptime(Time, "%m/%Y"))

        # if PATNO != 3401:
        #     continue
        if Var_Type == 'medical':
            continue
        if np.isnan(info_df.loc[PATNO, EVENT_ID]):
            info_df.loc[PATNO, EVENT_ID] = timestamp
        else:
            if info_df.loc[PATNO, EVENT_ID] > timestamp:
                info_df.loc[PATNO, EVENT_ID] = timestamp

    info_df = info_df.reset_index()

    for idx, row in info_df.iterrows():
        max_visit = None
        for v in vist_list:
            if not np.isnan(row[v]):
                max_visit = v
            info_df.loc[idx, "max_visit"] = max_visit

    info_df.to_csv("../"+version+"/patient_visit_info.csv", index=False)

