"""
To plot distribution of visit interval.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

version = "processed_data_2019-9-30"


info_df = pd.read_csv("../"+version+"/patient_visit_info.csv")


patient_df = pd.read_csv("../"+version+"/patient_info.csv")
entire_cohort = list(patient_df.PATNO.values)
PD_cohort = list(patient_df[patient_df['ENROLL_CAT'] == 'PD'].PATNO.values)
HC_cohort = list(patient_df[patient_df['ENROLL_CAT'] == 'HC'].PATNO.values)
SWEDD_cohort = list(patient_df[patient_df['ENROLL_CAT'] == 'SWEDD'].PATNO.values)
Other_cohort = [i for i in entire_cohort if i not in PD_cohort+HC_cohort+SWEDD_cohort]

print(len(entire_cohort), len(PD_cohort), len(SWEDD_cohort), len(HC_cohort), len(Other_cohort))


# load interval information

info_df = info_df[info_df['PATNO'].isin(PD_cohort+HC_cohort+SWEDD_cohort)].reset_index(drop=True)

N, M = info_df.shape

interval_info = {}
for v in range(16):
    interval_info[v] = []

for n in range(N):
    for m in range(16):
        t, t_next = info_df.iloc[n, m+1], info_df.iloc[n, m+2]
        if np.isnan(t) or np.isnan(t_next):
            continue
        else:
            if round((t_next - t) / 60 / 60 / 24 / 30) < 0:
                continue
            interval_info[m].append( round((t_next - t) / 60 / 60 / 24 / 30) )  # month (rounding)

# plot histogram
plt.subplots(figsize=(15, 14))
for i in interval_info:
    plt.subplot(4, 4, i+1)
    data = interval_info[i]
    max_interval, min_interval = max(data), min(data)
    bins = np.arange(min_interval, max_interval, 1)
    plt.hist(data)
    plt.xlim(min_interval, max_interval)
    plt.title("V%2d to V%2d" % (i, i+1))
plt.savefig("../"+version+"/visit_interval_distribution.png", dpi=300)

# datatype = {'PATNO':int}
# data_df = pd.read_csv("../data_processed/data_20190910.csv", dtype=datatype)
#
# data_df = data_df[data_df["PATNO"]==3354]
# time = set(data_df[data_df["EVENT_ID"].isin(["V10"])]["Time"].values)
# print(time)
# data_df = data_df[data_df["EVENT_ID"].isin(["V08"])]
# print(data_df[data_df["Time"]=="04/2014"])


# distribution of length of patient longitudinal information
plt.subplots(figsize=(10, 20))

plt.subplot(4, 1, 1)
PD_info = info_df[info_df["PATNO"].isin(PD_cohort)]
plot_PD = PD_info.reset_index().groupby("max_visit").size().plot(kind='bar', title="PD cohort", rot=0, ax=plt.gca())
plot_PD.set_ylabel("Number of Participants")
# plot_PD.set_xlabel("Max visit")

plt.subplot(4, 1, 2)
SWEDD_info = info_df[info_df["PATNO"].isin(SWEDD_cohort)]
plot_SWEDD = SWEDD_info.reset_index().groupby("max_visit").size().plot(kind='bar', title="SWEDD cohort", rot=0, ax=plt.gca())
plot_SWEDD.set_ylabel("Number of Participants")
# plot_SWEDD.set_xlabel("Max visit")

plt.subplot(4, 1, 3)
HC_info = info_df[info_df["PATNO"].isin(HC_cohort)]
plot_HC = HC_info.reset_index().groupby("max_visit").size().plot(kind='bar', title="HC cohort", rot=0, ax=plt.gca())
plot_HC.set_ylabel("Number of Participants")
# plot_HC.set_xlabel("Max visit")

plt.subplot(4, 1, 4)
study_info = info_df[info_df["PATNO"].isin(PD_cohort+SWEDD_cohort+HC_cohort)]
plot_total = study_info.reset_index().groupby("max_visit").size().plot(kind='bar', title="Study cohort (PD + SWEDD + HC)", rot=0, ax=plt.gca())
plot_total.set_ylabel("Number of Participants")
plot_total.set_xlabel("Max visit")

plt.savefig("../"+version+"/longitudinal_distribution.png", dpi=300)