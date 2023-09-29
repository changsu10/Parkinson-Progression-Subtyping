import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_patient_info(version, output_dir):
    cols = {} # column names
    cols["pd_features"] = ["PATNO", "SXMO","SXYEAR", "PDDXDT", "INFODT", "EVENT_ID"] # first symptom onset month, year, diagnosis date, SC event date
    # Subject Characteristics
    cols["family_history"] = ["PATNO", "BIOMOM", "BIOMOMPD", "BIODAD", "BIODADPD", "FULSIB", "FULSIBPD", "HAFSIB", "HAFSIBPD", "MAGPAR", "MAGPARPD", "PAGPAR", "PAGPARPD", "MATAU", "MATAUPD", "PATAU", "PATAUPD", "KIDSNUM", "KIDSPD"]
    cols["status"] = ["PATNO", "RECRUITMENT_CAT", "IMAGING_CAT", "ENROLL_DATE", "ENROLL_CAT"]
    cols["screening"] = ["PATNO", "BIRTHDT", "GENDER", "APPRDX", "CURRENT_APPRDX", "HISPLAT", "RAINDALS", "RAASIAN", "RABLACK", "RAHAWOPI", "RAWHITE", "RANOS", "PRJENRDT"]
    cols["socio"] = [ "PATNO", "EDUCYRS", "HANDED" ]
    # Subject Enrollment
    cols["primary_diag"] = [ "PATNO", "PRIMDIAG" ]


    # --- Medical ---
    # first symptom onset month, year, diagnosis date
    pd_start = pd.read_csv("../"+version+"/_Medical_History/ALL_Medical/PD_Features.csv", usecols=cols["pd_features"])

    # --- Subject Characteristics ---
    family_history = pd.read_csv("../"+version+"/_Subject_Characteristics/ALL_Family_History/Family_History__PD_.csv", index_col=["PATNO"], usecols=cols["family_history"])

    status = pd.read_csv("../"+version+"/_Subject_Characteristics/ALL_Patient_Status/Patient_Status.csv", index_col=["PATNO"], usecols=cols["status"])

    screening = pd.read_csv("../"+version+"/_Subject_Characteristics/ALL_Subject_Demographics/Screening___Demographics.csv", index_col=["PATNO"], usecols=cols["screening"])

    socio = pd.read_csv("../"+version+"/_Subject_Characteristics/ALL_Subject_Demographics/Socio-Economics.csv", index_col=["PATNO"], usecols=cols["socio"])

    # --- Subject Enrollment ---
    primary_diag = pd.read_csv("../"+version+"/_Enrollment/ALL_Subject_Enrollment/Primary_Diagnosis.csv")



    # Exatract cases and controls
    patients = {}
    for idx, row in primary_diag.iterrows():
        PATNO, PRIMDIAG = row['PATNO'], row['PRIMDIAG']
        if PATNO not in patients:
            patients[PATNO] = PRIMDIAG
        else:
            if patients[PATNO] == 1:
                continue
            else:
                patients[PATNO] = PRIMDIAG

    patient_df = pd.DataFrame.from_dict(patients, orient="index", columns=['diagnosis'])
    print("Patient distribution: ")
    print(patient_df.diagnosis.reset_index().groupby("diagnosis").size())
    patient_df['Symptom_date'] = -1
    patient_df['Diagnosis_date'] = -1

    # load date of symptom and diagnosis
    pd_start = pd_start.fillna(value={'SXMO':1})
    pd_start = pd_start.fillna(value={'PDDXDT':-1})
    for idx, row in pd_start.iterrows():
        PATNO, SXMO, SXYEAR, PDDXDT = row['PATNO'], row['SXMO'], row['SXYEAR'], row['PDDXDT']
        if PATNO not in patients:
            continue
        if not np.isnan(SXYEAR):
            symptom_date = SXYEAR
            if patient_df.loc[PATNO, "Symptom_date"] == -1 or patient_df.loc[PATNO, "Symptom_date"] > symptom_date:
                patient_df.loc[PATNO, "Symptom_date"] = symptom_date

        if PDDXDT != -1:
            diagnosis_date = int(PDDXDT[-4:])
            if patient_df.loc[PATNO, "Diagnosis_date"] == -1 or patient_df.loc[PATNO, "Diagnosis_date"] > diagnosis_date:
                patient_df.loc[PATNO, "Diagnosis_date"] = diagnosis_date

    patient_df['PATNO'] = patient_df.index
    print(patient_df)

    print(pd_start[["PATNO", "INFODT"]])
    pd_start_sc_only = pd_start[["PATNO", "INFODT", "EVENT_ID"]]
    pd_start_sc_only = pd_start_sc_only[pd_start_sc_only['EVENT_ID'] == 'SC'][["PATNO", "INFODT"]]
    print(pd_start_sc_only)


    patient_df = pd.merge(patient_df, pd_start_sc_only, how='left', on="PATNO")

    print(patient_df)

    # load demographics
    screening.reset_index().drop_duplicates(['PATNO'])
    # patient_df['PATNO'] = patient_df.index
    patient_df.reset_index(drop=True, inplace=True)
    patient_df = patient_df.reindex(columns=['PATNO', 'diagnosis', 'Symptom_date', 'Diagnosis_date', 'INFODT'])
    patient_df = pd.merge(patient_df, screening, on="PATNO")

    patient_df['INFODT'] = patient_df['INFODT'].fillna('None')
    patient_df['PRJENRDT'] = patient_df['PRJENRDT'].fillna('None')

    print(patient_df)

    # compute age at symptom and diagnosis
    patient_df['Age_at_symptom'] = -1
    patient_df['Age_at_diagnosis'] = -1
    patient_df['Age_at_baseline'] = np.nan
    for idx, row in patient_df.iterrows():
        PATNO, Symptom_date, Diagnosis_date, BIRTHDT, INFODT = row['PATNO'], row['Symptom_date'], row['Diagnosis_date'], row['BIRTHDT'], row['PRJENRDT']
        if Symptom_date != -1:
            age_at_symptom = Symptom_date - BIRTHDT
            patient_df.loc[idx, 'Age_at_symptom'] = age_at_symptom
        if Diagnosis_date != -1:
            age_at_diagnosis = Diagnosis_date - BIRTHDT
            patient_df.loc[idx, 'Age_at_diagnosis'] = age_at_diagnosis

        if INFODT == 'None':
            patient_df.loc[idx, 'Age_at_baseline'] = np.nan
        else:
            patient_df.loc[idx, 'Age_at_baseline'] = int(INFODT[-4:]) - BIRTHDT


    socio = socio.reset_index()
    patient_df = pd.merge(patient_df, socio, on="PATNO")

    status = status.reset_index()
    patient_df = pd.merge(patient_df, status, on="PATNO")

    # print(patient_df)
    # print(patient_df.loc[3403, 'diagnosis'])
    # print(patient_df.isnull().sum())

    patient_df.to_csv("../"+output_dir+"/patient_info.csv", index=False)
    #
    #
    # cases = list(set(patient_df[patient_df['ENROLL_CAT']=='PD'].PATNO))
    # controls = list(set(patient_df[patient_df['ENROLL_CAT']=='HC'].PATNO))
    # SWEDD = list(set(patient_df[patient_df['ENROLL_CAT']=='SWEDD'].PATNO))
    # print(len(cases))
    # print(len(controls))
    # print(len(SWEDD))