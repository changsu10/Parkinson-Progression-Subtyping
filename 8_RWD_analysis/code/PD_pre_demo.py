import sys
# for linux env.
sys.path.insert(0,'..')
from datetime import datetime, timedelta
import os
import pandas as pd
from tqdm import tqdm
import time
import pickle
import argparse
import csv
import utils
import numpy as np
import functools
print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--demo_file', default=r'K:\dcore-prj0166-SOURCE\wang_adrd\demographic.csv',
                        help='input demographics file with std format')
    parser.add_argument('--dx_file', default=r'K:\dcore-prj0166-SOURCE\wang_adrd\diagnosis.csv',
                        help='input diagnosis file with std format')
    # parser.add_argument('-out_file', default=r'output/patient_dates.pkl')
    args = parser.parse_args()
    return args


def read_demo(data_file, out_file=''):
    """
    :param data_file: input demographics file with std format
    :param out_file: output id_demo[patid] = (bdate,sex,  race, hispanic) pickle
    # :param patient_list: a patient ID set for whose demographic features returned.
    :return: id_demo[patid] = (bdate, sex, race, hispanic)
    :Notice:
        1.ori data: e.g:
        ssid	birth_date	birth_time	sex	hispanic	race
        463819	3/16/1967	0:00	M	NI	5
        468278	7/19/1955	0:00	M	N	NI

        df_demo['sex'].value_counts():
            F     68499
            M     57911
            UN        8
            NI        7
            OT        2
        df_demo['race'].value_counts():
            05    58811 05=White
            NI    34506 NI=No information
            03    13268 03=Black or African American
            OT    11378 OT=Other
            02     4759 02=Asian
            07     2065 07=Refuse to answer
            06     1248 06=Multiple race
            04      168 04=Native Hawaiian or Other Pacific Islander
            UN      119 UN=Unknown
            01      105 01=American Indian or Alaska Native

        df_demo['hispanic'].value_counts():
            N     77203   N=NO
            NI    32363
            Y     15913   Y=Yes
            UN      948
    """
    start_time = time.time()
    n_read = 0
    n_invalid = 0
    id_demo = {}
    with open(data_file, 'r') as f:
        col_name = next(csv.reader(f))  # read from first non-name row, above code from 2nd, wrong
        print("read from ", data_file, col_name)
        for row in csv.reader(f):
            n_read += 1
            patid = row[0]
            sex = 0 if (row[3] == 'F') else 1  # Female 0, Male, and UN, NI, OT 1
            try:
                bdate = utils.str_to_datetime(row[1])
            except:
                print('invalid birth date in ', n_read, row)
                n_invalid += 1
                continue

            hispanic = row[4]

            try:
                race = int(row[5])
            except:
                race = 0  # denote all OT, UN, NI as 0

            # if (patient_set is None) or (patid in patient_set):
            id_demo[patid] = (bdate, sex, race, hispanic)
    print('read {} rows, len(id_demo) {}'.format(n_read, len(id_demo)))
    print('n_invalid rows: ', n_invalid)
    print('read_demo done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    if out_file:
        utils.check_and_mkdir(out_file)
        pickle.dump(id_demo, open(out_file, 'wb'))

        df = pd.DataFrame(id_demo).T
        df = df.rename(columns={0: 'birth_date',
                                1: 'sex',
                                2: 'race',
                                3: 'hispanic'})
        df.to_csv(out_file.replace('.pkl', '') + '.csv')
        print('dump done to {}'.format(out_file))

    return id_demo


def build_patient_dates(demo_file, dx_file, out_file, ND_code_file, PD_drug_taken_by_patient_file):
    """
    # Start date: the first date in the EHR database
    # Initiation date: The first date when patients were diagnosed with MCI.
    # Index date:  the date of the first prescription of the assigned drug  (can not determine here. need
    #              mci_drug_taken_by_patient_from_dispensing_plus_prescribing later)
    # last date: last date of drug, or diagnosis? Use drug time in the cohort building and selection code part

    Caution:
    # First use DX_DATE, then ADMIT_DATE, then discard record
    # Only add validity check datetime(1990,1,1)<date<datetime(2030,1,1) for updating 1st and last diagnosis date

    # Input
    :param DATA_DIR:
    :return:
    """
    print("******build_patient_dates*******")
    id_demo = read_demo(demo_file)

    # 0-birth date
    # 1-start diagnosis date, 2-initial mci diagnosis date,
    # 3-first PD diagnosis date, 4-first ND diagnosis date (including fall, dementia, mental, PIGD, CI)
    # 5-first PDRD diagnosis, 6-last diagnosis
    # 7-first CI diagnosis date

    patient_dates = {pid: [val[0], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                     for pid, val in id_demo.items()}  # (pid, bdate) in zip(df_demo['PATID'], df_demo['BIRTH_DATE'])
    n_records = 0
    n_no_date = 0
    n_no_pid = 0
    min_date = datetime.max
    max_date = datetime.min
    with open(dx_file, 'r') as f:
        col_name = next(csv.reader(f))
        print('read from ', dx_file, col_name)
        # testing_num = 0
        for row in tqdm(csv.reader(f)):
            n_records += 1
            patid, date, dx = row[0], row[3], row[5]
            # dx_type = row[6]

            # First use ADMIT_DATE , then DX_DATE, then discard record
            # if (date == '') or (date == 'NULL'):
            #     n_no_date += 1
            #     continue
            # else:
            #     date = utils.str_to_datetime(date)
            try:
                date = utils.str_to_datetime(date)
            except:
                print('invalid date in ', n_records, row)
                n_no_date += 1
                continue

            if date > max_date:
                max_date = date
            if date < min_date:
                min_date = date

            if patid not in patient_dates:
                n_no_pid += 1
                print(patid, ' patients not in demographic files')
                continue

            # 1-start diagnosis date
            if pd.isna(patient_dates[patid][1]) or date < patient_dates[patid][1]:
                if datetime(1990, 1, 1) < date < datetime(2030, 1, 1):
                    patient_dates[patid][1] = date

            #  2-initial mci diagnosis date
            if utils.is_mci(dx):
                if pd.isna(patient_dates[patid][2]) or date < patient_dates[patid][2]:
                    patient_dates[patid][2] = date - timedelta(days=182) # date -> date -0.5 year

            # 3-firs PD diagnosis date
            if utils.is_PD(dx):
                if pd.isna(patient_dates[patid][3]) or date < patient_dates[patid][3]:
                    patient_dates[patid][3] = date - timedelta(days=182) # date -> date -0.5 year

            # 4-first ND code diagnosis date
            if utils.is_ND(dx,ND_code_file):
                if pd.isna(patient_dates[patid][4]) or date < patient_dates[patid][4]:
                    patient_dates[patid][4] = date

            # 5-first PD or PD_related diagnosis date
            if utils.is_PD(dx) or utils.is_PDRD(dx):
                if pd.isna(patient_dates[patid][5]) or date < patient_dates[patid][5]:
                    patient_dates[patid][5] = date - timedelta(days=182) # date -> date -0.5 year

            # 6-last diagnosis
            if pd.isna(patient_dates[patid][6]) or date > patient_dates[patid][6]:
                if datetime(1990, 1, 1) < date < datetime(2030, 1, 1):
                    patient_dates[patid][6] = date

            # 7-first CI diagnosis date
            if utils.is_CI(dx):
                if pd.isna(patient_dates[patid][7]) or date < patient_dates[patid][7]:
                    patient_dates[patid][7] = date - timedelta(days=182) # date -> date -0.5 year

            # testing_num = testing_num + 1
            # if testing_num>1000:
            #     break

    print('len(patient_dates)', len(patient_dates))
    print('n_records', n_records)
    print('n_no_date', n_no_date)
    print('n_no_pid', n_no_pid)

    # utils.check_and_mkdir(out_file)
    # pickle.dump(patient_dates, open(out_file, 'wb'))

    df = pd.DataFrame(patient_dates).T
    df = df.rename(columns={0: 'birth_date',
                            1: '1st_diagnosis_date',
                            2: '1st_mci_date',
                            3: '1st_PD_date',
                            4: '1st_ND_date',
                            5: '1st_PDRD_date',
                            6: 'last_diagnosis_date',
                            7: '1st_CI_date'})

    # consider the date of drug levodopa: the earliest levodopa prescription within the year preceding the first PD diagnosis
    for i, row in df.iterrows():
        patid = i # '468278'
        first_PD_date = row['1st_PD_date']
        first_PDRD_date = row['1st_PDRD_date']
        first_CI_date = row['1st_CI_date']
        if pd.notna(first_PD_date):
            date = first_PD_date
            try:
                drug_levodopa = PD_drug_taken_by_patient_file['6375']  # 6375--levodopa
                p_levodopa = drug_levodopa[patid]  # patid
                tem_date = [date]
                for p_l in p_levodopa:
                    p_l_date = pd.to_datetime(p_l[0])
                    p_l_date = p_l_date - timedelta(days=182) # date -> date -0.5 year
                    if (date - p_l_date).days >= 0 and (date - p_l_date).days <= 365:
                        tem_date.append(p_l_date)
                date = min(tem_date)
            except:
                date = date
                print('no levodopa for patient:', patid)

            # for subtype III, compare the PD date and CI date
            p_dates = patient_dates[patid]
            if pd.notna(first_CI_date):
                if (date - first_CI_date).days >= 0 or (first_CI_date-date).days <= 365:
                    df.loc[i, '1st_PD_date'] =first_CI_date #first_CI_date,date
                    p_dates[3] = first_CI_date#first_CI_date,date
                else:
                    df.loc[i, '1st_PD_date'] = np.nan
                    p_dates[3] = np.nan
            else:
                df.loc[i, '1st_PD_date'] = np.nan
                p_dates[3] = np.nan
            patient_dates[patid] = p_dates

        if pd.notna(first_PDRD_date):
            date = first_PDRD_date
            try:
                drug_levodopa = PD_drug_taken_by_patient_file['6375']  # 6375--levodopa
                p_levodopa = drug_levodopa[patid]  # patid
                tem_date = [date]
                for p_l in p_levodopa:
                    p_l_date = pd.to_datetime(p_l[0])
                    p_l_date = p_l_date - timedelta(days=182)
                    if (date - p_l_date).days >= 0 and (date - p_l_date).days <= 365:
                        tem_date.append(p_l_date)
                date = min(tem_date)
            except:
                date = date
                print('no levodopa for patient:', patid)

            # for subtype III, compare the PD date and CI date
            p_dates = patient_dates[patid]
            if pd.notna(first_CI_date):
                if (date - first_CI_date).days >= 0 or (first_CI_date-date).days <= 365:
                    df.loc[i, '1st_PDRD_date'] = first_CI_date#first_CI_date,,date
                    p_dates[5] = first_CI_date#first_CI_date,date
                else:
                    df.loc[i, '1st_PDRD_date'] = np.nan
                    p_dates[5] = np.nan
            else:
                df.loc[i, '1st_PDRD_date'] = np.nan
                p_dates[5] = np.nan
            patient_dates[patid] = p_dates

    # idx_ADRD = pd.notna(df['1st_dementia_date'])
    # df.loc[idx_ADRD, 'MCI<ADRD'] = df.loc[idx_ADRD, '1st_mci_date'] < df.loc[idx_ADRD, '1st_dementia_date']

    idx_PD = pd.notna(df['1st_PD_date'])
    df.loc[idx_PD, 'PD_age'] = (df.loc[idx_PD, '1st_PD_date'] - df.loc[idx_PD, 'birth_date']).dt.days//365

    idx_PDRD = pd.notna(df['1st_PDRD_date'])
    df.loc[idx_PDRD, 'PDRD_age'] = (df.loc[idx_PDRD, '1st_PDRD_date'] - df.loc[idx_PDRD, 'birth_date']).dt.days//365

    idx_diff_PD_CI = pd.notna(df['1st_PD_date'])
    df.loc[idx_diff_PD_CI, 'diff_PD_CI'] = (df.loc[idx_diff_PD_CI, '1st_PD_date'] - df.loc[idx_diff_PD_CI, '1st_CI_date']).dt.days//365

    idx_diff_PDRD_CI = pd.notna(df['1st_PDRD_date'])
    df.loc[idx_diff_PDRD_CI, 'diff_PDRD_CI'] = (df.loc[idx_diff_PDRD_CI, '1st_PDRD_date'] - df.loc[idx_diff_PDRD_CI, '1st_CI_date']).dt.days//365

    idx_RD = pd.notna(df['1st_ND_date'])
    df.loc[idx_RD, 'PD<ND'] = df.loc[idx_RD, '1st_PD_date'] < df.loc[idx_RD, '1st_ND_date']
    df.loc[idx_RD, 'PDRD<ND'] = df.loc[idx_RD, '1st_PDRD_date'] < df.loc[idx_RD, '1st_ND_date']

    # save patient_dates to pickle
    utils.check_and_mkdir(out_file)
    pickle.dump(patient_dates, open(out_file, 'wb'))

    df.to_csv(out_file.replace('.pkl', '') + '.csv')
    print('dump done')
    return patient_dates


def plot_MCI_to_ADRD():
    import matplotlib.pyplot as plt
    import pandas as pd
    pdates = pd.read_csv(os.path.join('debug', 'patient_dates.csv'))
    idx = pd.notna(pdates['1st_dementia_date'])
    pt = pd.to_datetime(pdates.loc[idx, '1st_dementia_date']) - pd.to_datetime(pdates.loc[idx, '1st_PD_date'])
    pt.apply(lambda x: x.days/365).hist()  # bins=25
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    print(args)
    id_demo = read_demo(args.demo_file, r'output/patient_demo.pkl')
    with open(r'K:\WorkArea-zhx2005\PycharmProjects\PD_3_subtype\pickles\ND_code.pkl', 'rb') as f:
        ND_code = pickle.load(f)

    with open(r'K:\WorkArea-zhx2005\PycharmProjects\PD_3_subtype\output\PD_drug_taken_by_patient.pkl', 'rb') as f:
        PD_drug_taken_by_patient = pickle.load(f)

    patient_dates = build_patient_dates(args.demo_file, args.dx_file, r'output/patient_dates.pkl', ND_code, PD_drug_taken_by_patient)
    # plot_MCI_to_ADRD()
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
