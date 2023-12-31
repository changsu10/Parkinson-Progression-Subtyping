import sys

# for linux env.
sys.path.insert(0, '..')
import argparse
import os
import time
from PD_std_eligibility_screen import exclude
from PD_std_pre_cohort_rx import pre_user_cohort_rx_v2
from PD_std_pre_cohort_dx import get_user_cohort_dx
from PD_std_user_cohort import pre_user_cohort_triplet
import pickle
import utils
import json
import pandas as pd
from tqdm import tqdm
import functools

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # use default value
    parser.add_argument('--min_patients', default=20, type=int,
                        help='minimum number of patients for each cohort. [Default 20]')  # 500, 100
    parser.add_argument('--min_age', default=50, type=int, help='minimum age at initiation date [Default 50 years old]')
    # Selection criterion: Value to play with
    parser.add_argument('--min_prescription', default=1, type=int, # 4, 3, 2
                        help='minimum times of prescriptions of each patient in each drug trial.')
    parser.add_argument('--exposure_interval', default=30, type=int, # 730, 360, 180 30
                        help='Drug exposure period, minimum time interval for the first and last prescriptions')
    parser.add_argument('--followup', default=730, type=int,
                        help='number of days of followup period, to define outcome')
    parser.add_argument('--baseline', default=365, type=int,
                        help='number of days of baseline period, to collect covariates')
    parser.add_argument('--index_minus_init_min', default=0, type=int, # 0, -30,-90   -999999
                        help='min <= (index_date - initiation_date).days <= max')
    parser.add_argument('--index_minus_init_max', default=99999999999999, type=int,
                        help='min <= (index_date - initiation_date).days <= max')
    parser.add_argument('--adrd_minus_index_min', default=0, type=int,
                        help='min bound <= (first_adrd_date - index_date).days')
    parser.add_argument('--drug_coding', choices=['rxnorm', 'gpi'], default='rxnorm')
    # others: folder and encodes
    parser.add_argument('--dx_file', default=r'K:\dcore-prj0166-SOURCE\wang_adrd\diagnosis.csv',
                        help='input diagnosis file with std format')
    parser.add_argument('--save_cohort_all', default='output/save_cohort_all/')
    parser.add_argument('--dx_coding', choices=['ccs', 'ccw'], default='ccw')

    # Deprecated
    # parser.add_argument('--demo_file', default=r'../data/florida/demographic.csv',
    #                     help='input demographics file with std format')
    # parser.add_argument('--input_data', default='../data/florida')
    # parser.add_argument('--pickles', default='pickles')
    # parser.add_argument('--outcome_icd9', default='outcome_icd9.txt', help='outcome definition with ICD-9 codes')
    # parser.add_argument('--outcome_icd10', default='outcome_icd10.txt', help='outcome definition with ICD-10 codes')
    args = parser.parse_args()
    return args


def get_patient_list(min_patient, prescription_taken_by_patient):
    # logics: maybe should return drug list?
    print('get_patient_list...min_patient:', min_patient)
    patients_list = set()
    for drug, patients in prescription_taken_by_patient.items():
        if len(patients) >= min_patient:
            for patient in patients:
                patients_list.add(patient)
    print('in get_patient_list, len(patients_list): ', len(patients_list))
    return patients_list


if __name__ == '__main__':
    start_time = time.time()
    # main(args=parse_args())
    args = parse_args()
    print(args)
    eligibility_criteria = {'min_age': args.min_age,
                            'min_prescription': args.min_prescription,
                            'exposure_interval': args.exposure_interval,
                            'followup': args.followup,
                            'baseline': args.baseline,
                            'index_minus_init_min': args.index_minus_init_min,
                            'index_minus_init_max': args.index_minus_init_max,
                            'adrd_minus_index_min': args.adrd_minus_index_min,
                            }
    dirname = os.path.dirname(args.save_cohort_all)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(args.save_cohort_all + r'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print(json.dumps(args.__dict__, sort_keys=False, indent=2))

    # with open(args.save_cohort_all+r'/commandline_args.txt', 'r') as f:
    #     args.__dict__ = json.load(f)

    print('**********Loading prescription data**********')
    # mci_drug_taken_by_patient_from_dispensing.pkl
    mci_prescription_taken_by_patient = pickle.load(
        open(os.path.join('output', 'PD_drug_taken_by_patient.pkl'), 'rb'))# mci_drug_taken_by_patient.pkl --> PD_drug_taken_by_patient.pkl

    print('**********Loading ND code list including "fall, dementia, mental, PIGD, CI" **********')
    ND_code = pickle.load(open(os.path.join('pickles', 'ND_code.pkl'), 'rb'))

    print('**********Loading patient demo dates**********')
    patient_dates = pickle.load(open(os.path.join('output', 'patient_dates.pkl'), 'rb'))
    patient_demo = pickle.load(open(os.path.join('output', 'patient_demo.pkl'), 'rb'))

    print('**********Loading icd to ccs/ccw code**********')
    icd9_to_ccs = pickle.load(open(os.path.join('pickles', 'icd9_to_ccs.pkl'), 'rb'))
    icd10_to_ccs = pickle.load(open(os.path.join('pickles', 'icd10_to_ccs.pkl'), 'rb'))
    icd_to_ccw, icd_to_ccwname, ccw_info = utils.load_icd_to_ccw('mapping/CCW_to_use_enriched.json')
    with open('mapping/CCW_AD_comorbidity.json') as f:
        ccw_ad_comorbidity = json.load(f)
    ccwcomorbidityid_name = {}
    for name, v in ccw_ad_comorbidity.items():
        ccwcomorbidityid_name[ccw_info[0][name]] = name

    atc2rxnorm = pickle.load(open(os.path.join('pickles', 'atcL2_rx.pkl'), 'rb'))  # ATC2DRUG.pkl
    if args.drug_coding.lower() == 'gpi':
        print('using GPI drug coding')
        is_antidiabetic = lambda x: (x[:2] == '27')
        is_antihypertensives = lambda x: (x[:2] == '36')
        drug_name_cnt = pickle.load(open(os.path.join('output', '_gpi_ingredients_nameset_cnt.pkl'), 'rb'))
        drug_name = {k: v[0] for k, v in drug_name_cnt.items()}
    else:
        print('using default rxnorm_cui drug coding')
        is_antidiabetic = lambda x: x in atc2rxnorm['A10']
        is_antihypertensives = lambda x: x in atc2rxnorm['C02']
        from PD_std2_pre_drug import load_latest_rxnorm_info

        drug_name, _ = load_latest_rxnorm_info()

    print('**********Preprocessing patient data**********')
    # Input: drug --> patient --> [(date, supply day),]     patient_dates: patient --> [birth_date, other dates]
    # Output: save_prescription: drug --> patients --> [date1, date2, ...] sorted
    #         save_patient: patient --> drugs --> [date1, date2, ...] sorted
    # Notes: 20210608: not using followup here, define time_interval as time between first and last prescriptions
    #        20210706: save_patient for baseline building, use all patient-drug info, instead of previous cohort-included drugs only.

    save_prescription, save_patient = exclude(mci_prescription_taken_by_patient, patient_dates, eligibility_criteria)

    # args.time_interval, args.followup, args.baseline, args.min_prescription)

    # Patient set after exclusion (patients who take drugs which have >= min_patients patients)
    patient_list = get_patient_list(args.min_patients, save_prescription)

    # Should I only calculate the (rx, dx) covariates in the baseline period within only baseline-time windows?
    # drug --> patient --> dates --> prescription list in the Baseline
    save_cohort_rx = pre_user_cohort_rx_v2(save_prescription, save_patient, args.min_patients)

    # drug --> patient --> dates --> diagnosis list in the Baseline
    # _user_dx: patient-->dates-->diagnosis list for patients in patient_list, for debug, not used
    # save_cohort_outcome: event type --> patient --> event date list
    # 2021-06-14 Only using AD, not using dementia: #{'AD': utils.is_AD, 'dementia': utils.is_dementia}
    # 2021-07-08 integrate pre_user_cohort_outcome into get_user_cohort_dx, scan diagnosis file once
    # # save_cohort_outcome = pre_user_cohort_outcome(args.dx_file, patient_list, {'AD': utils.is_AD})
    save_cohort_dx, _user_dx, save_cohort_outcome = get_user_cohort_dx(args.dx_file, save_prescription, icd9_to_ccs,
                                                                       icd10_to_ccs,
                                                                       icd_to_ccw, args.dx_coding, args.min_patients,
                                                                       patient_list,
                                                                       # {'AD': utils.is_AD}, # for AD cohort
                                                                       {'ND': utils.is_ND},  # for PD cohort, this is outcome of PD patients
                                                                       ND_code
                                                                       )
    # patient --> demo feature tuple
    save_cohort_demo = patient_demo  # read_demo(args.demo_file, patient_list)

    print('**********Generating patient cohort**********')
    # for each drug, dump (patients, [rx_codes, dx_codes, demo], outcome)
    # rx_codes, dx_codes: ordered list of varying-length list of codes (drug ingredient, ccs codes), not dates info kept
    pre_user_cohort_triplet(save_prescription, save_cohort_rx, save_cohort_dx,
                            save_cohort_outcome, save_cohort_demo, args.save_cohort_all,
                            patient_dates, args.followup, drug_name,
                            ccwcomorbidityid_name,
                            {'antidiabetic': is_antidiabetic,
                             'antihypertensives': is_antihypertensives})  # last 2 are new added
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
