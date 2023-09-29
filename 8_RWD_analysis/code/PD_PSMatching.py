import math
import sys

import numpy as np

# for linux env.
sys.path.insert(0, '..')
import time
from dataset import *
import pickle
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from evaluation import *
import torch.nn.functional as F
import os
from utils import save_model, load_model, check_and_mkdir
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
from utils import load_icd_to_ccw
from PSModels import mlp, lstm, ml
import itertools
import functools
print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--data_dir', type=str, default='output/save_cohort_all/')

    parser.add_argument('--treated_drug', type=str, default='6809') # #6809,6135,704
    parser.add_argument('--controlled_drug', choices=['atc', 'random'], default='atc')
    parser.add_argument('--controlled_drug_ratio', type=int, default=10)  # 2 seems not good. keep unchanged
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument('--run_model', choices=['LSTM', 'LR', 'MLP', 'XGBOOST', 'LIGHTGBM'], default='LR')
    parser.add_argument('--med_code_topk', type=int, default=200)
    parser.add_argument('--drug_coding', choices=['rxnorm', 'gpi'], default='rxnorm')
    parser.add_argument('--stats', action='store_true', default=True)
    parser.add_argument('--stats_exit', action='store_true')
    # Deep PSModels
    parser.add_argument('--batch_size', type=int, default=256)  #768)  # 64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 0.001
    parser.add_argument('--weight_decay', type=float, default=1e-6)  # )0001)
    parser.add_argument('--epochs', type=int, default=15)  # 30
    # LSTM
    parser.add_argument('--diag_emb_size', type=int, default=128)
    parser.add_argument('--med_emb_size', type=int, default=128)
    parser.add_argument('--med_hidden_size', type=int, default=64)
    parser.add_argument('--diag_hidden_size', type=int, default=64)
    parser.add_argument('--lstm_hidden_size', type=int, default=100)
    # MLP
    parser.add_argument('--hidden_size', type=str, default='', help=', delimited integers')
    # Output
    parser.add_argument('--output_dir', type=str, default='output/')

    # discarded
    # parser.add_argument('--save_db', type=str)
    # parser.add_argument('--outcome', choices=['bool', 'time'], default='bool')
    # parser.add_argument('--pickles_dir', type=str, default='pickles/')
    # parser.add_argument('--hidden_size', type=int, default=100)
    # parser.add_argument('--save_model_filename', type=str, default='tmp/1346823.pt')
    args = parser.parse_args()

    # Modifying args
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.random_seed >= 0:
        rseed = args.random_seed
    else:
        from datetime import datetime
        rseed = datetime.now()
    args.random_seed = rseed
    args.save_model_filename = os.path.join(args.output_dir, args.treated_drug,
                                            args.treated_drug + '_S{}D{}C{}_{}'.format(args.random_seed,
                                                                                       args.med_code_topk,
                                                                                       args.controlled_drug,
                                                                                       args.run_model))
    check_and_mkdir(args.save_model_filename)

    args.hidden_size = [int(x.strip()) for x in args.hidden_size.split(',')
                        if (x.strip() not in ('', '0'))]
    if args.med_code_topk < 1:
        args.med_code_topk = None

    return args


# flaten series into static
# train_x, train_t, train_y = flatten_data(my_dataset, train_indices)  # (1764,713), (1764,), (1764,)
def flatten_data(mdata, data_indices, verbose=1):
    x, t, y = [], [], []
    for idx in data_indices:
        confounder, treatment, outcome = mdata[idx][0], mdata[idx][1], mdata[idx][2]
        dx, rx, sex, age, days = confounder[0], confounder[1], confounder[2], confounder[3], confounder[4]
        dx, rx = np.sum(dx, axis=0), np.sum(rx, axis=0)
        dx = np.where(dx > 0, 1, 0)
        rx = np.where(rx > 0, 1, 0)

        x.append(np.concatenate((dx, rx, [sex], [age], [days])))
        t.append(treatment)
        y.append(outcome)

    x, t, y = np.asarray(x), np.asarray(t), np.asarray(y)
    if verbose:
        d1 = len(dx)
        d2 = len(rx)
        print('...dx:', x[:, :d1].shape, 'non-zero ratio:', (x[:, :d1] != 0).mean(), 'all-zero:',
              (x[:, :d1].mean(0) == 0).sum())
        print('...rx:', x[:, d1:d1 + d2].shape, 'non-zero ratio:', (x[:, d1:d1 + d2] != 0).mean(), 'all-zero:',
              (x[:, d1:d1 + d2].mean(0) == 0).sum())
        print('...all:', x.shape, 'non-zero ratio:', (x != 0).mean(), 'all-zero:', (x.mean(0) == 0).sum())
    return x, t, y


def _evaluation_helper(X, T, PS_logits, loss):
    y_pred_prob = logits_to_probability(PS_logits, normalized=False)
    auc = roc_auc_score(T, y_pred_prob)
    max_smd, smd, max_smd_weighted, smd_w = cal_deviation(X, T, PS_logits, normalized=False, verbose=False)
    n_unbalanced_feature = len(np.where(smd > SMD_THRESHOLD)[0])
    n_unbalanced_feature_weighted = len(np.where(smd_w > SMD_THRESHOLD)[0])
    result = (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
    return result


def _loss_helper(v_loss, v_weights):
    return np.dot(v_loss, v_weights) / np.sum(v_weights)


def compute_SMD_HR(df_ori,feature_list):
    df = df_ori.copy()
    balanced_feature_tem = []
    unbalanced_feature_tem = []
    unbalanced_feature_tem_SMD = []
    unbalanced_feature_tem_SMD_max = 0
    for fea in list(feature_list):
        # print('feature',fea)
        hidden_treated = df.loc[df['treatment'] == 1, fea].values
        hidden_controlled = df.loc[df['treatment'] == 0, fea].values

        hidden_treated_mu, hidden_treated_var = np.mean(hidden_treated), np.var(hidden_treated)
        # t1, t2 = hidden_treated.mean(), hidden_treated.var()
        hidden_controlled_mu, hidden_controlled_var = np.mean(hidden_controlled), np.var(hidden_controlled)
        # t3, t4 = hidden_controlled.mean(), hidden_controlled.var()
        VAR = np.sqrt((hidden_treated_var + hidden_controlled_var) / 2)
        hidden_deviation = np.abs(hidden_treated_mu - hidden_controlled_mu) / VAR
        # hidden_deviation = (hidden_treated_mu - hidden_controlled_mu) / VAR  # no abs

        if hidden_deviation <= SMD_THRESHOLD: # 0.1
            balanced_feature_tem.append(fea)
        else:
            if not math.isnan(float(hidden_deviation)):
                unbalanced_feature_tem.append(fea)
                unbalanced_feature_tem_SMD.append(hidden_deviation)
                if hidden_deviation>unbalanced_feature_tem_SMD_max:
                    unbalanced_feature_tem_SMD_max = hidden_deviation


    print('print--balanced_feature:', len(balanced_feature_tem))
    print('n_unbalanced_feature', len(unbalanced_feature_tem))
    print('unbalanced_feature', unbalanced_feature_tem)
    print('unbalanced_feature_SMD', unbalanced_feature_tem_SMD)
    print('unbalanced_feature_tem_SMD_max',unbalanced_feature_tem_SMD_max)

    # compute HR and save summary results
    df = df
    df.loc[df['outcome'] == -1, 'outcome'] = 0

    # for no adjusted HR
    cph_ori = CoxPHFitter()
    cox_data_ori = df[['treatment', 't2e', 'outcome']]
    cph_ori.fit(df=cox_data_ori, duration_col='t2e', event_col='outcome')
    HR_ori = cph_ori.hazard_ratios_['treatment']
    CI_ori = np.exp(cph_ori.confidence_intervals_.values.reshape(-1))
    P_value_ori = cph_ori.summary.loc['treatment', 'p']

    print('HR_ori:', HR_ori)
    print('CI_ori:', CI_ori)
    print('P_value_ori:', P_value_ori)

    # for adjusted HR
    # cph_adjusted = CoxPHFitter(penalizer=0.0001)
    # cox_data_adjusted = df_matched[list(feature_name) + ['treatment', 't2e', 'outcome']]
    # col_sums = cox_data_adjusted.sum(axis=0)
    # fixed_cols = cox_data_adjusted.columns[col_sums == 0]  # drop columns if the sum is 0
    # cox_data_adjusted = cox_data_adjusted.drop(columns=fixed_cols)
    # cph_adjusted.fit(df=cox_data_adjusted, duration_col='t2e', event_col='outcome')
    # HR_adjusted = cph_adjusted.hazard_ratios_['treatment']
    # CI_adjusted = np.exp(cph_adjusted.confidence_intervals_.values.reshape(-1))[-2:]
    # P_value_adjusted = cph_adjusted.summary.loc['treatment', 'p']
    #
    # print('HR_adjusted:', HR_adjusted)
    # print('CI_adjusted:', CI_adjusted)
    # print('P_value_adjusted:', P_value_adjusted)

    return len(unbalanced_feature_tem), unbalanced_feature_tem, unbalanced_feature_tem_SMD, unbalanced_feature_tem_SMD_max, HR_ori, CI_ori, P_value_ori


def compute_KMF(df_before_match, df_after_match):
    df_ori = df_before_match.copy()
    df_ori.loc[df_ori['outcome'] == -1, 'outcome'] = 0
    df_ori_treated = df_ori.loc[df_ori['treatment'] == 1]
    df_ori_control = df_ori.loc[df_ori['treatment'] == 0]
    kmf_ori_1 = KaplanMeierFitter(label='Treated_unmatched').fit(df_ori_treated['t2e'],
                                                                 event_observed=df_ori_treated['outcome'],
                                                                 label="Treated_unmatched")
    kmf_ori_0 = KaplanMeierFitter(label='Control_unmatched').fit(df_ori_control['t2e'],
                                                                 event_observed=df_ori_control['outcome'],
                                                                 label="Control_unmatched")

    df_m = df_after_match.copy()
    df_m.loc[df_m['outcome'] == -1, 'outcome'] = 0
    df_m_treated = df_m.loc[df_m['treatment'] == 1]
    df_m_control = df_m.loc[df_m['treatment'] == 0]
    kmf_m_1 = KaplanMeierFitter(label='Treated_matched').fit(df_m_treated['t2e'],
                                                             event_observed=df_m_treated['outcome'],
                                                             label="Treated_matched")
    kmf_m_0 = KaplanMeierFitter(label='Control_matched').fit(df_m_control['t2e'],
                                                             event_observed=df_m_control['outcome'],
                                                             label="Control_matched")

    ax = plt.subplot(111)
    figtitle = args.treated_drug + '_S' + str(args.random_seed) + 'C' + args.controlled_drug
    figsave = args.output_dir + args.treated_drug + '\\' + args.treated_drug + '_S' + str(
        args.random_seed) + 'C' + args.controlled_drug + '_KMF_result'
    ax.set_title(os.path.basename(figtitle))
    kmf_ori_1.plot_survival_function(ax=ax)
    kmf_ori_0.plot_survival_function(ax=ax)
    kmf_m_1.plot_survival_function(ax=ax)
    kmf_m_0.plot_survival_function(ax=ax)
    plt.savefig(figsave + '_km.png')
    plt.clf()

def compute_AF(df_before_match, df_after_match):
    df_ori = df_before_match.copy()
    df_ori.loc[df_ori['outcome'] == -1, 'outcome'] = 0
    df_ori_treated = df_ori.loc[df_ori['treatment'] == 1]
    df_ori_control = df_ori.loc[df_ori['treatment'] == 0]
    af_ori_1 = AalenJohansenFitter(calculate_variance=True).fit(df_ori_treated['t2e'],
                                                                df_ori_treated['outcome'],
                                                                label="Treated_unmatched", event_of_interest=1)
    af_ori_0 = AalenJohansenFitter(calculate_variance=True).fit(df_ori_control['t2e'],
                                                                df_ori_control['outcome'],
                                                                label="Control_unmatched", event_of_interest=1)
    df_m = df_after_match.copy()
    df_m_return = df_m[['treatment','outcome','t2e']]

    df_m.loc[df_m['outcome'] == -1, 'outcome'] = 0
    df_m_treated = df_m.loc[df_m['treatment'] == 1]
    df_m_control = df_m.loc[df_m['treatment'] == 0]
    af_m_1 = AalenJohansenFitter(calculate_variance=True).fit(df_m_treated['t2e'],
                                                              df_m_treated['outcome'],
                                                              label="Treated", event_of_interest=1)
    af_m_0 = AalenJohansenFitter(calculate_variance=True).fit(df_m_control['t2e'],
                                                              df_m_control['outcome'],
                                                              label="Control", event_of_interest=1)
    af_m_1_cumulative_density = af_m_1.cumulative_density_.copy()
    af_m_0_cumulative_density = af_m_0.cumulative_density_.copy()
    af_m_1_cumulative_density['CIF_1_risk'] = af_m_1_cumulative_density['CIF_1'].diff()
    af_m_0_cumulative_density['CIF_1_risk'] = af_m_0_cumulative_density['CIF_1'].diff()
    af_m_1_cumulative_density.loc[0.0, 'CIF_1_risk'] = 0
    af_m_0_cumulative_density.loc[0.0, 'CIF_1_risk'] = 0

    ax = plt.subplot(111)
    # figtitle = args.treated_drug + '_S' + str(args.random_seed) + 'C' + args.controlled_drug
    # figtitle = 'Drug: ketorolac'
    figsave = args.output_dir + args.treated_drug + '\\' + args.treated_drug + '_S' + str(
        args.random_seed) + 'C' + args.controlled_drug + '_AF_result'
    # ax.set_title(os.path.basename(figtitle))
    # af_ori_1.plot(ax=ax)
    # af_ori_0.plot(ax=ax)
    af_m_1.plot(ax=ax)
    af_m_0.plot(ax=ax)
    plt.xlabel('AD time (day)')
    plt.ylabel('Cumulative density')
    plt.savefig(figsave + '_af.png')
    plt.clf()

    return df_m_return, af_m_1_cumulative_density, af_m_0_cumulative_density

def compute_AF_integration(df_before_match, df_after_match):
    df_ori = df_before_match.copy()
    df_ori.loc[df_ori['outcome'] == -1, 'outcome'] = 0
    df_ori_treated = df_ori.loc[df_ori['treatment'] == 1]
    df_ori_control = df_ori.loc[df_ori['treatment'] == 0]
    af_ori_1 = AalenJohansenFitter(calculate_variance=True).fit(df_ori_treated['t2e'],
                                                                df_ori_treated['outcome'],
                                                                label="Treated_unmatched", event_of_interest=1)
    af_ori_0 = AalenJohansenFitter(calculate_variance=True).fit(df_ori_control['t2e'],
                                                                df_ori_control['outcome'],
                                                                label="Control_unmatched", event_of_interest=1)
    df_m = df_after_match.copy()
    df_m_return = df_m[['treatment','outcome','t2e']]

    df_m.loc[df_m['outcome'] == -1, 'outcome'] = 0
    df_m_treated = df_m.loc[df_m['treatment'] == 1]
    df_m_control = df_m.loc[df_m['treatment'] == 0]

    # read integrated control
    # data_path = r'K:\WorkArea-zhx2005\PycharmProjects\first_AD\output\35827_result_all'
    # all_case_data = pd.read_csv(data_path+'\\all_case_data.csv',index_col=False)
    # df_m_treated = all_case_data
    # df_m_treated.loc[df_m_treated['outcome'] == -1, 'outcome'] = 0
    #
    # all_control_data = pd.read_csv(data_path+'\\all_control_data.csv',index_col=False)
    # df_m_control = all_control_data
    # df_m_control.loc[df_m_control['outcome'] == -1, 'outcome'] = 0

    af_m_1 = AalenJohansenFitter(calculate_variance=True).fit(df_m_treated['t2e'],
                                                              df_m_treated['outcome'],
                                                              label="Treated", event_of_interest=1)
    af_m_0 = AalenJohansenFitter(calculate_variance=True).fit(df_m_control['t2e'],
                                                              df_m_control['outcome'],
                                                              label="Control (Random and ATC-L2)", event_of_interest=1)


    # df_m_case_control = pd.concat([df_m_treated, df_m_control],axis=0,ignore_index=True)
    # af_m_01 = AalenJohansenFitter(calculate_variance=True).fit(df_m_case_control['t2e'],
    #                                                           df_m_case_control['outcome'],
    #                                                           event_of_interest=1)

    # af_m_1_cumulative_density = af_m_1.cumulative_density_.copy()
    #
    # af_m_0_cumulative_density = af_m_0.cumulative_density_.copy()
    # af_m_1_cumulative_density['CIF_1_risk'] = af_m_1_cumulative_density['CIF_1'].diff()
    # af_m_0_cumulative_density['CIF_1_risk'] = af_m_0_cumulative_density['CIF_1'].diff()
    # af_m_1_cumulative_density.loc[0.0, 'CIF_1_risk'] = 0
    # af_m_0_cumulative_density.loc[0.0, 'CIF_1_risk'] = 0

    # read control cumulative_density
    data_path = r'K:\WorkArea-zhx2005\PycharmProjects\first_AD\output\35827_result_all'

    case_cum_density_CI = pd.read_csv(data_path+'\\case_cum_density_CI.csv',index_col=0)
    case_cum_density_average = pd.read_csv(data_path+'\\case_cum_density_average.csv',index_col=0)
    af_m_1.cumulative_density_ = case_cum_density_average
    af_m_1.confidence_interval_ = case_cum_density_CI
    af_m_1.confidence_interval_cumulative_density_ = case_cum_density_CI
    af_m_1.timeline = np.array(case_cum_density_average.index)

    control_cum_density_CI = pd.read_csv(data_path+'\\control_cum_density_CI.csv',index_col=0)
    control_cum_density_average = pd.read_csv(data_path+'\\control_cum_density_average.csv',index_col=0)
    af_m_0.cumulative_density_ = control_cum_density_average
    af_m_0.confidence_interval_ = control_cum_density_CI
    af_m_0.confidence_interval_cumulative_density_ = control_cum_density_CI
    af_m_0.timeline = np.array(control_cum_density_average.index)

    fig = plt.figure(figsize=(10,8))#
    ax = fig.add_subplot(111)
    # ax.patch.set_facecolor('white')
    # fig.patch.set_facecolor('white')

    # ax.grid(True)
    # ax = plt.subplot(111)
    # figtitle = args.treated_drug + '_S' + str(args.random_seed) + 'C' + args.controlled_drug
    figtitle = 'Drug: ketorolac'
    figsave = args.output_dir + args.treated_drug + '\\' + args.treated_drug
    ax.set_title(os.path.basename(figtitle))
    # af_ori_1.plot(ax=ax)
    # af_ori_0.plot(ax=ax)
    from lifelines.plotting import add_at_risk_counts

    af_m_1.plot(ax=ax)
    af_m_0.plot(ax=ax)
    # add_at_risk_counts(af_m_1, af_m_0, ax=ax) # ,col_name=
    # plt.tight_layout()
    # af_m_01.plot(ax=ax, at_risk_counts=True)
    ax.set(ylabel ='Cumulative density', xlabel = 'AD time (day)')


    # plt.xlabel('AD time (day)')
    # plt.ylabel('Cumulative density')
    # plt.show()
    # plt.savefig(figsave + '_af_integration.png')
    fig.savefig(figsave+'_af_integration_all.png')
    plt.clf()



# def main(args):
if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('SMD_THRESHOLD: ', SMD_THRESHOLD)
    print('random_seed: ', args.random_seed)
    print('device: ', args.device)
    print('Drug {} cohort: '.format(args.treated_drug))
    print('save_model_filename', args.save_model_filename)

    # %% 1. Load Data
    ## load drug code name mapping
    if args.drug_coding.lower() == 'gpi':
        _fname = os.path.join(args.data_dir, '../_gpi_ingredients_nameset_cnt.pkl')
        print('drug_name file: ', _fname)
        with open(_fname, 'rb') as f:
            # change later, move this file to pickles also
            gpiing_names_cnt = pickle.load(f)
            drug_name = {}
            for key, val in gpiing_names_cnt.items():
                drug_name[key] = '/'.join(val[0])
        print('Using GPI vocabulary, len(drug_name) :', len(drug_name))
    else:
        with open(r'pickles/rxnorm_label_mapping.pkl', 'rb') as f:
            drug_name = pickle.load(f)
            print('Using rxnorm_cui vocabulary, len(drug_name) :', len(drug_name))
    # Load diagnosis code mapping
    icd_to_ccw, icd_to_ccwname, ccw_info = load_icd_to_ccw('mapping/CCW_to_use_enriched.json')
    dx_name = ccw_info[1]

    # Load treated triple list and build control triple list
    treated = pickle.load(open(args.data_dir + args.treated_drug + '.pkl', 'rb'))

    controlled = []
    drugfile_in_dir = sorted([x for x in os.listdir(args.data_dir) if
                              (x.split('.')[0].isdigit() and x.split('.')[1] == 'pkl')])
    drug_in_dir = [x.split('.')[0] for x in drugfile_in_dir]
    cohort_size = pickle.load(open(os.path.join(args.data_dir, 'cohorts_size.pkl'), 'rb'))

    # 1-A: build control groups
    n_control_patient = 0
    controlled_drugs_range = []
    n_treat_patient = cohort_size.get(args.treated_drug + '.pkl')
    if args.controlled_drug == 'random':
        print('Control groups: random')
        # sorted for deterministic, listdir seems return randomly
        controlled_drugs = sorted(list(
            set(drugfile_in_dir) -
            set(args.treated_drug + '.pkl')
        ))
        np.random.shuffle(controlled_drugs)

        for c_id in controlled_drugs:
            # n_control_patient += cohort_size.get(c_id)
            # controlled_drugs_range.append(c_id)
            # if n_control_patient >= (args.controlled_drug_ratio + 1) * n_treat_patient:
            #     break

            controlled_drugs_range.append(c_id)
            c = pickle.load(open(args.data_dir + c_id, 'rb'))
            n_c = 0
            n_c_exclude = 0
            for p in c:
                p_drug = sum(p[1][0], [])
                if args.treated_drug not in p_drug:
                    controlled.append(p)
                    n_c += 1
                else:
                    n_c_exclude += 1
            n_control_patient += n_c
            if n_control_patient >= (args.controlled_drug_ratio + 1) * n_treat_patient:
                break

    else:
        print('Control groups: atc level3')
        ATC2DRUG = pickle.load(open(os.path.join('pickles/', 'atcL3_rx.pkl'), 'rb'))  # ATC2DRUG.pkl
        DRUG2ATC = pickle.load(open(os.path.join('pickles/', 'rx_atcL3.pkl'), 'rb'))  # DRUG2ATC.pkl

        if args.drug_coding.lower() == 'rxnorm':
            # if args.stats:
            ## atc drug statistics:
            in_atc = np.array([x in DRUG2ATC for x in drug_in_dir])
            print('Total drugs in dir: {}, {} ({:.2f}%) have atc mapping, {} ({:.2f}%) have not'.format(
                in_atc.shape[0],
                in_atc.sum(),
                in_atc.mean() * 100,
                in_atc.shape[0] - in_atc.sum(),
                (1 - in_atc.mean()) * 100,
            ))
            print('{} rxnorm codes without atc in DRUG2ATC are:\n'.format(len(set(drug_in_dir) - set(DRUG2ATC.keys()))),
                  set(drug_in_dir) - set(DRUG2ATC.keys()))
            ###

        atc_group = set()
        if args.drug_coding.lower() == 'gpi':
            drug_atc = args.treated_drug[:2]
            for d in drug_in_dir:
                if d[:2] == drug_atc:
                    atc_group.add(d)
        else:
            drug_atc = DRUG2ATC.get(args.treated_drug, [])
            for atc in drug_atc:
                if atc in ATC2DRUG:
                    atc_group.update(ATC2DRUG.get(atc))

        if len(atc_group) > 1:
            # atc control may not have n_treat * ratio number of patients
            controlled_drugs = [drug + '.pkl' for drug in atc_group if drug != args.treated_drug]
            controlled_drugs = sorted(list(
                set(drugfile_in_dir) -
                set(args.treated_drug + '.pkl') &
                set(controlled_drugs)
            ))
            np.random.shuffle(controlled_drugs)

        else:
            print("No atcl3 drugs for treated_drug {}, choose random".format(args.treated_drug))
            # all_atc = set(ATC2DRUG.keys()) - set(drug_atc)
            # sample_atc = [atc for atc in list(all_atc) if len(ATC2DRUG.get(atc)) == 1]
            # sample_drug = set()
            # for atc in sample_atc:
            #     for drug in ATC2DRUG.get(atc):
            #         sample_drug.add(drug)
            # controlled_drugs_range = [drug + '.pkl' for drug in sample_drug if drug != args.treated_drug]
            controlled_drugs = sorted(list(
                set(drugfile_in_dir) -
                set(args.treated_drug + '.pkl')
            ))
            np.random.shuffle(controlled_drugs)
            # n_control_patient = 0
            # controlled_drugs_range = []
            # n_treat_patient = cohort_size.get(args.treated_drug + '.pkl')
            # for c_id in controlled_drugs:
            #     n_control_patient += cohort_size.get(c_id)
            #     controlled_drugs_range.append(c_id)
            #     if n_control_patient >= (args.controlled_drug_ratio + 1) * n_treat_patient:
            #         break
        for c_id in controlled_drugs:
            controlled_drugs_range.append(c_id)
            c = pickle.load(open(args.data_dir + c_id, 'rb'))
            n_c = 0
            for p in c:
                p_drug = sum(p[1][0], [])
                if args.treated_drug not in p_drug:
                    controlled.append(p)
                    n_c += 1
            n_control_patient += n_c
            if n_control_patient >= (args.controlled_drug_ratio + 1) * n_treat_patient:
                break

    # for c_drug_id in controlled_drugs_range:
    #     c = pickle.load(open(args.data_dir + c_drug_id, 'rb'))
    #     controlled.extend(c)

    intersect = set(np.asarray(treated)[:, 0]).intersection(set(np.asarray(controlled)[:, 0]))

    controlled = np.asarray([controlled[i] for i in range(len(controlled)) if controlled[i][0] not in intersect])
    controlled_indices = list(range(len(controlled)))
    controlled_sample_index = int(args.controlled_drug_ratio * len(treated))

    np.random.shuffle(controlled_indices)

    controlled_sample_indices = controlled_indices[:controlled_sample_index]

    controlled_sample = controlled[controlled_sample_indices]

    n_user, n_nonuser = len(treated), len(controlled_sample)
    print('#treated: {}, #controls: {}'.format(n_user, n_nonuser),
          '(Warning: the args.controlled_drug_ratio is {},'
          ' and the atc control cohort may have less patients than expected)'.format(args.controlled_drug_ratio))

    # 1-B: calculate the statistics of treated v.s. control
    if args.stats or args.stats_exit:
        # demo_feature_vector: [age, sex, race, days_since_mci]
        # triple = (patient,
        #           [rx_codes, dx_codes, demo_feature_vector[0], demo_feature_vector[1], demo_feature_vector[3]],
        #           (outcome, outcome_t2e))
        print('Summarize statistics between treat v.s. control ...')
        from univariate_statistics import build_patient_characteristics_from_triples, \
            statistics_for_treated_control, build_patient_characteristics_from_triples_v2

        with open('mapping/CCW_AD_comorbidity.json') as f:
            ccw_ad_comorbidity = json.load(f)
        ccwcomorid_name = {}
        for name, v in ccw_ad_comorbidity.items():
            ccwcomorid_name[ccw_info[0][name]] = name

        atc2rxnorm = pickle.load(open(os.path.join('pickles', 'atcL3_rx.pkl'), 'rb'))  # ATC2DRUG.pkl
        if args.drug_coding.lower() == 'gpi':
            is_antidiabetic = lambda x: (x[:2] == '27')
            is_antihypertensives = lambda x: (x[:2] == '36')
        else:
            is_antidiabetic = lambda x: x in atc2rxnorm['A10']
            is_antihypertensives = lambda x: x in atc2rxnorm['C02']
        drug_criterion = {'antidiabetic': is_antidiabetic, 'antihypertensives': is_antihypertensives}

        # df_treated = build_patient_characteristics_from_triples(treated, ccwcomorid_name, drug_criterion)
        df_treated = build_patient_characteristics_from_triples_v2(treated, ccwcomorid_name, drug_criterion)
        # df_control = build_patient_characteristics_from_triples(controlled_sample, ccwcomorid_name, drug_criterion)
        df_control = build_patient_characteristics_from_triples_v2(controlled_sample, ccwcomorid_name, drug_criterion)
        # df_control.to_csv(r'K:\WorkArea-zhx2005\PycharmProjects\first_AD\output\5470_dx_exclude\5470_dx_characteristics.csv',index=True)
        add_row = pd.Series({'treat': args.treated_drug,
                             'control': ';'.join([x.split('.')[0] for x in controlled_drugs_range]),
                             'p-value': np.nan},
                            name='file')
        df_stats = statistics_for_treated_control(
            df_treated,
            df_control,
            args.save_model_filename + '_stats.csv',
            add_row)
        print('Characteristics statistic of treated v.s. control, done!')
        if args.stats_exit:
            print('Only run stats! stats_exit! Total Time used:',
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            sys.exit(5)  # stats_exit only run statas

    # 1-C: build pytorch dataset
    print("Constructed Dataset, choose med_code_topk:", args.med_code_topk)
    my_dataset = Dataset(treated, controlled_sample,
                         med_code_topk=args.med_code_topk,
                         diag_name=dx_name,
                         med_name=drug_name)  # int(len(treated)/5)) #150)

    n_feature = my_dataset.DIM_OF_CONFOUNDERS  # my_dataset.med_vocab_length + my_dataset.diag_vocab_length + 3
    feature_name = my_dataset.FEATURE_NAME
    print('n_feature: ', n_feature, ':')
    # print(feature_name)

    train_ratio = 0.8  # 0.8 0.5
    val_ratio = 0.01 #  0.1
    print('train_ratio: ', train_ratio,
          'val_ratio: ', val_ratio,
          'test_ratio: ', 1 - (train_ratio + val_ratio))

    dataset_size = len(my_dataset)
    indices = list(range(dataset_size))
    train_index = int(np.floor(train_ratio * dataset_size))
    val_index = int(np.floor(val_ratio * dataset_size))

    np.random.shuffle(indices)

    train_indices, val_indices, test_indices = indices[:train_index], \
                                               indices[train_index:train_index + val_index], \
                                               indices[train_index + val_index:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                             sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                              sampler=test_sampler)
    data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                              sampler=SubsetRandomSampler(indices))

    # %% Logistic regression PS PSModels
    if args.run_model in ['LR']:
        print("**************************************************")
        print("**************************************************")
        print(args.run_model, ' PS model learning:')

        print('Train data:')
        train_x, train_t, train_y = flatten_data(my_dataset, train_indices)
        print('Validation data:')
        val_x, val_t, val_y = flatten_data(my_dataset, val_indices)
        print('Test data:')
        test_x, test_t, test_y = flatten_data(my_dataset, test_indices)
        print('All data:')
        x, t, y = flatten_data(my_dataset, indices)  # all the data

        # zxx add ----Propensity score matching---start
        from psmpy import PsmPy
        from psmpy.plotting import *
        t = t.reshape((t.shape[0],1))
        x_t_y = np.concatenate((x, t, y), axis=1)
        feature_name_psm = list(feature_name)#
        col_treatment = ['treatment']
        col_outcome = ['outcome','t2e']
        df_psm_col_name = feature_name_psm + col_treatment + col_outcome
        df_psm = pd.DataFrame(x_t_y, index=range(len(indices)), columns= df_psm_col_name)
        df_psm = df_psm.assign(deal_id=indices)

        # df_psm.to_csv(r'K:\WorkArea-zhx2005\PycharmProjects\first_AD\output\\' + str(args.treated_drug)+'_'+'df_psm.csv',index=True)
        # feature_name = ['sex','age']
        # feature_name_psm = list(feature_name)
        # df_unmatched = df_psm[['sex','age','treatment','outcome','t2e','deal_id']] # only consider demographic
        df_unmatched = df_psm # consider 265 ...
        # exclude variables that not for matching
        exclude_var = ['outcome','t2e']
        exclude_var_df = df_psm[['deal_id']+exclude_var]
        # exclude_var_df = exclude_var_df.set_index('deal_id')

        # perform psm
        psm = PsmPy(df_psm, indx='deal_id', treatment='treatment', exclude=exclude_var)
        psm.logistic_ps(balance=False)
        # print(psm.predicted_data)
        # psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None) # caliper = None
        psm.knn_matched_12n(matcher='propensity_logit',how_many=1) # how_many = 1
        df_matched = psm.df_matched

        # contact a completed df
        df_matched = pd.merge(df_matched,exclude_var_df,on='deal_id')

        # compute SMD
        num_treated_unmatched = len(df_unmatched[df_unmatched['treatment']==1])
        num_controlled_unmatched = len(df_unmatched[df_unmatched['treatment']==0])
        num_treated_matched = len(df_matched[df_matched['treatment']==1])
        num_controlled_matched = len(df_matched[df_matched['treatment']==0])

        num_unbalanced_feature_unmatched, unbalanced_feature_unmatched, unbalanced_feature_unmatched_SMD, unbalanced_feature_unmatched_SMD_max, HR_unmatched,CI_unmatched, P_value_unmatched = compute_SMD_HR(df_unmatched,feature_name)
        num_unbalanced_feature_matched, unbalanced_feature_matched, unbalanced_feature_matched_SMD, unbalanced_feature_matched_SMD_max, HR_matched, CI_matched, P_value_matched = compute_SMD_HR(df_matched, feature_name)

        # # save HR result
        res_col = ['drug','seed', 'num_treated_unmatched','num_controlled_unmatched',
                   'num_unbalanced_feature_unmatched','unbalanced_feature_unmatched','unbalanced_feature_unmatched_SMD', 'unbalanced_feature_unmatched_SMD_max', 'HR_unmatched', 'CI_unmatched','P_value_unmatched',
                   'num_treated_matched', 'num_controlled_matched',
                   'num_unbalanced_feature_matched', 'unbalanced_feature_matched', 'unbalanced_feature_matched_SMD', 'unbalanced_feature_matched_SMD_max', 'HR_matched', 'CI_matched', 'P_value_matched']

        res_df = pd.DataFrame(index=[0],columns=res_col)
        res_df.loc[0, 'drug'] = args.treated_drug
        res_df.loc[0, 'seed'] = args.random_seed

        res_df.loc[0, 'num_treated_unmatched'] = num_treated_unmatched
        res_df.loc[0, 'num_controlled_unmatched'] = num_controlled_unmatched

        res_df.loc[0, 'num_unbalanced_feature_unmatched'] = num_unbalanced_feature_unmatched
        res_df.at[0, 'unbalanced_feature_unmatched'] = unbalanced_feature_unmatched
        res_df.loc[0, 'unbalanced_feature_unmatched_SMD'] = unbalanced_feature_unmatched_SMD
        res_df.loc[0, 'unbalanced_feature_unmatched_SMD_max'] = unbalanced_feature_unmatched_SMD_max

        res_df.loc[0, 'HR_unmatched'] = HR_unmatched
        res_df.at[0, 'CI_unmatched'] = CI_unmatched
        res_df.loc[0, 'P_value_unmatched'] = P_value_unmatched

        res_df.loc[0, 'num_treated_matched'] = num_treated_matched
        res_df.loc[0, 'num_controlled_matched'] = num_controlled_matched

        res_df.loc[0, 'num_unbalanced_feature_matched'] = num_unbalanced_feature_matched
        res_df.at[0, 'unbalanced_feature_matched'] = unbalanced_feature_matched
        res_df.loc[0, 'unbalanced_feature_matched_SMD'] = unbalanced_feature_matched_SMD
        res_df.loc[0, 'unbalanced_feature_matched_SMD_max'] = unbalanced_feature_matched_SMD_max

        res_df.loc[0, 'HR_matched'] = HR_matched
        res_df.at[0, 'CI_matched'] = CI_matched
        res_df.loc[0, 'P_value_matched'] = P_value_matched

        # res_df.to_csv(args.output_dir+args.treated_drug+'\\'+ args.treated_drug +'_S'+str(args.random_seed)+'C'+args.controlled_drug+'_HR_result.csv',index=True)
        res_df.to_csv(args.output_dir + args.treated_drug + '\\' + args.treated_drug + '_S' + str(args.random_seed) + 'C' + args.controlled_drug + '_ratio_'+ str(args.controlled_drug_ratio) + '_HR_result.csv', index=True)

        # using KMF,  KaplanMeierFitter()
        # compute_KMF(df_unmatched,df_matched)
        # using AalenJohansenFitter
        num_unbalanced_feature_matched_threshold = 6 # <2% total number of covariates
        if num_unbalanced_feature_matched<=num_unbalanced_feature_matched_threshold:
            df_return, af_m_1_cumulative_density, af_m_0_cumulative_density = compute_AF(df_unmatched,df_matched)
            # compute_AF_integration(df_unmatched, df_matched)

            # df_return.to_csv(args.output_dir+args.treated_drug+'\\'+ args.treated_drug +'_S'+str(args.random_seed)+'C'+args.controlled_drug+'_AF_data.csv',index=True)
            df_return.to_csv(args.output_dir + args.treated_drug + '\\' + args.treated_drug + '_S' + str(args.random_seed) + 'C' + args.controlled_drug +'_ratio_' + str(args.controlled_drug_ratio) + '_AF_data.csv', index=True)

            # af_m_1_cumulative_density.to_csv(args.output_dir+args.treated_drug+'\\'+ args.treated_drug +'_S'+str(args.random_seed)+'C'+args.controlled_drug+'_AF_m_1.csv',index=True)
            # af_m_0_cumulative_density.to_csv(args.output_dir+args.treated_drug+'\\'+ args.treated_drug +'_S'+str(args.random_seed)+'C'+args.controlled_drug+'_AF_m_0.csv',index=True)
