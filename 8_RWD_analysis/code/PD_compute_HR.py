# compute HR 
import os
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
import numpy as np
from lifelines import AalenJohansenFitter
import matplotlib.pyplot as plt
from lifelines.plotting import add_at_risk_counts

# #################################################   for all drugs  # compute HR,--CI, --p-value, plot AF integration figure   ################################################# ################################################# #################################################
def boot_matrix(z, B):
    """Bootstrap sample

    Returns all bootstrap samples in a matrix"""
    z = np.array(z).flatten()
    n = len(z)  # sample size
    idz = np.random.randint(0, n, size=(B, n))  # indices to pick for all boostrap samples
    return z[idz]

def bootstrap_mean_pvalue(x, expected_mean=0., B=1000):
    """
    Ref:
    1. https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#cite_note-:0-1
    2. https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Hypothesis.pdf
    3. https://github.com/mayer79/Bootstrap-p-values/blob/master/Bootstrap%20p%20values.ipynb
    4. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html?highlight=one%20sample%20ttest
    Bootstrap p values for one-sample t test
    Returns boostrap p value, test statistics and parametric p value"""
    n = len(x)
    orig = stats.ttest_1samp(x, expected_mean)
    # Generate boostrap distribution of sample mean
    x_boots = boot_matrix(x - x.mean() + expected_mean, B=B)
    x_boots_mean = x_boots.mean(axis=1)
    t_boots = (x_boots_mean - expected_mean) / (x_boots.std(axis=1, ddof=1) / np.sqrt(n))
    p = np.mean(t_boots >= orig[0])
    p_final = 2 * min(p, 1 - p)
    # Plot bootstrap distribution
    # if plot:
    #     plt.figure()
    #     plt.hist(x_boots_mean, bins="fd")
    return p_final, orig


file_path = 'K:\WorkArea-zhx2005\PycharmProjects\PD\output_fall_all_cohort\\'

drug_list = ['6387', '6809', '1151', '620', '6135', '6375','33738', '704', '42316', '2626']
num_unbalanced_feature_matched_threshold = 6 # <2%, total features
num_trials = 100
drug_name = {
    "6387": "lidocaine", "6809": "metformin", "1151": "ascorbic acid",
    "620": "amantadine", "6135": "ketoconazole", "6375": "levodopa",
    "33738": "pioglitazone", "704": "amitriptyline", "42316": "tacrolimus",
    "2626": "clozapine"
}

for drug in drug_list:
    final_result_df = pd.DataFrame(
        index=['Random', 'ATC-L3', 'All'],
        columns=['n_unbalanced_feature','HR','95% CI', 'p-value']
    )
    print('drug:', drug)
    HR_random = []
    HR_atc = []
    HR_all = []
    num_unbalanced_feature_random = []
    num_unbalanced_feature_atc = []
    num_unbalanced_feature_all = []
    AF_random_df = pd.DataFrame()
    AF_atc_df = pd.DataFrame()
    AF_all_df = pd.DataFrame()

    # generated HR and AF data
    for i in range(num_trials):
        if i < 50: # random
            HR_data = pd.read_csv(file_path+drug+'\\'+drug+'_S'+str(i)+'Crandom_HR_result.csv',index_col=0)
            num_unbalanced_feature_matched = HR_data['num_unbalanced_feature_matched'].values[0]
            if num_unbalanced_feature_matched<=num_unbalanced_feature_matched_threshold: # balanced emulated trials
                # print('seed:',i)
                num_unbalanced_feature_random.append(num_unbalanced_feature_matched)
                num_unbalanced_feature_all.append(num_unbalanced_feature_matched)
                # collect HR data, random seed
                HR_matched = HR_data['HR_matched'].values[0]
                HR_random.append(HR_matched)
                HR_all.append(HR_matched)

                # collect AF data,  atc seed
                AF_data = pd.read_csv(file_path + drug + '\\' + drug + '_S' + str(i) + 'Crandom_AF_data.csv', index_col=0)
                AF_random_df = pd.concat([AF_random_df,AF_data])
                AF_random_df = AF_random_df.reset_index(drop=True)

                AF_all_df = pd.concat([AF_all_df,AF_data])
                AF_all_df = AF_all_df.reset_index(drop=True)

                # print('len(AF_random_df)',len(AF_random_df))

        else: # ATC-L3
            HR_data = pd.read_csv(file_path + drug + '\\' + drug + '_S' + str(i) + 'Catc_HR_result.csv', index_col=0)
            num_unbalanced_feature_matched = HR_data['num_unbalanced_feature_matched'].values[0]
            if num_unbalanced_feature_matched <= num_unbalanced_feature_matched_threshold:
                num_unbalanced_feature_atc.append(num_unbalanced_feature_matched)
                num_unbalanced_feature_all.append(num_unbalanced_feature_matched)
                # print('seed:', i)
                # collect HR data, atc seed
                HR_matched = HR_data['HR_matched'].values[0]
                HR_atc.append(HR_matched)
                HR_all.append(HR_matched)

                # collect AF data,  atc seed
                AF_data = pd.read_csv(file_path + drug + '\\' + drug + '_S' + str(i) + 'Catc_AF_data.csv', index_col=0)
                AF_atc_df = pd.concat([AF_atc_df, AF_data])
                AF_atc_df = AF_atc_df.reset_index(drop=True)

                AF_all_df = pd.concat([AF_all_df, AF_data])
                AF_all_df = AF_all_df.reset_index(drop=True)

    # compute HR, CI, and p-value, plot AF figures
    print('balanced emulated trials----len(HR_random)', len(HR_random))
    print('balanced emulated trials----len(HR_atc)', len(HR_atc))
    print('balanced emulated trials----len(HR_all)', len(HR_all))
    if len(HR_random)<1 or len(HR_atc)<1 or len(HR_all)<1:
        continue

    for HR_type in ['Random','ATC-L3','All']:
        if HR_type =='Random':
            HR_value = HR_random
            AF_value = AF_random_df
            num_unbalanced_feature = num_unbalanced_feature_random
        elif HR_type == 'ATC-L3':
            HR_value = HR_atc
            AF_value = AF_atc_df
            num_unbalanced_feature = num_unbalanced_feature_atc
        else:
            HR_value = HR_all
            AF_value = AF_all_df
            num_unbalanced_feature = num_unbalanced_feature_all

        num_unbalanced_feature_value_median = float('%.2f' % np.median(np.array(num_unbalanced_feature)))
        num_unbalanced_feature_value_mean = float('%.2f' % np.mean(np.array(num_unbalanced_feature)))

        HR_value_median = float('%.2f' % np.median(np.array(HR_value)))
        HR_value_mean = float('%.2f' % np.mean(np.array(HR_value)))
        # HR_value_bootstrap_ci = bootstrap((HR_value,),np.median,confidence_level=0.95,random_state=42,method='percentile')
        HR_value_bootstrap_ci = bootstrap((HR_value,), np.mean, confidence_level=0.95, random_state=42, method='percentile')

        HR_value_CI_low = float('%.2f' % HR_value_bootstrap_ci.confidence_interval[0])
        HR_value_CI_high = float('%.2f' % HR_value_bootstrap_ci.confidence_interval[1])
        HR_value_p_value, _ = bootstrap_mean_pvalue(np.array(HR_value), expected_mean=1)
        # HR_value_p_value = float('%.2f' % HR_value_p_value)

        # save final results
        final_result_df.loc[HR_type, 'n_unbalanced_feature'] = num_unbalanced_feature_value_mean # num_unbalanced_feature_value_median, num_unbalanced_feature_value_mean
        final_result_df.loc[HR_type, 'HR'] = HR_value_mean # HR_value_median
        final_result_df.at[HR_type, '95% CI'] = "["+ str(HR_value_CI_low) + ", "+ str(HR_value_CI_high) +"]"
        final_result_df.loc[HR_type, 'p-value'] = HR_value_p_value

        # plot AF figures
        df_m = AF_value.copy()
        df_m = df_m[['treatment', 'outcome', 't2e']]

        df_m.loc[df_m['outcome'] == -1, 'outcome'] = 0
        df_m_treated = df_m.loc[df_m['treatment'] == 1]
        df_m_control = df_m.loc[df_m['treatment'] == 0]

        af_m_1 = AalenJohansenFitter(calculate_variance=True).fit(df_m_treated['t2e'],
                                                                  df_m_treated['outcome'],
                                                                  label="Treated", event_of_interest=1)
        af_m_0 = AalenJohansenFitter(calculate_variance=True).fit(df_m_control['t2e'],
                                                                  df_m_control['outcome'],
                                                                  label="Control "+"("+HR_type+")",
                                                                  event_of_interest=1)  

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        figtitle = 'Drug: '+drug_name[drug]
        figsave = file_path + drug + '\\' + drug
        ax.set_title(os.path.basename(figtitle))

        af_m_1.plot(ax=ax)
        af_m_0.plot(ax=ax)
        add_at_risk_counts(af_m_1, af_m_0, ax=ax)
        plt.tight_layout()
        ax.legend(loc='upper left', frameon=False)
        ax.set(ylabel='Cumulative density', xlabel='ND time (day)')
        # plt.show()
        fig.savefig(figsave + "_AF_"+HR_type+".pdf")
        plt.clf()
    # save final result to csv on disk
    final_result_df.to_csv(file_path + drug + '\\' + drug + '_final_statistics_result.csv',index=True)
    print(final_result_df)
print('Done!')
