"""
_*_ coding: utf-8 _*_
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pkl
import copy
import scipy.stats

fea_dict_df = pd.read_csv('[your directory]/validation/processed_data/features_dict_PPMI.csv')
fea_dict = fea_dict_df.set_index('PPMI Codes')['Data Element'].to_dict()


class EvaluationTables:

    def __init__(self, feature_list, static_data, seq_data, save_path=None):
        self.save_path = save_path
        tables_path = None
        if save_path is not None:
            tables_path = save_path  # + "/visit_data/"
        self.tables = self.get_var_table(feature_list, static_data, seq_data, tables_path)

    def evaluation_tables(self):
        return self.tables

    def UPDRS1(self, dataframe, seq_data, feature_map, visit):
        target_vars = ['updrs1', 'hallucination', 'Apathy', 'Pain', 'Fatigue']
        code_list = ["code_upd2101_cognitive_impairment", "code_upd2102_hallucinations_and_psychosis",
                     "code_upd2103_depressed_mood", "code_upd2104_anxious_mood", "code_upd2105_apathy",
                     "code_upd2106_dopamine_dysregulation_syndrome_features",
                     "code_upd2107_pat_quest_sleep_problems", "code_upd2108_pat_quest_daytime_sleepiness",
                     "code_upd2109_pat_quest_pain_and_other_sensations", "code_upd2110_pat_quest_urinary_problems",
                     "code_upd2111_pat_quest_constipation_problems",
                     "code_upd2112_pat_quest_lightheadedness_on_standing", "code_upd2113_pat_quest_fatigue"]
        updrs1_idxs = [feature_map[ft] for ft in code_list if ft in code_list]
        hal_idx, apa_idx, pain_idx, fat_idx = feature_map["code_upd2102_hallucinations_and_psychosis"], feature_map[
            "code_upd2105_apathy"], feature_map["code_upd2109_pat_quest_pain_and_other_sensations"], feature_map["code_upd2113_pat_quest_fatigue"]

        dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
        for idx, row in dataframe.iterrows():
            PATNO = row['participant_id']
            if PATNO not in seq_data:
                print("Error: participant_id %s not in sequence data!" % PATNO)
                return
            data_mtx = seq_data[PATNO]
            if visit >= len(data_mtx):
                continue

            if len(updrs1_idxs) == len(code_list):
                updrs1_score = data_mtx[visit, updrs1_idxs].sum()
            else:
                updrs1_score = np.nan
            hal, apa, pain, fat = data_mtx[visit, hal_idx], data_mtx[visit, apa_idx], data_mtx[visit, pain_idx], \
                                  data_mtx[visit, fat_idx]
            dataframe.loc[idx, target_vars] = [updrs1_score, hal, apa, pain, fat]

        print("UPDRS1 domain finished!")
        return dataframe

    def Epworth(self, dataframe, seq_data, feature_map, visit):
        target_vars = ['epworth']
        code_list = ["code_ess0101_sitting_and_reading", "code_ess0102_watching_tv",
                     "code_ess0103_sitting_inactive_in_public_place", "code_ess0104_passenger_in_car_for_hour",
                     "code_ess0105_lying_down_to_rest_in_afternoon", "code_ess0106_sitting_and_talking_to_someone",
                     "code_ess0107_sitting_after_lunch", "code_ess0108_car_stopped_in_traffic"]
        epworth_idxs = [feature_map[ft] for ft in code_list if ft in code_list]

        dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
        for idx, row in dataframe.iterrows():
            PATNO = row['participant_id']
            if PATNO not in seq_data:
                print("Error: participant_id %s not in sequence data!" % PATNO)
                return
            data_mtx = seq_data[PATNO]
            if visit >= len(data_mtx):
                continue
            if len(code_list) == len(epworth_idxs):
                epworth_score = data_mtx[visit, epworth_idxs].sum()
            else:
                epworth_score = np.nan
            dataframe.loc[idx, target_vars] = [epworth_score]

        print("Epworth domain finished!")
        return dataframe

    def MOCA(self, dataframe, seq_data, static_df, feature_map, visit):
        target_vars = ['moca', 'moca_visuospatial', 'moca_naming', 'moca_attention',
                       'moca_language', 'moca_delayed_recall']
        moca_codes = ['moca_total_score']
        moca_idxs = [feature_map[ft] for ft in moca_codes if ft in moca_codes]

        visu_codes = ["moca01_alternating_trail_making", "moca02_visuoconstr_skills_cube",
                      "moca03_visuoconstr_skills_clock_cont", "moca04_visuoconstr_skills_clock_num",
                      "moca05_visuoconstr_skills_clock_hands"]
        visu_idxs = [feature_map[ft] for ft in visu_codes if ft in feature_map]

        nam_codes = ["moca06_naming_lion", "moca07_naming_rhino", "moca08_naming_camel"]
        nam_idxs = [feature_map[ft] for ft in nam_codes if ft in feature_map]

        att_codes = ["moca09_attention_forward_digit_span", "moca10_attention_backward_digit_span",
                     "moca11_attention_vigilance", "moca12_attention_serial_7s"]
        att_idxs = [feature_map[ft] for ft in att_codes if ft in feature_map]

        lang_codes = ["moca13_sentence_repetition", "moca15_verbal_fluency"]
        lang_idxs = [feature_map[ft] for ft in lang_codes if ft in feature_map]

        dere_codes = ["moca17_delayed_recall_face", "moca18_delayed_recall_velvet", "moca19_delayed_recall_church",
                      "moca20_delayed_recall_daisy", "moca21_delayed_recall_red"]
        dere_idxs = [feature_map[ft] for ft in dere_codes if ft in feature_map]

        dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
        for idx, row in dataframe.iterrows():
            PATNO = row['participant_id']
            if PATNO not in seq_data:
                print("Error: PATNO %s not in sequence data!" % PATNO)
                return
            data_mtx = seq_data[PATNO]
            if visit >= len(data_mtx):
                continue

            EDUCYRS = static_df.loc[PATNO, 'education_level_years']  # education years

            if len(moca_codes) == len(moca_idxs):
                moca = data_mtx[visit, moca_idxs].sum()
            else:
                moca = np.nan
            if len(visu_codes) == len(visu_idxs):
                visu = data_mtx[visit, visu_idxs].sum()
            else:
                visu = np.nan
            if len(nam_codes) == len(nam_idxs):
                nam = data_mtx[visit, nam_idxs].sum()
            else:
                nam = np.nan
            if len(att_codes) == len(att_idxs):
                att = data_mtx[visit, att_idxs].sum()
            else:
                att = np.nan
            if len(lang_codes) == len(lang_idxs):
                lang = data_mtx[visit, lang_idxs].sum()
            else:
                lang = np.nan
            if len(dere_codes) == len(dere_idxs):
                dere = data_mtx[visit, dere_idxs].sum()
            else:
                dere = np.nan

            adj_moca = moca
            if not np.isnan(moca):
                if EDUCYRS == "Less than 12 years" and moca < 30:
                    adj_moca = moca + 1
                else:
                    adj_moca = moca

            dataframe.loc[idx, target_vars] = [adj_moca, visu, nam, att, lang, dere]

        print("MOCA domain finished!")
        return dataframe

    def RBD(self, dataframe, seq_data, feature_map, visit):
        target_vars = ['RBD']

        term1_codes = ["code_rbd01_vivid_dreams", "code_rbd02_aggressive_or_action_packed_dreams",
                       "code_rbd03_nocturnal_behaviour", "code_rbd04_move_arms_legs_during_sleep",
                       "code_rbd05_hurt_bed_partner", "code_rbd06_1_speaking_in_sleep",
                       "code_rbd06_2_sudden_limb_movements", "code_rbd06_3_complex_movements",
                       "code_rbd06_4_things_fell_down", "code_rbd07_my_movements_awake_me",
                       "code_rbd08_remember_dreams", "code_rbd09_sleep_is_disturbed"]
        term2_codes = ["code_rbd10a_stroke", "code_rbd10b_head_trauma", "code_rbd10c_parkinsonism", "code_rbd10d_rls",
                       "code_rbd10e_narcolepsy", "code_rbd10f_depression", "code_rbd10g_epilepsy",
                       "code_rbd10h_brain_inflammatory_disease", "code_rbd10i_other"]
        term1_idxs = [feature_map[ft] for ft in term1_codes if ft in feature_map]
        term2_idxs = [feature_map[ft] for ft in term2_codes if ft in feature_map]

        dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
        for idx, row in dataframe.iterrows():
            PATNO = row['participant_id']
            if PATNO not in seq_data:
                print("Error: PATNO %s not in sequence data!" % PATNO)
                return
            data_mtx = seq_data[PATNO]
            if visit >= len(data_mtx):
                continue

            if len(term1_codes) == len(term1_idxs):
                term1 = data_mtx[visit, term1_idxs].sum()
            else:
                term1 = np.nan
            if len(term2_codes) == len(term2_idxs):
                term2 = data_mtx[visit, term2_idxs].sum()
            else:
                term2 = np.nan

            if (not np.isnan(term1)) and (not np.isnan(term2)):
                rbd = term1
                if term2 >= 1:
                    rbd += 1
            else:
                rbd = term1

            dataframe.loc[idx, target_vars] = [rbd]

        print("RBD domain finished!")
        return dataframe

    def UPDRS2(self, dataframe, seq_data, feature_map, visit):
        target_vars = ['updrs2']
        updrs2_codes = ["code_upd2201_speech", "code_upd2202_saliva_and_drooling",
                        "code_upd2203_chewing_and_swallowing", "code_upd2204_eating_tasks",
                        "code_upd2205_dressing", "code_upd2206_hygiene", "code_upd2207_handwriting",
                        "code_upd2208_doing_hobbies_and_other_activities", "code_upd2209_turning_in_bed",
                        "code_upd2210_tremor", "code_upd2211_get_out_of_bed_car_or_deep_chair",
                        "code_upd2212_walking_and_balance", "code_upd2213_freezing"]
        updrs2_idxs = [feature_map[ft] for ft in updrs2_codes if ft in feature_map]

        dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
        for idx, row in dataframe.iterrows():
            PATNO = row['participant_id']
            if PATNO not in seq_data:
                print("Error: PATNO %s not in sequence data!" % PATNO)
                return
            data_mtx = seq_data[PATNO]
            if visit >= len(data_mtx):
                continue

            if len(updrs2_codes) == len(updrs2_idxs):
                updrs2_score = data_mtx[visit, updrs2_idxs].sum()
            else:
                updrs2_score = np.nan
            dataframe.loc[idx, target_vars] = [updrs2_score]

        print("UPDRS2 domain finished!")
        return dataframe

    def UPDRS3(self, dataframe, seq_data, feature_map, visit):
        target_vars = ['updrs3', 'H&Y stage']
        updrs3_codes = ["code_upd2301_speech_problems", "code_upd2302_facial_expression", "code_upd2303a_rigidity_neck",
                        "code_upd2303b_rigidity_rt_upper_extremity", "code_upd2303c_rigidity_left_upper_extremity",
                        "code_upd2303d_rigidity_rt_lower_extremity", "code_upd2303e_rigidity_left_lower_extremity",
                        "code_upd2304a_right_finger_tapping", "code_upd2304b_left_finger_tapping",
                        "code_upd2305a_right_hand_movements", "code_upd2305b_left_hand_movements",
                        "code_upd2306a_pron_sup_movement_right_hand", "code_upd2306b_pron_sup_movement_left_hand",
                        "code_upd2307a_right_toe_tapping", "code_upd2307b_left_toe_tapping",
                        "code_upd2308a_right_leg_agility", "code_upd2308b_left_leg_agility",
                        "code_upd2309_arising_from_chair", "code_upd2310_gait", "code_upd2311_freezing_of_gait",
                        "code_upd2312_postural_stability", "code_upd2313_posture", "code_upd2314_body_bradykinesia",
                        "code_upd2315a_postural_tremor_of_right_hand", "code_upd2315b_postural_tremor_of_left_hand",
                        "code_upd2316a_kinetic_tremor_of_right_hand", "code_upd2316b_kinetic_tremor_of_left_hand",
                        "code_upd2317a_rest_tremor_amplitude_right_upper_extremity",
                        "code_upd2317b_rest_tremor_amplitude_left_upper_extremity",
                        "code_upd2317c_rest_tremor_amplitude_right_lower_extremity",
                        "code_upd2317d_rest_tremor_amplitude_left_lower_extremity",
                        "code_upd2317e_rest_tremor_amplitude_lip_or_jaw", "code_upd2318_consistency_of_rest_tremor"]
        updrs3_idxs = [feature_map[ft] for ft in updrs3_codes if ft in feature_map]

        hy_idx = feature_map["code_upd2hy_hoehn_and_yahr_stage"]

        dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
        for idx, row in dataframe.iterrows():
            PATNO = row['participant_id']
            if PATNO not in seq_data:
                print("Error: PATNO %s not in sequence data!" % PATNO)
                return
            data_mtx = seq_data[PATNO]
            if visit >= len(data_mtx):
                continue

            if len(updrs3_codes) == len(updrs3_idxs):
                updrs3_score = data_mtx[visit, updrs3_idxs].sum()
            else:
                updrs3_score = np.nan
            hy = data_mtx[visit, hy_idx]
            dataframe.loc[idx, target_vars] = [updrs3_score, hy]

        print("UPDRS3 domain finished!")
        return dataframe

    def Schwab(self, dataframe, seq_data, feature_map, visit):
        target_vars = ['Schwab']
        schwab_codes = ["mod_schwab_england_pct_adl_score"]
        schwab_idxs = [feature_map[ft] for ft in schwab_codes if ft in feature_map]

        dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
        for idx, row in dataframe.iterrows():
            PATNO = row['participant_id']
            if PATNO not in seq_data:
                print("Error: PATNO %s not in sequence data!" % PATNO)
                return
            data_mtx = seq_data[PATNO]
            if visit >= len(data_mtx):
                continue

            if len(schwab_codes) == len(schwab_idxs):
                schwab_score = data_mtx[visit, schwab_idxs].sum()
            else:
                schwab_score = np.nan
            dataframe.loc[idx, target_vars] = [schwab_score]

        print("Schwab domain finished!")
        return dataframe

    def Tremor_PIGD(self, dataframe, seq_data, feature_map, visit):
        TD_PIGD_vars = ['Tremor_score', 'PIGD_score', 'is_TD', 'is_Intermediate', 'is_PIGD']
        tremor_codes = [fea_dict['NP2TRMR'], fea_dict['NP3PTRMR'], fea_dict['NP3PTRML'], fea_dict['NP3KTRMR'],
                        fea_dict['NP3KTRML'], fea_dict['NP3RTARU'], fea_dict['NP3RTALU'], fea_dict['NP3RTARL'],
                        fea_dict['NP3RTALL'], fea_dict['NP3RTALJ'], fea_dict['NP3RTCON']]
        tremor_idxs = [feature_map[ft] for ft in tremor_codes if ft in feature_map]
        pigd_codes = [fea_dict['NP2WALK'], fea_dict['NP2FREZ'], fea_dict['NP3GAIT'], fea_dict['NP3FRZGT'],
                      fea_dict['NP3PSTBL']]
        pigd_idxs = [feature_map[ft] for ft in pigd_codes if ft in feature_map]
        dataframe = dataframe.reindex(columns=list(dataframe.columns) + TD_PIGD_vars)
        for idx, row in dataframe.iterrows():
            PATNO = row['participant_id']
            if PATNO not in seq_data:
                print("Error: PATNO %s not in sequence data!" % PATNO)
                return
            data_mtx = seq_data[PATNO]
            if visit >= len(data_mtx):
                continue

            tremor_score = np.nanmean(data_mtx[visit, tremor_idxs])

            pigd_score = np.nanmean(data_mtx[visit, pigd_idxs])

            if pigd_score == 0:
                ratio = 1
            else:
                ratio = tremor_score / pigd_score
            is_TD = is_Intermediate = is_PIGD = 0

            if (ratio > 1.15) or (pigd_score == 0 and tremor_score > 0):
                is_TD = 1
            elif ratio <= 0.9:
                is_PIGD = 1
            else:
                is_Intermediate = 1

            dataframe.loc[idx, TD_PIGD_vars] = [tremor_score, pigd_score, is_TD, is_Intermediate, is_PIGD]

        print("Tremor_PIGD domain finished!")
        return dataframe

    # def Bio(self, dataframe, seq_data, feature_map, visit):
    #     target_vars = ["Tau", "Abeta", "p-Tau", "abeta_42_total_tau_ratio"]
    #
    #     abeta_42_idxs = [feature_map[ft] for ft in ["Abeta"]]
    #     p_tau181p_idxs = [feature_map[ft] for ft in ["p-Tau"]]
    #     total_tau_idxs = [feature_map[ft] for ft in ["Tau"]]
    #
    #     dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
    #     for idx, row in dataframe.iterrows():
    #         PATNO = row['participant_id']
    #         if PATNO not in seq_data:
    #             print("Error: PATNO %s not in sequence data!" % PATNO)
    #             return
    #         data_mtx = seq_data[PATNO]
    #         if visit >= len(data_mtx):
    #             continue
    #
    #         abeta_42 = data_mtx[visit, abeta_42_idxs].sum()
    #         p_tau181p = data_mtx[visit, p_tau181p_idxs].sum()
    #         total_tau = data_mtx[visit, total_tau_idxs].sum()
    #
    #         abeta_42_total_tau_ratio = abeta_42 / total_tau
    #
    #         dataframe.loc[idx, target_vars] = [abeta_42, p_tau181p, total_tau,
    #                                            abeta_42_total_tau_ratio]
    #     print("Biospecimen domain finished!")
    #     return dataframe

    def get_var_table(self, feature_list, static_data, seq_data, data_path=None):

        feat_map = {}
        idx = 0
        for feat in feature_list:
            feat_map[feat] = idx
            idx += 1

        static_df = static_data.set_index('participant_id')

        df = pd.DataFrame(data=list(seq_data.keys()), columns=['participant_id'])

        # initialize
        tables = {}
        for v in range(11):
            tables[v] = copy.deepcopy(df)

        # update
        if data_path is not None:
            folder = data_path

            if os.path.exists(folder) == False:
                os.mkdir(folder)
            for v in tables.keys():
                tables[v] = self.UPDRS1(tables[v], seq_data, feat_map, v)
                tables[v] = self.Epworth(tables[v], seq_data, feat_map, v)
                tables[v] = self.MOCA(tables[v], seq_data, static_df, feat_map, v)  # adjusted
                tables[v] = self.RBD(tables[v], seq_data, feat_map, v)

                tables[v] = self.UPDRS2(tables[v], seq_data, feat_map, v)
                tables[v] = self.UPDRS3(tables[v], seq_data, feat_map, v)
                tables[v] = self.Schwab(tables[v], seq_data, feat_map, v)
                tables[v] = self.Tremor_PIGD(tables[v], seq_data, feat_map, v)
                # tables[v] = self.Bio(tables[v], seq_data, feat_map, v)

                tables[v].to_csv(folder + '/' + 'V%02d.csv' % v, index=False)
                print("------------ Visit %s finished! -------------" % v)
        else:
            for v in tables.keys():
                tables[v] = self.UPDRS1(tables[v], seq_data, feat_map, v)
                tables[v] = self.Epworth(tables[v], seq_data, feat_map, v)
                tables[v] = self.MOCA(tables[v], seq_data, static_df, feat_map, v)  # adjusted
                tables[v] = self.RBD(tables[v], seq_data, feat_map, v)

                tables[v] = self.UPDRS2(tables[v], seq_data, feat_map, v)
                tables[v] = self.UPDRS3(tables[v], seq_data, feat_map, v)
                tables[v] = self.Schwab(tables[v], seq_data, feat_map, v)
                tables[v] = self.Tremor_PIGD(tables[v], seq_data, feat_map, v)
                # tables[v] = self.Bio(tables[v], seq_data, feat_map, v)
                print("------------ Visit %s finished! -------------" % v)
        print("Generate all tables successfully!")
        return tables


class Evaluation:

    def __init__(self, label_df, save_path):
        labels = label_df['Agglomerative'].values
        self.labels = labels
        self.label_df = label_df
        self.save_path = save_path

    def cluster_visualization(self, X):
        C = len(set(self.labels))
        for c in range(C):
            plt.scatter(X[self.labels == c, 0], X[self.labels == c, 1], label='subtype_' + str(c))
        plt.legend()
        plt.show()
        if self.save_path is not None:
            if os.path.exists(self.save_path) == False:
                os.mkdir(self.save_path)
            plt.savefig(self.save_path + "/" + "Agglomerative_tsne.png")
        plt.close()

    def progression_plot(self, visit_tables, displayed_features, displayed_visits=[0, 2, 4, 6, 8, 10]):
        visit_data = pd.DataFrame()
        for visit in displayed_visits:
            v_table = visit_tables[visit]
            v_table['visit'] = visit
            visit_data = pd.concat([visit_data, v_table], ignore_index=True)

        data_set = {}
        for l in set(list(self.labels)):
            temp_df = pd.merge(visit_data, self.label_df.loc[self.label_df['Agglomerative'] == l], on='participant_id')
            data_set['subtype_' + str(l)] = temp_df

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 10))
        color_list = plt.get_cmap("Set1")
        for idx, feature in enumerate(displayed_features):
            for i, data_name in enumerate(data_set):
                line_data = []
                line_upper = []
                line_lower = []
                for visit in displayed_visits:
                    temp_data = data_set[data_name].loc[data_set[data_name]['visit'] == visit]
                    mean_data = np.nanmean(temp_data[feature])
                    std_data = scipy.stats.sem(temp_data.dropna(subset=[feature])[feature])
                    line_data.append(mean_data)
                    line_upper.append(mean_data + std_data)
                    line_lower.append(mean_data - std_data)

                line_data = np.array(line_data).astype(np.double)
                line_lower = np.array(line_lower).astype(np.double)
                line_upper = np.array(line_upper).astype(np.double)
                xs = np.arange(len(displayed_visits))
                if np.isnan(line_data).any():
                    line_data_mask = np.isfinite(line_data)
                    line_lower_mask = np.isfinite(line_lower)
                    line_upper_mask = np.isfinite(line_upper)
                    ax[int(idx / 3), idx % 3].plot(xs[line_data_mask], line_data[line_data_mask], color=color_list(i),
                                                   label=data_name,
                                                   marker="o")
                    ax[int(idx / 3), idx % 3].fill_between(xs[line_data_mask], line_lower[line_lower_mask],
                                                           line_upper[line_upper_mask],
                                                           color=color_list(i), alpha=.2)
                else:
                    ax[int(idx / 3), idx % 3].plot(xs, line_data, color=color_list(i), label=data_name, marker="o")
                    ax[int(idx / 3), idx % 3].fill_between(xs, line_lower, line_upper, color=color_list(i), alpha=.2)
                ax[int(idx / 3), idx % 3].set_title(feature)
                plt.legend()
        plt.show()
        plt.close()
        # if self.save_path is not None:
        #     folder = self.save_path
        #     if not os.path.exists(folder):
        #         os.mkdir(folder)
        #     plt.savefig(folder + "/progression.png")
        # plt.close()


def main():
    with open('[your directory]/validation/processed_data/feature_list.txt', 'rb') as f:
        feature_list = pkl.load(f)
    f.close()
    static_data = pd.read_csv('[your directory]/validation/processed_data/static_info.csv')
    with open('[your directory]/PPMI/validation/processed_data/sequence_data.pkl', 'rb') as f:
        sequence_data = pkl.load(f)
    f.close()
    path = '[your directory]/PPMI/validation/non_imputed_visit_2'
    ET = EvaluationTables(feature_list, static_data, sequence_data, path)
    tables = ET.evaluation_tables()
    with open('[your directory]/validation/processed_data/visit_tables.obj', 'wb') as f:
        pkl.dump(tables, f)
    f.close()

    save_path = "[your directory]/validation/evaluation_result"
    label_df = pd.read_csv('[your directory]/validation/processed_data/label.csv')
    with open('[your directory]/validation/processed_data/tsne_data.obj', 'rb') as f:
        tsne_data = pkl.load(f)
    f.close()

    EV = Evaluation(label_df, save_path)
    EV.cluster_visualization(tsne_data)

    displayed_features = ['updrs1', 'updrs2', 'updrs3', 'Schwab', 'moca', 'epworth']
    EV.progression_plot(tables, displayed_features)


if __name__ == '__main__':
    main()
