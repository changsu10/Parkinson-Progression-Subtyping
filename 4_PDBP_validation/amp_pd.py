"""
_*_ coding: utf-8 _*_
"""

import pandas as pd
import numpy as np
import os
import pickle as pkl
import json
import copy
import re

data_folder = '[your directory]/AMP-PD/2019_v1release_1015/'


class DataPreparation:

    def __init__(self, cohort_name, table_list):
        self.table_list = table_list

        participant_df = pd.read_csv(data_folder + 'amp_pd_participants.csv')
        self.patient_list = participant_df[participant_df['study'] == cohort_name]['participant_id'].tolist()

        participant_df = participant_df[participant_df['participant_id'].isin(self.patient_list)]
        demo_df = pd.read_csv(data_folder + 'Demographics.csv')
        control_df = pd.read_csv(data_folder + 'amp_pd_case_control.csv')
        participant_df = pd.merge(participant_df, demo_df, on='participant_id')
        participant_df = pd.merge(participant_df, control_df, on='participant_id')
        self.participant_df = participant_df

        # Load assessment data
        history_df = pd.read_csv(data_folder + 'Family_History_PD.csv')
        updrs1_df = pd.read_csv(data_folder + 'MDS_UPDRS_Part_I.csv')
        updrs2_df = pd.read_csv(data_folder + 'MDS_UPDRS_Part_II.csv')
        updrs3_df = pd.read_csv(data_folder + 'MDS_UPDRS_Part_III.csv')
        updrs4_df = pd.read_csv(data_folder + 'MDS_UPDRS_Part_IV.csv')
        schwab_df = pd.read_csv(data_folder + 'Modified_Schwab___England_ADL.csv')
        moca_df = pd.read_csv(data_folder + 'MOCA.csv')
        upsit_df = pd.read_csv(data_folder + 'UPSIT.csv')
        ess_df = pd.read_csv(data_folder + 'Epworth_Sleepiness_Scale.csv')
        rbd_df = pd.read_csv(data_folder + 'REM_Sleep_Behavior_Disorder_Questionnaire_Stiasny_Kolster.csv')
        medication_df = pd.read_csv(data_folder + 'PD_Medical_History.csv')
        bio_df = pd.read_csv(data_folder + 'Biospecimen_analyses_CSF_abeta_tau_ptau.csv')

        self.history_df = history_df.loc[history_df['participant_id'].isin(self.patient_list)]
        self.updrs1_df = updrs1_df.loc[updrs1_df['participant_id'].isin(self.patient_list)]
        self.updrs2_df = updrs2_df.loc[updrs2_df['participant_id'].isin(self.patient_list)]
        self.updrs3_df = updrs3_df.loc[updrs3_df['participant_id'].isin(self.patient_list)]
        self.updrs4_df = updrs4_df.loc[updrs4_df['participant_id'].isin(self.patient_list)]
        self.schwab_df = schwab_df.loc[schwab_df['participant_id'].isin(self.patient_list)]
        self.moca_df = moca_df.loc[moca_df['participant_id'].isin(self.patient_list)]
        self.upsit_df = upsit_df.loc[upsit_df['participant_id'].isin(self.patient_list)]
        self.ess_df = ess_df.loc[ess_df['participant_id'].isin(self.patient_list)]
        self.rbd_df = rbd_df.loc[rbd_df['participant_id'].isin(self.patient_list)]
        self.medication_df = medication_df.loc[medication_df['participant_id'].isin(self.patient_list)]
        self.bio_df = bio_df.loc[bio_df['participant_id'].isin(self.patient_list)]

        demographics = self.extract_demographics()
        history = self.extract_history()
        updrs1 = self.extract_updrs1()
        updrs2 = self.extract_updrs2()
        updrs3 = self.extract_updrs3()
        updrs4 = self.extract_updrs4()
        schwab = self.extract_schwab()
        moca = self.extract_moca()
        upsit = self.extract_upsit()
        ess = self.extract_ess()
        rbd = self.extract_rbd()
        med = self.extract_medication()
        bio = self.extract_bio()

        self.table_dict = {"demographics": demographics, "history": history, "updrs1": updrs1, "updrs2": updrs2,
                           "updrs3": updrs3, "updrs4": updrs4, "schwab": schwab, "moca": moca, "upsit": upsit,
                           "ess": ess, "rbd": rbd, "med": med, "bio": bio}

        self.feature_knowledge = ["code_upd2101_cognitive_impairment", "code_upd2102_hallucinations_and_psychosis",
                                  "code_upd2103_depressed_mood",
                                  "code_upd2104_anxious_mood", "code_upd2105_apathy",
                                  "code_upd2106_dopamine_dysregulation_syndrome_features",
                                  "code_upd2107_pat_quest_sleep_problems", "code_upd2108_pat_quest_daytime_sleepiness",
                                  "code_upd2109_pat_quest_pain_and_other_sensations",
                                  "code_upd2110_pat_quest_urinary_problems",
                                  "code_upd2111_pat_quest_constipation_problems",
                                  "code_upd2112_pat_quest_lightheadedness_on_standing",
                                  "code_upd2113_pat_quest_fatigue", "code_upd2201_speech",
                                  "code_upd2202_saliva_and_drooling", "code_upd2203_chewing_and_swallowing",
                                  "code_upd2204_eating_tasks", "code_upd2205_dressing", "code_upd2206_hygiene",
                                  "code_upd2207_handwriting",
                                  "code_upd2208_doing_hobbies_and_other_activities", "code_upd2209_turning_in_bed",
                                  "code_upd2210_tremor",
                                  "code_upd2211_get_out_of_bed_car_or_deep_chair", "code_upd2212_walking_and_balance",
                                  "code_upd2213_freezing",
                                  "code_upd2301_speech_problems", "code_upd2302_facial_expression",
                                  "code_upd2303a_rigidity_neck",
                                  "code_upd2303b_rigidity_rt_upper_extremity",
                                  "code_upd2303c_rigidity_left_upper_extremity",
                                  "code_upd2303d_rigidity_rt_lower_extremity",
                                  "code_upd2303e_rigidity_left_lower_extremity",
                                  "code_upd2304a_right_finger_tapping", "code_upd2304b_left_finger_tapping",
                                  "code_upd2305a_right_hand_movements",
                                  "code_upd2305b_left_hand_movements", "code_upd2306a_pron_sup_movement_right_hand",
                                  "code_upd2306b_pron_sup_movement_left_hand",
                                  "code_upd2307a_right_toe_tapping", "code_upd2307b_left_toe_tapping",
                                  "code_upd2308a_right_leg_agility",
                                  "code_upd2308b_left_leg_agility", "code_upd2309_arising_from_chair",
                                  "code_upd2310_gait", "code_upd2311_freezing_of_gait",
                                  "code_upd2312_postural_stability", "code_upd2313_posture",
                                  "code_upd2314_body_bradykinesia",
                                  "code_upd2315a_postural_tremor_of_right_hand",
                                  "code_upd2315b_postural_tremor_of_left_hand",
                                  "code_upd2316a_kinetic_tremor_of_right_hand",
                                  "code_upd2316b_kinetic_tremor_of_left_hand",
                                  "code_upd2317a_rest_tremor_amplitude_right_upper_extremity",
                                  "code_upd2317b_rest_tremor_amplitude_left_upper_extremity",
                                  "code_upd2317c_rest_tremor_amplitude_right_lower_extremity",
                                  "code_upd2317d_rest_tremor_amplitude_left_lower_extremity",
                                  "code_upd2317e_rest_tremor_amplitude_lip_or_jaw",
                                  "code_upd2318_consistency_of_rest_tremor",
                                  'code_upd2hy_hoehn_and_yahr_stage',
                                  "mod_schwab_england_pct_adl_score",
                                  "moca01_alternating_trail_making", "moca02_visuoconstr_skills_cube",
                                  "moca03_visuoconstr_skills_clock_cont",
                                  "moca04_visuoconstr_skills_clock_num", "moca05_visuoconstr_skills_clock_hands",
                                  "moca06_naming_lion",
                                  "moca07_naming_rhino", "moca08_naming_camel", "moca09_attention_forward_digit_span",
                                  "moca10_attention_backward_digit_span", "moca11_attention_vigilance",
                                  "moca12_attention_serial_7s",
                                  "moca13_sentence_repetition", "moca14_verbal_fluency_number_of_words",
                                  "moca15_verbal_fluency",
                                  "moca16_abstraction", "moca17_delayed_recall_face", "moca18_delayed_recall_velvet",
                                  "moca19_delayed_recall_church",
                                  "moca20_delayed_recall_daisy", "moca21_delayed_recall_red",
                                  "moca22_orientation_date_score",
                                  "moca23_orientation_month_score", "moca24_orientation_year_score",
                                  "moca25_orientation_day_score",
                                  "moca26_orientation_place_score", "moca27_orientation_city_score", "moca_total_score",
                                  "code_ess0101_sitting_and_reading", "code_ess0102_watching_tv",
                                  "code_ess0103_sitting_inactive_in_public_place",
                                  "code_ess0104_passenger_in_car_for_hour",
                                  "code_ess0105_lying_down_to_rest_in_afternoon",
                                  "code_ess0106_sitting_and_talking_to_someone", "code_ess0107_sitting_after_lunch",
                                  "code_ess0108_car_stopped_in_traffic",
                                  "code_rbd01_vivid_dreams", "code_rbd02_aggressive_or_action_packed_dreams",
                                  "code_rbd03_nocturnal_behaviour",
                                  "code_rbd04_move_arms_legs_during_sleep", "code_rbd05_hurt_bed_partner",
                                  "code_rbd06_1_speaking_in_sleep",
                                  "code_rbd06_2_sudden_limb_movements", "code_rbd06_3_complex_movements",
                                  "code_rbd06_4_things_fell_down",
                                  "code_rbd07_my_movements_awake_me", "code_rbd08_remember_dreams",
                                  "code_rbd09_sleep_is_disturbed",
                                  "code_rbd10a_stroke", "code_rbd10b_head_trauma", "code_rbd10c_parkinsonism",
                                  "code_rbd10d_rls", "code_rbd10e_narcolepsy",
                                  "code_rbd10f_depression", "code_rbd10g_epilepsy",
                                  "code_rbd10h_brain_inflammatory_disease", "code_rbd10i_other",
                                  "Tau", "Abeta", "p-Tau"]

    def extract_demographics(self):
        features = ["age_at_baseline", "sex", "race", "education_level_years", "diagnosis_at_baseline",
                    "diagnosis_latest"]
        demographics = self.participant_df[["participant_id", "visit_name", "visit_month"] + features]
        res_df = pd.DataFrame()
        for f in features:
            if demographics[f].isnull().values.all():
                continue
            else:
                temp_df = pd.DataFrame({"participant_id": demographics["participant_id"].values.tolist(),
                                        "visit_name": demographics["visit_name"].values.tolist(),
                                        "visit_month": demographics["visit_month"].values.tolist(),
                                        "variable": [f] * len(demographics),
                                        "var_type": ['subject_characteristics'] * len(demographics),
                                        "source": ['demographics'] * len(demographics),
                                        "value": demographics[f].values.tolist()})
                res_df = pd.concat([res_df, temp_df], ignore_index=True)
        return res_df

    def extract_history(self):
        features = ["biological_mother_with_pd", "biological_father_with_pd", "other_relative_with_pd"]
        history = self.history_df[["participant_id", "visit_name", "visit_month"] + features]
        res_df = pd.DataFrame()
        for f in features:
            if history[f].isnull().values.all():
                continue
            else:
                temp_df = pd.DataFrame({"participant_id": history["participant_id"].values.tolist(),
                                        "visit_name": history["visit_name"].values.tolist(),
                                        "visit_month": history["visit_month"].values.tolist(),
                                        "variable": [f] * len(history),
                                        "var_type": ['subject_characteristics'] * len(history),
                                        "source": ['family_history'] * len(history),
                                        "value": history[f].values.tolist()})
                res_df = pd.concat([res_df, temp_df], ignore_index=True)
        return res_df

    def extract_updrs1(self):
        features = ["code_upd2101_cognitive_impairment", "code_upd2102_hallucinations_and_psychosis",
                    "code_upd2103_depressed_mood",
                    "code_upd2104_anxious_mood", "code_upd2105_apathy",
                    "code_upd2106_dopamine_dysregulation_syndrome_features",
                    "code_upd2107_pat_quest_sleep_problems", "code_upd2108_pat_quest_daytime_sleepiness",
                    "code_upd2109_pat_quest_pain_and_other_sensations", "code_upd2110_pat_quest_urinary_problems",
                    "code_upd2111_pat_quest_constipation_problems",
                    "code_upd2112_pat_quest_lightheadedness_on_standing",
                    "code_upd2113_pat_quest_fatigue"]
        updrs1 = self.updrs1_df[["participant_id", "visit_name", "visit_month"] + features]
        res_df = pd.DataFrame()
        for f in features:
            if updrs1[f].isnull().values.all():
                continue
            else:
                temp_df = pd.DataFrame({"participant_id": updrs1["participant_id"].values.tolist(),
                                        "visit_name": updrs1["visit_name"].values.tolist(),
                                        "visit_month": updrs1["visit_month"].values.tolist(),
                                        "variable": [f] * len(updrs1),
                                        "var_type": ['motor'] * len(updrs1),
                                        "source": ['updrs1'] * len(updrs1),
                                        "value": updrs1[f].values.tolist()})
                res_df = pd.concat([res_df, temp_df], ignore_index=True)
        return res_df

    def extract_updrs2(self):
        features = ["code_upd2201_speech", "code_upd2202_saliva_and_drooling", "code_upd2203_chewing_and_swallowing",
                    "code_upd2204_eating_tasks", "code_upd2205_dressing", "code_upd2206_hygiene",
                    "code_upd2207_handwriting",
                    "code_upd2208_doing_hobbies_and_other_activities", "code_upd2209_turning_in_bed",
                    "code_upd2210_tremor",
                    "code_upd2211_get_out_of_bed_car_or_deep_chair", "code_upd2212_walking_and_balance",
                    "code_upd2213_freezing"]
        updrs2 = self.updrs2_df[["participant_id", "visit_name", "visit_month"] + features]
        res_df = pd.DataFrame()
        for f in features:
            if updrs2[f].isnull().values.all():
                continue
            else:
                temp_df = pd.DataFrame({"participant_id": updrs2["participant_id"].values.tolist(),
                                        "visit_name": updrs2["visit_name"].values.tolist(),
                                        "visit_month": updrs2["visit_month"].values.tolist(),
                                        "variable": [f] * len(updrs2),
                                        "var_type": ['motor'] * len(updrs2),
                                        "source": ['updrs2'] * len(updrs2),
                                        "value": updrs2[f].values.tolist()})
                res_df = pd.concat([res_df, temp_df], ignore_index=True)
        return res_df

    def extract_updrs3(self):
        features = ["code_upd2301_speech_problems", "code_upd2302_facial_expression", "code_upd2303a_rigidity_neck",
                    "code_upd2303b_rigidity_rt_upper_extremity", "code_upd2303c_rigidity_left_upper_extremity",
                    "code_upd2303d_rigidity_rt_lower_extremity", "code_upd2303e_rigidity_left_lower_extremity",
                    "code_upd2304a_right_finger_tapping", "code_upd2304b_left_finger_tapping",
                    "code_upd2305a_right_hand_movements",
                    "code_upd2305b_left_hand_movements", "code_upd2306a_pron_sup_movement_right_hand",
                    "code_upd2306b_pron_sup_movement_left_hand",
                    "code_upd2307a_right_toe_tapping", "code_upd2307b_left_toe_tapping",
                    "code_upd2308a_right_leg_agility",
                    "code_upd2308b_left_leg_agility", "code_upd2309_arising_from_chair", "code_upd2310_gait",
                    "code_upd2311_freezing_of_gait",
                    "code_upd2312_postural_stability", "code_upd2313_posture", "code_upd2314_body_bradykinesia",
                    "code_upd2315a_postural_tremor_of_right_hand", "code_upd2315b_postural_tremor_of_left_hand",
                    "code_upd2316a_kinetic_tremor_of_right_hand", "code_upd2316b_kinetic_tremor_of_left_hand",
                    "code_upd2317a_rest_tremor_amplitude_right_upper_extremity",
                    "code_upd2317b_rest_tremor_amplitude_left_upper_extremity",
                    "code_upd2317c_rest_tremor_amplitude_right_lower_extremity",
                    "code_upd2317d_rest_tremor_amplitude_left_lower_extremity",
                    "code_upd2317e_rest_tremor_amplitude_lip_or_jaw", "code_upd2318_consistency_of_rest_tremor",
                    'code_upd2hy_hoehn_and_yahr_stage']
        updrs3 = self.updrs3_df[["participant_id", "visit_name", "visit_month"] + features]
        res_df = pd.DataFrame()
        for f in features:
            if updrs3[f].isnull().values.all():
                continue
            else:
                temp_df = pd.DataFrame({"participant_id": updrs3["participant_id"].values.tolist(),
                                        "visit_name": updrs3["visit_name"].values.tolist(),
                                        "visit_month": updrs3["visit_month"].values.tolist(),
                                        "variable": [f] * len(updrs3),
                                        "var_type": ['motor'] * len(updrs3),
                                        "source": ['updrs3'] * len(updrs3),
                                        "value": updrs3[f].values.tolist()})
                res_df = pd.concat([res_df, temp_df], ignore_index=True)
        return res_df

    def extract_updrs4(self):
        features = ["code_upd2401_time_spent_with_dyskinesias", "code_upd2402_functional_impact_of_dyskinesias",
                    "code_upd2403_time_spent_in_the_off_state", "code_upd2404_functional_impact_of_fluctuations",
                    "code_upd2405_complexity_of_motor_fluctuations", "code_upd2406_painful_off_state_dystonia"]
        updrs4 = self.updrs4_df[["participant_id", "visit_name", "visit_month"] + features]
        res_df = pd.DataFrame()
        for f in features:
            if updrs4[f].isnull().values.all():
                continue
            else:
                temp_df = pd.DataFrame({"participant_id": updrs4["participant_id"].values.tolist(),
                                        "visit_name": updrs4["visit_name"].values.tolist(),
                                        "visit_month": updrs4["visit_month"].values.tolist(),
                                        "variable": [f] * len(updrs4),
                                        "var_type": ['motor'] * len(updrs4),
                                        "source": ['updrs4'] * len(updrs4),
                                        "value": updrs4[f].values.tolist()})
                res_df = pd.concat([res_df, temp_df], ignore_index=True)
        return res_df

    def extract_schwab(self):
        features = ["mod_schwab_england_pct_adl_score"]
        schwab = self.schwab_df[["participant_id", "visit_name", "visit_month"] + features]
        res_df = pd.DataFrame()
        for f in features:
            if schwab[f].isnull().values.all():
                continue
            else:
                temp_df = pd.DataFrame({"participant_id": schwab["participant_id"].values.tolist(),
                                        "visit_name": schwab["visit_name"].values.tolist(),
                                        "visit_month": schwab["visit_month"].values.tolist(),
                                        "variable": [f] * len(schwab),
                                        "var_type": ['motor'] * len(schwab),
                                        "source": ['schwab'] * len(schwab),
                                        "value": schwab[f].values.tolist()})
                res_df = pd.concat([res_df, temp_df], ignore_index=True)
        return res_df

    def extract_moca(self):
        features = ["moca01_alternating_trail_making", "moca02_visuoconstr_skills_cube",
                    "moca03_visuoconstr_skills_clock_cont",
                    "moca04_visuoconstr_skills_clock_num", "moca05_visuoconstr_skills_clock_hands",
                    "moca06_naming_lion",
                    "moca07_naming_rhino", "moca08_naming_camel", "moca09_attention_forward_digit_span",
                    "moca10_attention_backward_digit_span", "moca11_attention_vigilance", "moca12_attention_serial_7s",
                    "moca13_sentence_repetition", "moca14_verbal_fluency_number_of_words", "moca15_verbal_fluency",
                    "moca16_abstraction", "moca17_delayed_recall_face", "moca18_delayed_recall_velvet",
                    "moca19_delayed_recall_church",
                    "moca20_delayed_recall_daisy", "moca21_delayed_recall_red", "moca22_orientation_date_score",
                    "moca23_orientation_month_score", "moca24_orientation_year_score", "moca25_orientation_day_score",
                    "moca26_orientation_place_score", "moca27_orientation_city_score", "moca_total_score"]
        moca = self.moca_df[["participant_id", "visit_name", "visit_month"] + features]
        res_df = pd.DataFrame()
        for f in features:
            if moca[f].isnull().values.all():
                continue
            else:
                temp_df = pd.DataFrame({"participant_id": moca["participant_id"].values.tolist(),
                                        "visit_name": moca["visit_name"].values.tolist(),
                                        "visit_month": moca["visit_month"].values.tolist(),
                                        "variable": [f] * len(moca),
                                        "var_type": ['non_motor'] * len(moca),
                                        "source": ['moca'] * len(moca),
                                        "value": moca[f].values.tolist()})
                res_df = pd.concat([res_df, temp_df], ignore_index=True)
        return res_df

    def extract_upsit(self):
        features = ["score_from_booklet_1", "score_from_booklet_2", "score_from_booklet_3", "score_from_booklet_4"]
        upsit = self.upsit_df[["participant_id", "visit_name", "visit_month"] + features]
        res_df = pd.DataFrame()
        for f in features:
            if upsit[f].isnull().values.all():
                continue
            else:
                temp_df = pd.DataFrame({"participant_id": upsit["participant_id"].values.tolist(),
                                        "visit_name": upsit["visit_name"].values.tolist(),
                                        "visit_month": upsit["visit_month"].values.tolist(),
                                        "variable": [f] * len(upsit),
                                        "var_type": ['non_motor'] * len(upsit),
                                        "source": ['upsit'] * len(upsit),
                                        "value": upsit[f].values.tolist()})
                res_df = pd.concat([res_df, temp_df], ignore_index=True)
        return res_df

    def extract_ess(self):
        features = ["code_ess0101_sitting_and_reading", "code_ess0102_watching_tv",
                    "code_ess0103_sitting_inactive_in_public_place",
                    "code_ess0104_passenger_in_car_for_hour", "code_ess0105_lying_down_to_rest_in_afternoon",
                    "code_ess0106_sitting_and_talking_to_someone",
                    "code_ess0107_sitting_after_lunch", "code_ess0108_car_stopped_in_traffic"]
        ess = self.ess_df[["participant_id", "visit_name", "visit_month"] + features]
        res_df = pd.DataFrame()
        for f in features:
            if ess[f].isnull().values.all():
                continue
            else:
                temp_df = pd.DataFrame({"participant_id": ess["participant_id"].values.tolist(),
                                        "visit_name": ess["visit_name"].values.tolist(),
                                        "visit_month": ess["visit_month"].values.tolist(),
                                        "variable": [f] * len(ess),
                                        "var_type": ['non_motor'] * len(ess),
                                        "source": ['ess'] * len(ess),
                                        "value": ess[f].values.tolist()})
                res_df = pd.concat([res_df, temp_df], ignore_index=True)
        return res_df

    def extract_rbd(self):
        features = ["code_rbd01_vivid_dreams", "code_rbd02_aggressive_or_action_packed_dreams",
                    "code_rbd03_nocturnal_behaviour",
                    "code_rbd04_move_arms_legs_during_sleep", "code_rbd05_hurt_bed_partner",
                    "code_rbd06_1_speaking_in_sleep",
                    "code_rbd06_2_sudden_limb_movements", "code_rbd06_3_complex_movements",
                    "code_rbd06_4_things_fell_down",
                    "code_rbd07_my_movements_awake_me", "code_rbd08_remember_dreams", "code_rbd09_sleep_is_disturbed",
                    "code_rbd10a_stroke", "code_rbd10b_head_trauma", "code_rbd10c_parkinsonism", "code_rbd10d_rls",
                    "code_rbd10e_narcolepsy",
                    "code_rbd10f_depression", "code_rbd10g_epilepsy", "code_rbd10h_brain_inflammatory_disease",
                    "code_rbd10i_other"]
        rbd = self.rbd_df[["participant_id", "visit_name", "visit_month"] + features]
        res_df = pd.DataFrame()
        for f in features:
            if rbd[f].isnull().values.all():
                continue
            else:
                temp_df = pd.DataFrame({"participant_id": rbd["participant_id"].values.tolist(),
                                        "visit_name": rbd["visit_name"].values.tolist(),
                                        "visit_month": rbd["visit_month"].values.tolist(),
                                        "variable": [f] * len(rbd),
                                        "var_type": ['non_motor'] * len(rbd),
                                        "source": ['rbd'] * len(rbd),
                                        "value": rbd[f].values.tolist()})
                res_df = pd.concat([res_df, temp_df], ignore_index=True)
        return res_df

    def extract_medication(self):
        features = ["on_levodopa", "on_dopamine_agonist", "on_other_pd_medications"]
        medication = self.medication_df[["participant_id", "visit_name", "visit_month"] + features]
        res_df = pd.DataFrame()
        for f in features:
            if medication[f].isnull().values.all():
                continue
            else:
                temp_df = pd.DataFrame({"participant_id": medication["participant_id"].values.tolist(),
                                        "visit_name": medication["visit_name"].values.tolist(),
                                        "visit_month": medication["visit_month"].values.tolist(),
                                        "variable": [f] * len(medication),
                                        "var_type": ['medical'] * len(medication),
                                        "source": ['medication'] * len(medication),
                                        "value": medication[f].values.tolist()})
                res_df = pd.concat([res_df, temp_df], ignore_index=True)
        res_df = res_df.replace("Yes", 1)
        res_df = res_df.replace("No", 0)
        res_df = res_df.replace("None", np.nan)
        return res_df

    def extract_bio(self):
        bio = self.bio_df[["participant_id", "visit_name", "visit_month", "test_name", "test_value"]]
        res_df = pd.DataFrame({"participant_id": bio["participant_id"].values.tolist(),
                               "visit_name": bio["visit_name"].values.tolist(),
                               "visit_month": bio["visit_month"].values.tolist(),
                               "variable": bio["test_name"].values.tolist(),
                               "var_type": ['biospecimen'] * len(bio),
                               "source": ['CSF_abeta_tau_ptau'] * len(bio),
                               "value": bio["test_value"].values.tolist()})
        return res_df

    def query(self):
        new_dict = {}
        for table_name in self.table_list:
            new_dict[table_name] = self.table_dict[table_name]
        return new_dict

    # Save cohort's demographic information
    def static_data(self, save_path=None):
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path + '/static_info.csv', 'w') as f:
                f.write(self.participant_df.to_csv(index=False))
        return self.participant_df

    # Combine the data from the extracted assessment data
    def longitudinal_data(self, save_path=None):
        selected_table = self.query()
        res_df = pd.DataFrame()
        feature_list = []
        for table_name in selected_table:
            temp_feature_list = list(
                selected_table[table_name].drop_duplicates(subset='variable', keep='first')['variable'].values)
            feature_list = feature_list + temp_feature_list
            res_df = pd.concat([res_df, selected_table[table_name]], ignore_index=True)

        res_df = res_df.replace("Yes", 1)
        res_df = res_df.replace("No", 0)
        res_df = res_df.replace("None", np.nan)
        feature_list = list(set(self.feature_knowledge) & set(feature_list))

        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path + '/data.csv', 'w') as f:
                f.write(res_df.to_csv(index=False))
            with open(save_path + "/feature_list.txt", "wb") as fp:
                pkl.dump(feature_list, fp)

        return res_df, feature_list

    # Get the number of visit for each extracted assessment data
    def visit_list(self, save_path=None):
        visit_total = []
        for table_name in self.table_dict:
            visit_list = self.table_dict[table_name].drop_duplicates(subset='visit_name', keep='first')[
                'visit_name'].values.tolist()
            visit_total = list(set(visit_total + visit_list))
        visit_total.sort(key=self.natural_keys)
        # visit_total.remove("LOG")
        # visit_total.remove("SC")
        # visit_total.remove("SC#2")
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path + "/visit_list.txt", "wb") as fp:
                pkl.dump(visit_total, fp)

        return visit_total

    def visit_info(self, data, visit_list, save_path=None):

        visit_info_df = pd.DataFrame(self.patient_list, columns=["participant_id"])
        visit_info_df = visit_info_df.reindex(columns=['participant_id'] + visit_list + ['max_visit'])
        visit_info_df.set_index(['participant_id'], inplace=True)

        for idx, row in data.iterrows():
            pid, visit_name, var_type = row['participant_id'], row['visit_name'], row['var_type']
            if visit_name in visit_info_df.columns:
                visit_info_df.loc[pid, visit_name] = 1

        visit_info_df = visit_info_df.reset_index()

        for idx, row in visit_info_df.iterrows():
            max_visit = None
            for v in visit_list:
                if not np.isnan(row[v]):
                    max_visit = v
                visit_info_df.loc[idx, "max_visit"] = max_visit
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path + '/visit_info.csv', 'w') as f:
                f.write(visit_info_df.to_csv(index=False))

        return visit_info_df

    def concatenate(self, data_df, visit_info, visit_list, feature_list, save_path=None):

        visit_id_map = {}
        for v in range(len(visit_list)):
            visit_id_map[visit_list[v]] = v

        patient_length = {}
        for idx, row in visit_info.iterrows():
            pid, BL, max_visit = row['participant_id'], row['M0'], row['max_visit']
            if np.isnan(BL):  # check is there missing of BL visit, if so, exclude the patient
                print("!!!! Patient %s has no BL information!", pid)
            else:
                patient_length[pid] = visit_id_map[max_visit] + 1

        data_df = data_df.loc[data_df["participant_id"].isin(list(patient_length.keys()))]
        data_df = data_df.loc[data_df["visit_name"].isin(visit_list)]

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
        data_df = data_df[data_df['variable'].isin(feature_id_map)]  # exclude unused features
        for idx, row in data_df.iterrows():
            pid, visit_name, variable, value = row['participant_id'], row['visit_name'], row['variable'], row['value']
            v_id = visit_id_map[visit_name]
            f_id = feature_id_map[variable]
            patient_arrays[pid][v_id, f_id] = value

        # compute feature median
        patient_null_column = []
        feature_median = {}  # key: feature, value: median of the feature
        patient_feature_median = {}  # key: patient, value: { feature : median of the feature of patient }
        for pid in patient_length:
            patient_feature_median[pid] = {}  # initialize
        for var in feature_list:
            # feature median
            temp_data = data_df[data_df['variable'] == var]
            feature_median[var] = np.nanmedian(list(temp_data['value'].astype('float').values))
            # patient median
            for p in patient_feature_median:
                patient_temp_data = temp_data[temp_data['participant_id'] == p]
                tmp_values = list(patient_temp_data['value'].astype('float').values)
                if (len(tmp_values) == 0) or (len(tmp_values) == sum(
                        np.isnan(tmp_values))):  # patient do not have information of this feature
                    patient_feature_median[p][var] = feature_median[var]
                    patient_null_column.append(p)
                else:
                    patient_feature_median[p][var] = round(np.nanmedian(tmp_values))

        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path + "/sequence_data.pkl", "wb") as wf:
                pkl.dump(patient_arrays, wf)
            with open(save_path + "/feature_median.json", 'w') as wf:
                json.dump(feature_median, wf, indent=4)
            with open(save_path + "/patient_feature_median.json", 'w') as wf:
                json.dump(patient_feature_median, wf, indent=4)

        return (patient_arrays, feature_median, patient_feature_median)

    # Impute the missing data
    def interpolate_imputation(self, sequence_data, feature_median, feature_list, save_path=None):
        """
        Imputation based on pandas' interpolate method.

        :param sequence_data: key: participant_id, value: data matrix
        :param feature_median: median of each feature
        :param feature_list: list of features

        :return: imputed_data
        """

        M = len(feature_list)

        imputed_data = {}

        for p in sequence_data:
            data = pd.DataFrame(sequence_data[p])

            # address the issue that all column are nan
            for m in range(M):
                if data[m].isnull().all():
                    data[m] = feature_median[feature_list[m]]

            data = data.interpolate(method='linear', axis=0, limit_direction='both')

            imputed_data[p] = data.values

        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path + "/sequence_data_interpolate_imputation.pkl", "wb") as wf:
                pkl.dump(imputed_data, wf)

        return imputed_data

    def LOCF_FOCB_imputation(self, sequence_data, patient_feature_median, feature_list, save_path=None):
        """
        LOFC: last occurrence carry forward strategy
        FOCB: first occurrence carry backward strategy

        :param sequence_data: key: participant_id, value: data matrix
        :param patient_feature_median: median of each feature of each patient
        :param feature_list: list of features

        :return: imputed_data
        """

        imputed_data = {}

        for p in sequence_data:
            data = sequence_data[p]
            L, N = data.shape

            # build mask matrix (0: has missing value in the location, 1: other)
            data = (pd.DataFrame(data)).fillna(-1).values  # fill NaN as -1
            mask_idx = np.where(data == -1)  # first row: x-axis, second row: y-axis
            mask_matrix = np.ones((L, N), dtype='int')
            mask_matrix[mask_idx] = 0

            missing_num = len(mask_idx[0])
            for i in range(missing_num):
                row_idx = mask_idx[0][i]
                col_idx = mask_idx[1][i]

                if L == 1:  # only one visit
                    data[row_idx, col_idx] = patient_feature_median[p][feature_list[col_idx]]

                else:  # multiple visit
                    if row_idx == 0:  # first visit is NaN

                        if int(data[row_idx + 1, col_idx]) != -1:  # using FOCB
                            data[row_idx, col_idx] = data[row_idx + 1, col_idx]
                        else:  # using median of feature of patient
                            data[row_idx, col_idx] = patient_feature_median[p][feature_list[col_idx]]

                    elif row_idx == L - 1:  # last visit
                        if int(data[row_idx - 1, col_idx]) != -1:  # using LOCF
                            data[row_idx, col_idx] = data[row_idx - 1, col_idx]
                        else:  # using median of feature of patient
                            data[row_idx, col_idx] = patient_feature_median[p][feature_list[col_idx]]

                    else:
                        if int(data[row_idx - 1, col_idx]) != -1:  # using LOCF
                            data[row_idx, col_idx] = data[row_idx - 1, col_idx]
                            continue
                        if int(data[row_idx + 1, col_idx]) != -1:  # using FOCB
                            data[row_idx, col_idx] = data[row_idx + 1, col_idx]
                            continue
                        data[row_idx, col_idx] = patient_feature_median[p][feature_list[col_idx]]

            imputed_data[p] = data

        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path + "/sequence_data_LOCF_FOCB_imputation.pkl", "wb") as wf:
                pkl.dump(imputed_data, wf)

        return imputed_data

    # Data normalization
    def Z_score_normalization(self, sequence_data, feature_list, save_path=None):
        """
        (value - mean) / standard deviation

        :param sequence_data: key: participant_id, value: data matrix

        :return: normalized_sequence_data
        """

        M = len(feature_list)

        patients = sequence_data.keys()

        normalized_sequence_data = copy.deepcopy(sequence_data)

        for m in range(M):
            values = np.array([])  # initialize the vector of m-th feature
            for p in patients:
                values = np.concatenate((values, sequence_data[p][:, m]))
            # take the mean()
            feature_mean = np.mean(values)
            # take standard deviation
            feature_std = np.std(values)
            if feature_std == 0:
                print(feature_list[m])
            # update the data with Z score
            for p in patients:
                # Zscore = stats.zscore(sequence_data[p][:, m])
                Zscore = (sequence_data[p][:, m] - feature_mean) / feature_std
                normalized_sequence_data[p][:, m] = Zscore

        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path + "/sequence_data_Zscore.pkl", "wb") as wf:
                pkl.dump(normalized_sequence_data, wf)

        return normalized_sequence_data

    def minmax_normalization(self, sequence_data, feature_list, save_path=None):
        """
            (value - min) / (max - min)

            :param sequence_data: key: participant_id, value: data matrix

            :return: (normalized) sequence_data
            """

        M = len(feature_list)

        patients = sequence_data.keys()

        normalized_sequence_data = copy.deepcopy(sequence_data)

        for m in range(M):
            values = np.array([])  # initialize the vector of m-th feature
            for p in patients:
                values = np.concatenate((values, sequence_data[p][:, m]))
            # take the min()
            feature_min = values.min()
            # take the max()
            feature_max = values.max()

            # update the data with min-max normalization value
            for p in patients:
                norm_value = (sequence_data[p][:, m] - feature_min) / (feature_max - feature_min)
                normalized_sequence_data[p][:, m] = norm_value

        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path + "/sequence_data_minmax.pkl", "wb") as wf:
                pkl.dump(normalized_sequence_data, wf)

        return normalized_sequence_data

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]


def main():
    table_list = ["demographics", "history", "updrs1", "updrs2", "updrs3", "updrs4", "schwab", "moca", "upsit", "ess",
                  "rbd", "med",
                  # "bio"
                  ]
    cohort_name = 'PDBP'
    DP = DataPreparation(cohort_name, table_list)

    path = "[your directory]/validation/processed_data"  # Files save path
    data, feature_list = DP.longitudinal_data(save_path=path)
    static_data = DP.static_data(save_path=path)
    visit_list = DP.visit_list(save_path=path)
    visit_info = DP.visit_info(data, visit_list, save_path=path)

    patient_arrays, feature_median, patient_feature_median = DP.concatenate(data, visit_info, visit_list, feature_list, save_path=path)

    imputed_data = DP.LOCF_FOCB_imputation(patient_arrays, patient_feature_median, feature_list, save_path=path)
    Z_score_normalized_data = DP.Z_score_normalization(imputed_data, feature_list, save_path=path)


if __name__ == '__main__':
    main()
