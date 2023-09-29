"""

To extract all useful data into a unique csv file.

"""

import pandas as pd
import numpy as np
import os

# motor
def extract_updrs1(version):
    col_updrs1 = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "NP1COG", "NP1HALL", "NP1DPRS", "NP1ANXS", "NP1APAT", "NP1DDS"]
    updrs1 = pd.read_csv("../"+version+"/_Motor_Assessments/ALL_Motor_MDS-UPDRS/MDS_UPDRS_Part_I.csv", index_col=["PATNO", "EVENT_ID"], usecols=col_updrs1)
    return updrs1

def extract_updrs1pq(version):
    col_updrs1pq = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "NP1SLPN", "NP1SLPD", "NP1PAIN", "NP1URIN", "NP1CNST", "NP1LTHD", "NP1FATG"]
    updrs1pq = pd.read_csv("../"+version+"/_Motor_Assessments/ALL_Motor_MDS-UPDRS/MDS_UPDRS_Part_I__Patient_Questionnaire.csv", index_col=["PATNO", "EVENT_ID"], usecols=col_updrs1pq)
    return updrs1pq.reset_index()

def extract_updrs2pq(version):
    col_updrs2pq = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "NP2SPCH", "NP2SALV", "NP2SWAL", "NP2EAT", "NP2DRES", "NP2HYGN", "NP2HWRT", "NP2HOBB", "NP2TURN", "NP2TRMR", "NP2RISE", "NP2WALK", "NP2FREZ"]
    updrs2pq = pd.read_csv("../"+version+"/_Motor_Assessments/ALL_Motor_MDS-UPDRS/MDS_UPDRS_Part_II__Patient_Questionnaire-2.csv", index_col=["PATNO", "EVENT_ID"], usecols=col_updrs2pq)
    return updrs2pq.reset_index()

def extract_updrs3(version):
    col_updrs3_temp = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "PAG_NAME", "CMEDTM", "EXAMTM", "NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU", "NP3RIGLU", "PN3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL", "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL", "NP3LGAGR", "NP3LGAGL", "NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR", "NP3BRADY", "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON", "DYSKPRES", "DYSKIRAT", "NHY", "ANNUAL_TIME_BTW_DOSE_NUPDRS", "ON_OFF_DOSE", "PD_MED_USE"]
    col_updrs3 = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU", "NP3RIGLU", "PN3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL", "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL", "NP3LGAGR", "NP3LGAGL", "NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR", "NP3BRADY", "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON"]
    updrs3_temp = pd.read_csv("../"+version+"/_Motor_Assessments/ALL_Motor_MDS-UPDRS/MDS_UPDRS_Part_III.csv",
                              index_col=["PATNO", "EVENT_ID"], usecols=col_updrs3_temp)
    updrs3 = updrs3_temp[updrs3_temp.PAG_NAME == 'NUPDRS3']  # before dose
    updrs3a = updrs3_temp[updrs3_temp.PAG_NAME == 'NUPDRS3A']  # after dose

    updrs3 = updrs3.reset_index().drop(['PAG_NAME', 'CMEDTM', 'EXAMTM', 'PD_MED_USE', 'ON_OFF_DOSE', 'ANNUAL_TIME_BTW_DOSE_NUPDRS'], axis=1)
    updrs3a = updrs3a.reset_index().drop(['PAG_NAME', 'CMEDTM', 'EXAMTM', 'PD_MED_USE', 'ON_OFF_DOSE', 'ANNUAL_TIME_BTW_DOSE_NUPDRS'], axis=1)
    return updrs3, updrs3a

def extract_updrs4(version):
    col_updrs4 = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "NP4WDYSK", "NP4DYSKI", "NP4OFF", "NP4FLCTI", "NP4FLCTX", "NP4DYSTN"]
    updrs4 = pd.read_csv("../"+version+"/_Motor_Assessments/ALL_Motor_MDS-UPDRS/MDS_UPDRS_Part_IV.csv",
                         index_col=["PATNO", "EVENT_ID"], usecols=col_updrs4)
    return updrs4.reset_index()

def extract_schwab(version):
    col_schwab = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "MSEADLG"]
    schwab = pd.read_csv("../"+version+"/_Motor_Assessments/ALL_Motor_MDS-UPDRS/Modified_Schwab_+_England_ADL.csv",
                         index_col=["PATNO", "EVENT_ID"], usecols=col_schwab)
    return schwab.reset_index()

def extract_pase_house(version):
    col_pase_house = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "LTHSWRK", "HVYHSWRK", "HMREPR", "LAWNWRK", "OUTGARDN", "CAREGVR", "WRKVL", "WRKVLHR", "WRKVLACT"]
    pase_house = pd.read_csv("../"+version+"/_Motor_Assessments/ALL_Motor_MDS-UPDRS/PASE_-_Household_Activity.csv",
                             index_col=["PATNO", "EVENT_ID"], usecols=col_pase_house)
    return pase_house.reset_index()

# non-motor
def extract_aut(version):
    col_aut = [ "PATNO", "EVENT_ID", "ORIG_ENTRY", "SCAU1", "SCAU2", "SCAU3", "SCAU4", "SCAU5", "SCAU6", "SCAU7", "SCAU8", "SCAU9", "SCAU10", "SCAU11", "SCAU12", "SCAU13", "SCAU14", "SCAU15", "SCAU16", "SCAU17", "SCAU18", "SCAU19", "SCAU20", "SCAU21", "SCAU22", "SCAU23", "SCAU23A", "SCAU23AT", "SCAU24", "SCAU25", "SCAU26A", "SCAU26AT", "SCAU26B", "SCAU26BT", "SCAU26C", "SCAU26CT", "SCAU26D", "SCAU26DT" ]
    col_aut_gastrointestinal_up = [ "SCAU1", "SCAU2", "SCAU3" ]
    col_aut_gastrointestinal_down = [ "SCAU4", "SCAU5", "SCAU6", "SCAU7" ]
    col_aut_urinary = [ "SCAU8", "SCAU9", "SCAU10", "SCAU11", "SCAU12", "SCAU13" ]
    col_aut_cardiovascular = [  "SCAU14", "SCAU15", "SCAU16" ]
    col_aut_thermoregulatory = [ "SCAU17", "SCAU18" ]
    col_aut_pupillomotor = [ "SCAU19" ]
    col_aut_skin = [ "SCAU20", "SCAU21" ]
    col_aut_sexual = [ "SCAU22", "SCAU23", "SCAU24", "SCAU25"] # 9 for NA, might skew the results signific for M/F better to remove

    aut = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Autonomic_Tests/SCOPA-AUT.csv",
                      index_col=["PATNO", "EVENT_ID"], usecols=col_aut)
    aut["gastrointestinal_up"] = aut[col_aut_gastrointestinal_up].sum(axis=1)
    aut["gastrointestinal_down"] = aut[col_aut_gastrointestinal_down].sum(axis=1)
    aut["urinary"] = aut[col_aut_urinary].sum(axis=1)
    aut["cardiovascular"] = aut[col_aut_cardiovascular].sum(axis=1)
    aut["thermoregulatory"] = aut[col_aut_thermoregulatory].sum(axis=1)
    aut["pupillomotor"] = aut[col_aut_pupillomotor].sum(axis=1)
    aut["skin"] = aut[col_aut_skin].sum(axis=1)
    aut["sexual"] = aut[col_aut_sexual].sum(axis=1) # NA is assigned as 9, throwing things off, in case adding it, edit the next line too
    aut = aut[["ORIG_ENTRY", "gastrointestinal_up", "gastrointestinal_down", "urinary", "cardiovascular", "thermoregulatory", "pupillomotor", "skin", "sexual"]]
    return aut.reset_index()

def extract_cog_catg(version):
    col_cog_catg = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "COGDECLN", "FNCDTCOG", "COGSTATE"]
    cog_catg = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Cognition/Cognitive_Categorization.csv",
                           index_col=["PATNO", "EVENT_ID"], usecols=col_cog_catg)
    return cog_catg.reset_index()

def extract_geriatric(version):
    col_geriatric = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "GDSSATIS", "GDSDROPD", "GDSEMPTY", "GDSBORED", "GDSGSPIR",
                         "GDSAFRAD", "GDSHAPPY", "GDSHLPLS", "GDSHOME", "GDSMEMRY", "GDSALIVE", "GDSWRTLS", "GDSENRGY",
                         "GDSHOPLS", "GDSBETER"]
    col_geriatric_pos = ["GDSDROPD", "GDSEMPTY", "GDSBORED", "GDSAFRAD", "GDSHLPLS", "GDSHOME", "GDSMEMRY",
                             "GDSWRTLS", "GDSHOPLS", "GDSBETER"]
    col_geriatric_neg = ["GDSSATIS", "GDSGSPIR", "GDSHAPPY", "GDSALIVE", "GDSENRGY"]

    geriatric = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Neurobehavioral_Tests/Geriatric_Depression_Scale__Short_.csv",
                            index_col=["PATNO", "EVENT_ID"], usecols=col_geriatric)
    geriatric["total_pos"] = geriatric[col_geriatric_pos].sum(axis=1)
    geriatric["total_neg"] = geriatric[col_geriatric_neg].sum(axis=1)
    geriatric["geriatric_total"] = geriatric["total_pos"] + 5 - geriatric["total_neg"]
    geriatric = geriatric[["ORIG_ENTRY", "total_pos", "total_neg", "geriatric_total"]]  # drop the rest
    return geriatric.reset_index()

def extract_quip(version):
    col_quip = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "TMGAMBLE", "CNTRLGMB", "TMSEX", "CNTRLSEX", "TMBUY", "CNTRLBUY",
                    "TMEAT", "CNTRLEAT", "TMTORACT", "TMTMTACT", "TMTRWD"]
    quip = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Neurobehavioral_Tests/QUIP_Current_Short.csv",
                       index_col=["PATNO", "EVENT_ID"], usecols=col_quip)
    return quip.reset_index()

def extract_stai(version):
    col_stai = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "STAIAD1", "STAIAD2", "STAIAD3", "STAIAD4", "STAIAD5", "STAIAD6",
                    "STAIAD7", "STAIAD8", "STAIAD9", "STAIAD10", "STAIAD11", "STAIAD12", "STAIAD13", "STAIAD14",
                    "STAIAD15", "STAIAD16", "STAIAD17", "STAIAD18", "STAIAD19", "STAIAD20", "STAIAD21", "STAIAD22",
                    "STAIAD23", "STAIAD24", "STAIAD25", "STAIAD26", "STAIAD27", "STAIAD28", "STAIAD29", "STAIAD30",
                    "STAIAD31", "STAIAD32", "STAIAD33", "STAIAD34", "STAIAD35", "STAIAD36", "STAIAD37", "STAIAD38",
                    "STAIAD39", "STAIAD40"]
    col_stai_a_state_pos = ["STAIAD3", "STAIAD4", "STAIAD6", "STAIAD7", "STAIAD9", "STAIAD12", "STAIAD13",
                                "STAIAD14", "STAIAD17", "STAIAD18"]
    col_stai_a_state_neg = ["STAIAD1", "STAIAD2", "STAIAD5", "STAIAD8", "STAIAD10", "STAIAD11", "STAIAD15",
                                "STAIAD16", "STAIAD19", "STAIAD20"]
    col_stai_a_trait_pos = ["STAIAD22", "STAIAD24", "STAIAD25", "STAIAD28", "STAIAD29", "STAIAD31", "STAIAD32",
                                "STAIAD35", "STAIAD37", "STAIAD38", "STAIAD40"]
    col_stai_a_trait_neg = ["STAIAD21", "STAIAD23", "STAIAD26", "STAIAD27", "STAIAD30", "STAIAD33", "STAIAD34",
                                "STAIAD36", "STAIAD39"]
    stai = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Neurobehavioral_Tests/State-Trait_Anxiety_Inventory.csv",
                        index_col=["PATNO", "EVENT_ID"], usecols=col_stai)
    stai["a_state"] = stai[col_stai_a_state_pos].sum(axis=1) + (
                5 * len(col_stai_a_state_neg) - stai[col_stai_a_state_neg].sum(axis=1))
    stai["a_trait"] = stai[col_stai_a_trait_pos].sum(axis=1) + (
                5 * len(col_stai_a_trait_neg) - stai[col_stai_a_trait_neg].sum(axis=1))
    stai = stai[["ORIG_ENTRY", "a_state", "a_trait"]]
    return stai.reset_index()

def extract_benton(version):
    col_benton = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "JLO_TOTRAW", "DVS_JLO_MSSA", "DVS_JLO_MSSAE"] # "AGE_ASSESS_JLO",
    benton = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Neuropsychological_Tests/Benton_Judgment_of_Line_Orientation.csv",
        index_col=["PATNO", "EVENT_ID"], usecols=col_benton)
    return benton.reset_index().drop_duplicates(['PATNO', 'EVENT_ID'], keep='first')

def extract_hopkins_verbal(version):
    col_hopkins_verbal = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "HVLTRT1", "HVLTRT2", "HVLTRT3", "HVLTRDLY", "HVLTREC", "HVLTFPRL", "HVLTFPUN"]
    hopkins_verbal = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Neuropsychological_Tests/Hopkins_Verbal_Learning_Test.csv",
                                index_col=["PATNO", "EVENT_ID"], usecols=col_hopkins_verbal)
    return hopkins_verbal.reset_index()

def extract_letter_seq(version):
    col_letter_seq = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "LNS1A", "LNS1B", "LNS1C", "LNS2A","LNS2B", "LNS2C", "LNS3A", "LNS3B", "LNS3C", "LNS4A", "LNS4B", "LNS4C", "LNS5A", "LNS5B", "LNS5C", "LNS6A", "LNS6B", "LNS6C", "LNS7A", "LNS7B", "LNS7C", "LNS_TOTRAW"]
    col_letter_seq_details = ["LNS1A", "LNS1B", "LNS1C", "LNS2A","LNS2B", "LNS2C", "LNS3A", "LNS3B", "LNS3C", "LNS4A", "LNS4B", "LNS4C", "LNS5A", "LNS5B", "LNS5C", "LNS6A", "LNS6B", "LNS6C", "LNS7A", "LNS7B", "LNS7C"]

    letter_seq = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Neuropsychological_Tests/Letter_-_Number_Sequencing__PD_.csv",
        index_col=["PATNO", "EVENT_ID"], usecols=col_letter_seq)
    letter_seq["total"] = letter_seq[col_letter_seq_details].sum(axis=1)
    letter_seq = letter_seq[["ORIG_ENTRY", "total"]]  # letter_seq[["total"]] or letter_seq[["LNS_TOTRAW"]]
    return letter_seq.reset_index()

def extract_moca(version):
    col_moca = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "MCAALTTM", "MCACUBE", "MCACLCKC", "MCACLCKN", "MCACLCKH", "MCALION", "MCARHINO", "MCACAMEL", "MCAFDS", "MCABDS", "MCAVIGIL", "MCASER7", "MCASNTNC", "MCAVFNUM", "MCAVF", "MCAABSTR", "MCAREC1", "MCAREC2", "MCAREC3", "MCAREC4", "MCAREC5", "MCADATE", "MCAMONTH", "MCAYR", "MCADAY", "MCAPLACE", "MCACITY", "MCATOT"]
    col_moca_visuospatial = [ "MCAALTTM", "MCACUBE", "MCACLCKC", "MCACLCKN", "MCACLCKH"]
    col_moca_naming = [ "MCALION", "MCARHINO", "MCACAMEL"]
    col_moca_attention = [ "MCAFDS", "MCABDS", "MCAVIGIL", "MCASER7"]
    col_moca_language = [ "MCASNTNC", "MCAVF"]
    col_moca_delayed_recall = [ "MCAREC1", "MCAREC2", "MCAREC3", "MCAREC4", "MCAREC5"]
    col_moca_orientation = [ "MCADATE", "MCAMONTH", "MCAYR", "MCADAY", "MCAPLACE", "MCACITY"]

    moca = pd.read_csv(
        "../"+version+"/_Non-motor_Assessments/ALL_Neuropsychological_Tests/Montreal_Cognitive_Assessment__MoCA_.csv",
        index_col=["PATNO", "EVENT_ID"], usecols=col_moca)
    moca["visuospatial"] = moca[col_moca_visuospatial].sum(axis=1)
    moca["naming"] = moca[col_moca_naming].sum(axis=1)
    moca["attention"] = moca[col_moca_attention].sum(axis=1)
    moca["language"] = moca[col_moca_language].sum(axis=1)
    moca["delayed_recall"] = moca[col_moca_delayed_recall].sum(axis=1)
    moca = moca[["ORIG_ENTRY", "visuospatial", "naming", "attention", "language", "delayed_recall", "MCAABSTR", "MCAVFNUM", "MCATOT"]]  # drop extra
    return moca.reset_index()

def extract_semantic(version):
    col_semantic = [ "PATNO", "EVENT_ID", "ORIG_ENTRY", "VLTANIM", "VLTVEG", "VLTFRUIT" ]
    semantic = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Neuropsychological_Tests/Semantic_Fluency.csv",
                           index_col=["PATNO", "EVENT_ID"], usecols=col_semantic)
    return semantic.reset_index()

def extract_sdm(version):
    col_sdm = [ "PATNO", "EVENT_ID", "ORIG_ENTRY", "ORIG_ENTRY", "SDMTOTAL", "DVT_SDM"]
    sdm = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Neuropsychological_Tests/Symbol_Digit_Modalities.csv",
                      index_col=["PATNO", "EVENT_ID"], usecols=col_sdm)
    return sdm.reset_index()

def extract_upsit(version):
    col_upsit = [ "SUBJECT_ID", "SCENT_10_RESPONSE", "SCENT_09_RESPONSE", "SCENT_08_RESPONSE", "SCENT_07_RESPONSE", "SCENT_06_RESPONSE", "SCENT_05_RESPONSE", "SCENT_04_RESPONSE", "SCENT_03_RESPONSE", "SCENT_02_RESPONSE", "SCENT_01_RESPONSE", "SCENT_20_RESPONSE", "SCENT_19_RESPONSE", "SCENT_18_RESPONSE", "SCENT_17_RESPONSE", "SCENT_16_RESPONSE", "SCENT_15_RESPONSE", "SCENT_14_RESPONSE", "SCENT_13_RESPONSE", "SCENT_12_RESPONSE", "SCENT_11_RESPONSE", "SCENT_30_RESPONSE", "SCENT_29_RESPONSE", "SCENT_28_RESPONSE", "SCENT_27_RESPONSE", "SCENT_26_RESPONSE", "SCENT_25_RESPONSE", "SCENT_24_RESPONSE", "SCENT_23_RESPONSE", "SCENT_22_RESPONSE", "SCENT_21_RESPONSE", "SCENT_40_RESPONSE", "SCENT_39_RESPONSE", "SCENT_38_RESPONSE", "SCENT_37_RESPONSE", "SCENT_36_RESPONSE", "SCENT_35_RESPONSE", "SCENT_34_RESPONSE", "SCENT_33_RESPONSE", "SCENT_32_RESPONSE", "SCENT_31_RESPONSE", "SCENT_10_CORRECT", "SCENT_09_CORRECT", "SCENT_08_CORRECT", "SCENT_07_CORRECT", "SCENT_06_CORRECT", "SCENT_05_CORRECT", "SCENT_04_CORRECT", "SCENT_03_CORRECT", "SCENT_02_CORRECT", "SCENT_01_CORRECT", "SCENT_20_CORRECT", "SCENT_19_CORRECT", "SCENT_18_CORRECT", "SCENT_17_CORRECT", "SCENT_16_CORRECT", "SCENT_15_CORRECT", "SCENT_14_CORRECT", "SCENT_13_CORRECT", "SCENT_12_CORRECT", "SCENT_11_CORRECT", "SCENT_30_CORRECT", "SCENT_29_CORRECT", "SCENT_28_CORRECT", "SCENT_27_CORRECT", "SCENT_26_CORRECT", "SCENT_25_CORRECT", "SCENT_24_CORRECT", "SCENT_23_CORRECT", "SCENT_22_CORRECT", "SCENT_21_CORRECT", "SCENT_40_CORRECT", "SCENT_39_CORRECT", "SCENT_38_CORRECT", "SCENT_37_CORRECT", "SCENT_36_CORRECT", "SCENT_35_CORRECT", "SCENT_34_CORRECT", "SCENT_33_CORRECT", "SCENT_32_CORRECT", "SCENT_31_CORRECT", "TOTAL_CORRECT"]
    upsit = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Olfactory_Tests/Olfactory_UPSIT.csv", index_col=["SUBJECT_ID"], usecols=col_upsit)
    return upsit.reset_index()

def extract_upsit_booklet(version):
    col_upsit_booklet = [ "PATNO", "EVENT_ID", "ORIG_ENTRY", "UPSITBK1", "UPSITBK2", "UPSITBK3", "UPSITBK4" ]
    upsit_booklet = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Olfactory_Tests/University_of_Pennsylvania_Smell_ID_Test.csv",
        index_col=["PATNO", "EVENT_ID"], usecols=col_upsit_booklet)
    return upsit_booklet.reset_index()

def extract_epworth(version):
    col_epworth = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "ESS1", "ESS2", "ESS3", "ESS4", "ESS5", "ESS6", "ESS7", "ESS8"]
    epworth = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Sleep_Disorder_Tests/Epworth_Sleepiness_Scale.csv",
        index_col=["PATNO", "EVENT_ID"], usecols=col_epworth)
    return epworth.reset_index()

def extract_rem(version):
    col_rem = [ "PATNO", "EVENT_ID", "ORIG_ENTRY", "DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV", "SLPINJUR", "DRMVERBL", "DRMFIGHT", "DRMUMV", "DRMOBJFL", "MVAWAKEN", "DRMREMEM", "SLPDSTRB", "STROKE", "HETRA", "PARKISM", "RLS", "NARCLPSY", "DEPRS", "EPILEPSY", "BRNINFM", "CNSOTH" ]
    rem = pd.read_csv("../"+version+"/_Non-motor_Assessments/ALL_Sleep_Disorder_Tests/REM_Sleep_Disorder_Questionnaire.csv",
        index_col=["PATNO", "EVENT_ID"], usecols=col_rem)
    return rem.reset_index()

# biospecimen
def extract_biospecimen(version):
    col_biospecimen = ["PATNO", "CLINICAL_EVENT", "RUNDATE", "TYPE", "TESTNAME", "TESTVALUE", "UNITS"]

    biospecimen = pd.read_csv("../"+version+"/_Biospecimen/ALL_Biospecimen_Sample_Analysis/Current_Biospecimen_Analysis_Results.csv",
                              index_col=["PATNO"], usecols=col_biospecimen, dtype={'UNITS': str})
    biospecimen.rename(columns={'CLINICAL_EVENT':'EVENT_ID'}, inplace=True)
    csf = biospecimen[(biospecimen["TYPE"] == 'Cerebrospinal Fluid') & ~(biospecimen["TESTVALUE"] == "below detection limit")][["EVENT_ID", "RUNDATE", "TESTNAME", "TESTVALUE"]]
    hemoglobin = csf[csf["TESTNAME"] == "CSF Hemoglobin"].reset_index().drop_duplicates(["PATNO","EVENT_ID","TESTNAME"])
    alpha_syn = csf[csf["TESTNAME"] == "CSF Alpha-synuclein"].reset_index().drop_duplicates(["PATNO","EVENT_ID","TESTNAME"])
    total_tau = csf[csf["TESTNAME"] == "tTau"].reset_index().drop_duplicates(["PATNO","EVENT_ID","TESTNAME"])
    abeta_42 = csf[csf["TESTNAME"] == "ABeta 1-42"].reset_index().drop_duplicates(["PATNO","EVENT_ID","TESTNAME"])
    p_tau181p = csf[csf["TESTNAME"] == "pTau"].reset_index().drop_duplicates(["PATNO","EVENT_ID","TESTNAME"])
    dna = biospecimen[(biospecimen["TYPE"] == 'DNA')][["EVENT_ID", "RUNDATE", "TESTNAME", "TESTVALUE"]]
    rna = biospecimen[(biospecimen["TYPE"] == 'RNA')][["EVENT_ID", "RUNDATE", "TESTNAME", "TESTVALUE"]]
    plasma = biospecimen[(biospecimen["TYPE"] == 'Plasma')][["EVENT_ID", "RUNDATE", "TESTNAME", "TESTVALUE"]]
    serum = biospecimen[(biospecimen["TYPE"] == 'Serum')][["EVENT_ID", "RUNDATE", "TESTNAME", "TESTVALUE"]]

    return hemoglobin, alpha_syn, total_tau, abeta_42, p_tau181p, dna, rna, plasma, serum

# medical
def extract_pd_start(version):
   col_pd_features = ["PATNO", "SXMO","SXYEAR", "PDDXDT"] # first symptom onset month, year, diagnosis date
   # first symptom onset month, year, diagnosis date
   pd_start = pd.read_csv("../"+version+"/_Medical_History/ALL_Medical/PD_Features.csv", index_col=["PATNO"],
                          usecols=col_pd_features)
   return pd_start.reset_index().fillna(value={'SXMO': 0})

def extract_medication(version):
    col_pd_medication = ["PATNO", "EVENT_ID", "INFODT", "PDMEDYN", "ONLDOPA", "ONDOPAG",
                             "ONOTHER"]  # on medication, Levodopa, Dopamine Agonist, other
    pd_medication = pd.read_csv("../"+version+"/_Medical_History/ALL_Medical/Use_of_PD_Medication.csv",
                                index_col=["PATNO", "EVENT_ID"], usecols=col_pd_medication)
    pd_medication.rename(columns={'INFODT': 'ORIG_ENTRY'}, inplace=True)
    return pd_medication.reset_index()

def extract_vital_sign(version):
    col_vital_sign = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "WGTKG","HTCM"]
    vital_signs = pd.read_csv("../"+version+"/_Medical_History/ALL_Medical/Vital_Signs.csv",
                              index_col=["PATNO", "EVENT_ID"], usecols=col_vital_sign)
    return vital_signs.reset_index()

# Medical-Neurological Exam
def extract_neuro_cranial(version):
    col_neuro_cranial = ["PATNO", "EVENT_ID", "ORIG_ENTRY", "CN1RSP", "CN2RSP", "CN346RSP", "CN5RSP", "CN7RSP", "CN8RSP", "CN910RSP", "CN11RSP", "CN12RSP"]
    neuro_cranial = pd.read_csv("../"+version+"/_Medical_History/ALL_Neurological_Exam/Neurological_Exam_-_Cranial_Nerves.csv",
        index_col=["PATNO", "EVENT_ID"], usecols=col_neuro_cranial)
    return neuro_cranial.reset_index()

# Subject Enrollment
def extract_primary_diag(version):
    col_primary_diag = ["PATNO", "PRIMDIAG"]
    primary_diag = pd.read_csv("../"+version+"/_Enrollment/ALL_Subject_Enrollment/Primary_Diagnosis.csv",
                               index_col=["PATNO"], usecols=col_primary_diag)
    return primary_diag.reset_index()

# Subject Characteristics
def extract_family_history(version):
    col_family_history = ["PATNO", "BIOMOM", "BIOMOMPD", "BIODAD", "BIODADPD", "FULSIB", "FULSIBPD", "HAFSIB",
                              "HAFSIBPD", "MAGPAR", "MAGPARPD", "PAGPAR", "PAGPARPD", "MATAU", "MATAUPD", "PATAU",
                              "PATAUPD", "KIDSNUM", "KIDSPD"]
    family_history = pd.read_csv("../"+version+"/_Subject_Characteristics/ALL_Family_History/Family_History__PD_.csv", index_col=["PATNO"],
                            usecols=col_family_history)
    return family_history

def extract_status(version):
    col_status = ["PATNO", "RECRUITMENT_CAT", "IMAGING_CAT", "ENROLL_DATE", "ENROLL_CAT"]
    status = pd.read_csv("../"+version+"/_Subject_Characteristics/ALL_Patient_Status/Patient_Status.csv",
                         index_col=["PATNO"], usecols=col_status)
    return status

def extract_screening(version):
    col_screening = ["PATNO", "BIRTHDT", "GENDER", "APPRDX", "CURRENT_APPRDX", "HISPLAT", "RAINDALS", "RAASIAN", "RABLACK", "RAHAWOPI", "RAWHITE", "RANOS"]
    screening = pd.read_csv("../"+version+"/_Subject_Characteristics/ALL_Subject_Demographics/Screening___Demographics.csv",
                index_col=["PATNO"], usecols=col_screening)
    return screening

def extract_socio(version):
    col_socio = [ "PATNO", "EDUCYRS", "HANDED" ]
    socio = pd.read_csv("../"+version+"/_Subject_Characteristics/ALL_Subject_Demographics/Socio-Economics.csv",
                        index_col=["PATNO"], usecols=col_socio)
    return socio

# -------------extract data-----------------
datasets_type = {
    'updrs1': 'motor',
    'updrs1pq': 'motor',
    'updrs2pq': 'motor',
    'updrs3': 'motor',
    'updrs4': 'motor',
    'schwab': 'motor',
    'pase_house': 'motor',

    'aut': 'non-motor',
    'cog_catg': 'non-motor',
    'geriatric': 'non-motor',
    'quip': 'non-motor',
    'stai': 'non-motor',
    'benton': 'non-motor',
    'hopkins_verbal': 'non-motor',
    'letter_seq': 'non-motor',
    'moca': 'non-motor',
    'semantic': 'non-motor',
    'sdm': 'non-motor',
    'upsit': 'non-motor',
    'upsit_booklet': 'non-motor',
    'epworth': 'non-motor',
    'rem': 'non-motor',

    'hemoglobin': 'biospecimen',
    'alpha_syn': 'biospecimen',
    'total_tau': 'biospecimen',
    'abeta_42': 'biospecimen',
    'p_tau181p': 'biospecimen',
    'dna': 'biospecimen',
    'rna': 'biospecimen',
    'plasma': 'biospecimen',
    'serum': 'biospecimen',

    'pd_start': 'medical',
    'pd_medication': 'medical',
    'vital_signs': 'medical',
    'neuro_cranial': 'medical',

    'family_history': 'subject_characteristics',
    'status': 'subject_characteristics',
    'screening': 'subject_characteristics',
    'socio': 'subject_characteristics',
    'primary_diag': 'subject_characteristics'
}



def extract_data(version, output_dir):
    if os.path.exists("../"+output_dir) == False:
        os.mkdir("../"+output_dir)

    data_file = "../"+output_dir+"/data.csv"

    wf = open(data_file, "w")
    wf.write("PATNO,EVENT_ID,Time,Variable,Var_Type,Source,Value\n")

    # ---static features---
    # age of onset
    datatype = 'subject_characteristics'
    pd_start = extract_pd_start(version)
    for idx, row in pd_start.iterrows():
        PATNO, SXMO, SXYEAR, PDDXDT = row['PATNO'], row['SXMO'], row['SXYEAR'], row['PDDXDT']
        if not np.isnan(SXYEAR):
            if np.isnan(SXMO) or SXMO == 0:
                symptom_date = "%s/%s" % (1, SXYEAR)
            else:
                symptom_date = "%s/%s" % (int(SXMO), SXYEAR)
            wf.write('%s,%s,%s,%s,%s,%s,%s\n' % (
            PATNO, 'static', 'static', 'SymptomDate', datatype, 'pd_start', symptom_date))
        wf.write('%s,%s,%s,%s,%s,%s,%s\n' % (PATNO, 'static', 'static', 'PDDXDT', datatype, 'pd_start', PDDXDT))
    print("pd_start finished!")

    # other static features
    family_history = extract_family_history(version)
    status = extract_status(version)
    screening = extract_screening(version)
    socio = extract_socio(version)
    static_datasets = ['family_history', 'status', 'screening', 'socio']
    for dataset in static_datasets:
        datatype = datasets_type[dataset]
        dataset_df = eval(dataset).reset_index()
        variable_list = list(dataset_df.columns)[1:]
        for idx, row in dataset_df.iterrows():
            PATNO = row['PATNO']
            for var in variable_list:
                var_value = row[var]
                wf.write('%s,%s,%s,%s,%s,%s,%s\n' % (PATNO, 'static', 'static', var, datatype, dataset, var_value))
        print("%s finished!" % dataset)

    # ---longitudinal features---
    longitudinal_datasets = ['updrs1', 'updrs1pq', 'updrs2pq', 'updrs3', 'schwab',
                             'updrs4', 'pase_house',
                             'aut', 'cog_catg', 'geriatric', 'quip', 'stai', 'benton', 'hopkins_verbal', 'letter_seq',
                             'moca', 'semantic', 'sdm', 'upsit_booklet', 'epworth', 'rem',
                             'pd_medication', 'vital_signs', 'neuro_cranial']
    updrs1 = extract_updrs1(version)
    updrs1pq = extract_updrs1pq(version)
    updrs2pq = extract_updrs2pq(version)
    updrs3, _ = extract_updrs3(version)
    updrs4 = extract_updrs4(version)
    schwab = extract_schwab(version)
    pase_house = extract_pase_house(version)
    aut = extract_aut(version)
    cog_catg = extract_cog_catg(version)
    geriatric = extract_geriatric(version)
    quip = extract_quip(version)
    stai = extract_stai(version)
    benton = extract_benton(version)
    hopkins_verbal = extract_hopkins_verbal(version)
    letter_seq = extract_letter_seq(version)
    moca = extract_moca(version)
    semantic = extract_semantic(version)
    sdm = extract_sdm(version)
    upsit_booklet = extract_upsit_booklet(version)
    epworth = extract_epworth(version)
    rem = extract_rem(version)
    pd_medication = extract_medication(version)
    vital_signs = extract_vital_sign(version)
    neuro_cranial = extract_neuro_cranial(version)

    longitudinal_feature_vocabulary = {}
    longitudinal_feature_source = {}
    longitudinal_feature_list = []

    ## longitudinal clinical data
    for dataset in longitudinal_datasets:
        datatype = datasets_type[dataset]
        dataset_df = eval(dataset).reset_index()
        variable_list = [x for x in list(dataset_df.columns) if x not in ['PATNO', 'EVENT_ID', 'ORIG_ENTRY', 'index']]
        # print(dataset, variable_list)

        for var in variable_list:
            if var in longitudinal_feature_vocabulary:
                print("!!!! Error: feature got same name!")
                print(var, longitudinal_feature_source[var])
                print(dataset)
            else:
                longitudinal_feature_vocabulary[var] = datatype
                longitudinal_feature_source[var] = dataset
                longitudinal_feature_list.append(var)

        for idx, row in dataset_df.iterrows():
            PATNO, EVENT_ID, Time = row['PATNO'], row['EVENT_ID'], row['ORIG_ENTRY']
            for var in variable_list:
                var_value = row[var]
                wf.write('%s,%s,%s,%s,%s,%s,%s\n' % (PATNO, EVENT_ID, Time, var, datatype, dataset, var_value))
        print("%s finished!" % dataset)

    ## longitudinal biospecimen
    hemoglobin, alpha_syn, total_tau, abeta_42, p_tau181p, dna, rna, plasma, serum = extract_biospecimen(version)
    for dataset in ['alpha_syn', 'total_tau', 'abeta_42', 'p_tau181p']:
        datatype = datasets_type[dataset]
        dataset_df = eval(dataset).reset_index()

        var = dataset
        if var in longitudinal_feature_vocabulary:
            print("!!!! Error: feature got same name!")
            print(var, longitudinal_feature_source[var])
            print(dataset)
        else:
            longitudinal_feature_vocabulary[var] = datatype
            longitudinal_feature_source[var] = dataset
            longitudinal_feature_list.append(var)

        for idx, row in dataset_df.iterrows():
            PATNO, EVENT_ID, Time, var_value = row['PATNO'], row['EVENT_ID'], row['RUNDATE'], str(row['TESTVALUE'])
            if '<' in var_value:
                var_value = float(var_value[1:])
            else:
                var_value = float(var_value)
            Time = Time.split('-')
            y, m = Time[0], Time[1]
            Time = "%s/%s" % (m, y)

            wf.write('%s,%s,%s,%s,%s,%s,%s\n' % (PATNO, EVENT_ID, Time, var, datatype, dataset, var_value))
        print("%s finished!" % dataset)

    wf.close()

    datatype = {'PATNO': int}
    df = pd.read_csv(data_file, dtype=datatype)

    df = df.dropna()

    df.to_csv(data_file, index=False)

    with open("../"+output_dir+"/feature_dictionary.csv", "w") as wf:
        wf.write("Variable,Var_Type,Source\n")
        for var in longitudinal_feature_list:
            wf.write("%s,%s,%s\n" % (var, longitudinal_feature_vocabulary[var], longitudinal_feature_source[var]))

    # print(df)
    # print(df.isnull().sum())

    print("------ All Finished! -------")


