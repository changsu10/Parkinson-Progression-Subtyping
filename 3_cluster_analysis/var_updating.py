import pandas as pd
import numpy as np

# non-motor
def UPDRS1(dataframe, seq_data, feature_map, visit):
    target_vars = ['updrs1', 'hallucination', 'Apathy', 'Pain', 'Fatigue']
    updrs1_idxs = [feature_map[ft] for ft in ['NP1COG', 'NP1HALL', 'NP1DPRS', 'NP1ANXS', 'NP1APAT', 'NP1DDS', 'NP1SLPN', 'NP1SLPD', 'NP1PAIN', 'NP1URIN', 'NP1CNST', 'NP1LTHD', 'NP1FATG']]
    hal_idx, apa_idx, pain_idx, fat_idx = feature_map['NP1HALL'], feature_map['NP1APAT'], feature_map['NP1PAIN'], feature_map['NP1FATG']

    dataframe = dataframe.reindex(columns=list(dataframe.columns)+target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        updrs1_score = data_mtx[visit, updrs1_idxs].sum()
        hal, apa, pain, fat = data_mtx[visit, hal_idx], data_mtx[visit, apa_idx], data_mtx[visit, pain_idx], data_mtx[visit, fat_idx]
        dataframe.loc[idx, target_vars] = [updrs1_score, hal, apa, pain, fat]

    print("UPDRS1 domain finished!")
    return dataframe

def Epworth(dataframe, seq_data, feature_map, visit):
    target_vars = ['epworth']
    epworth_idxs = [feature_map[ft] for ft in
                   ['ESS1', 'ESS2', 'ESS3', 'ESS4', 'ESS5', 'ESS6', 'ESS7', 'ESS8']]

    dataframe = dataframe.reindex(columns=list(dataframe.columns)+target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        epworth_score = data_mtx[visit, epworth_idxs].sum()
        dataframe.loc[idx, target_vars] = [epworth_score]

    print("Epworth domain finished!")
    return dataframe

def GDS(dataframe, seq_data, feature_map, visit):
    target_vars = ['GDS']
    gds_idxs = [feature_map[ft] for ft in ['geriatric_total']]

    dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        gds_score = data_mtx[visit, gds_idxs].sum()
        dataframe.loc[idx, target_vars] = [gds_score]
    print("GDS domain finished!")
    return dataframe

def STAI(dataframe, seq_data, feature_map, visit):
    target_vars = ['stai_state', 'stai_trait']
    state_idxs = [feature_map[ft] for ft in ['a_state']]
    trait_idxs = [feature_map[ft] for ft in ['a_trait']]

    dataframe = dataframe.reindex(columns=list(dataframe.columns)+target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        stai_state = data_mtx[visit, state_idxs].sum()
        stai_trait = data_mtx[visit, trait_idxs].sum()

        dataframe.loc[idx, target_vars] = [stai_state, stai_trait]

    print("STAI domain finished!")
    return dataframe

def SCOPA_Aut(dataframe, seq_data, feature_map, visit):
    target_vars = ['aut_gastrointestinal_up', 'aut_gastrointestinal_down', 'aut_urinary',
                   'aut_cardiovascular', 'aut_thermoregulatory', 'aut_pupillomotor',
                   'aut_skin', 'aut_sexual']

    up_idxs = [feature_map[ft] for ft in ['gastrointestinal_up']]
    down_idxs = [feature_map[ft] for ft in ['gastrointestinal_down']]
    uri_idxs = [feature_map[ft] for ft in ['urinary']]
    card_idxs = [feature_map[ft] for ft in ['cardiovascular']]
    ther_idxs = [feature_map[ft] for ft in ['thermoregulatory']]
    pupi_idxs = [feature_map[ft] for ft in ['pupillomotor']]
    skin_idxs = [feature_map[ft] for ft in ['skin']]
    sex_idxs = [feature_map[ft] for ft in ['sexual']]

    dataframe = dataframe.reindex(columns=list(dataframe.columns)+target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        up = data_mtx[visit, up_idxs].sum()
        down = data_mtx[visit, down_idxs].sum()
        uri = data_mtx[visit, uri_idxs].sum()
        card = data_mtx[visit, card_idxs].sum()
        ther = data_mtx[visit, ther_idxs].sum()
        pupi = data_mtx[visit, pupi_idxs].sum()
        skin = data_mtx[visit, skin_idxs].sum()
        sex = data_mtx[visit, sex_idxs].sum()

        dataframe.loc[idx, target_vars] = [up, down, uri, card, ther, pupi, skin, sex]

    print("SCOPA_Aut domain finished!")
    return dataframe

def MOCA(dataframe, seq_data, static_df, feature_map, visit):
    target_vars = ['moca', 'moca_visuospatial', 'moca_naming', 'moca_attention',
                   'moca_language', 'moca_delayed_recall']

    moca_idxs = [feature_map[ft] for ft in ['MCATOT']]
    visu_idxs = [feature_map[ft] for ft in ['visuospatial']]
    nam_idxs = [feature_map[ft] for ft in ['naming']]
    att_idxs = [feature_map[ft] for ft in ['attention']]
    lang_idxs = [feature_map[ft] for ft in ['language']]
    dere_idxs = [feature_map[ft] for ft in ['delayed_recall']]

    dataframe = dataframe.reindex(columns=list(dataframe.columns)+target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        EDUCYRS = static_df.loc[PATNO, 'EDUCYRS'] # education years

        moca = data_mtx[visit, moca_idxs].sum()
        visu = data_mtx[visit, visu_idxs].sum()
        nam = data_mtx[visit, nam_idxs].sum()
        att = data_mtx[visit, att_idxs].sum()
        lang = data_mtx[visit, lang_idxs].sum()
        dere = data_mtx[visit, dere_idxs].sum()

        if EDUCYRS <= 12 and moca < 30:
            adj_moca = moca + 1
        else:
            adj_moca = moca

        dataframe.loc[idx, target_vars] = [adj_moca, visu, nam, att, lang, dere]

    print("MOCA domain finished!")
    return dataframe

def Benton(dataframe, seq_data, feature_map, visit):
    target_vars = ['benton']
    benton_idxs = [feature_map[ft] for ft in ['JLO_TOTRAW']]

    dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        benton = data_mtx[visit, benton_idxs].sum()
        dataframe.loc[idx, target_vars] = [benton]

    print("Benton domain finished!")
    return dataframe

def Hopkins(dataframe, seq_data, feature_map, visit):
    target_vars = ['HVLT_total_recall', 'HVLT_Discrimination_Recognition', 'HVLT_Retention', 'HVLT_Delayed_Recall']
    HVLTRT1_idxs = [feature_map[ft] for ft in ['HVLTRT1']]
    HVLTRT2_idxs = [feature_map[ft] for ft in ['HVLTRT2']]
    HVLTRT3_idxs = [feature_map[ft] for ft in ['HVLTRT3']]
    HVLTREC_idxs = [feature_map[ft] for ft in ['HVLTREC']]
    HVLTFPRL_idxs = [feature_map[ft] for ft in ['HVLTFPRL']]
    HVLTFPUN_idxs = [feature_map[ft] for ft in ['HVLTFPUN']]
    HVLTRDLY_idxs = [feature_map[ft] for ft in ['HVLTRDLY']]

    dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        HVLTRT1 = data_mtx[visit, HVLTRT1_idxs].sum()
        HVLTRT2 = data_mtx[visit, HVLTRT2_idxs].sum()
        HVLTRT3 = data_mtx[visit, HVLTRT3_idxs].sum()
        HVLTREC = data_mtx[visit, HVLTREC_idxs].sum()
        HVLTFPRL = data_mtx[visit, HVLTFPRL_idxs].sum()
        HVLTFPUN = data_mtx[visit, HVLTFPUN_idxs].sum()
        HVLTRDLY = data_mtx[visit, HVLTRDLY_idxs].sum()

        total = np.sum([HVLTRT1, HVLTRT2, HVLTRT3])
        discrimination = HVLTREC - (HVLTFPRL + HVLTFPUN)
        retention = HVLTRDLY / max([HVLTRT2, HVLTRT3])
        dere = HVLTRDLY

        dataframe.loc[idx, target_vars] = [total, discrimination, retention, dere]
    #     print(HVLTRT1, HVLTRT2, HVLTRT3, HVLTREC, HVLTFPRL, HVLTFPUN, HVLTRDLY)
    #     print(total, discrimination, retention, dere)
    # print(dataframe[target_vars])
    print("Hopkins domain finished!")
    return dataframe

def RBD(dataframe, seq_data, feature_map, visit):
    target_vars = ['RBD']

    term1_idxs = [feature_map[ft] for ft in ["DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV",
                                             "SLPINJUR", "DRMVERBL", "DRMFIGHT", "DRMUMV",
                                             "DRMOBJFL", "MVAWAKEN", "DRMREMEM", "SLPDSTRB"]]
    term2_idxs = [feature_map[ft] for ft in ["STROKE", "HETRA", "PARKISM", "RLS", "NARCLPSY",
                                             "DEPRS", "EPILEPSY", "BRNINFM", "CNSOTH"]]


    dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        term1 = data_mtx[visit, term1_idxs].sum()
        term2 = data_mtx[visit, term2_idxs].sum()

        rbd = term1
        if term2 >= 1:
            rbd += 1

        dataframe.loc[idx, target_vars] = [rbd]

    print("RBD domain finished!")
    return dataframe

def QUIP(dataframe, seq_data, feature_map, visit):
    target_vars = ['QUIP']

    secA_idxs = [feature_map[ft] for ft in ["CNTRLGMB", "TMGAMBLE"]]
    secB_idxs = [feature_map[ft] for ft in ["CNTRLSEX", "TMSEX"]]
    secC_idxs = [feature_map[ft] for ft in ["CNTRLBUY", "TMBUY"]]
    secD_idxs = [feature_map[ft] for ft in ["CNTRLEAT", "TMEAT"]]
    secE_idxs = [feature_map[ft] for ft in ["TMTORACT", "TMTMTACT", "TMTRWD"]]

    dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        secA = data_mtx[visit, secA_idxs].sum()
        secB = data_mtx[visit, secB_idxs].sum()
        secC = data_mtx[visit, secC_idxs].sum()
        secD = data_mtx[visit, secD_idxs].sum()
        secE = data_mtx[visit, secE_idxs].sum()

        quip = secE
        if secA >= 1: quip += 1
        if secB >= 1: quip += 1
        if secC >= 1: quip += 1
        if secD >= 1: quip += 1

        dataframe.loc[idx, target_vars] = [quip]

    print("QUIP domain finished!")
    return dataframe

def LNS(dataframe, seq_data, feature_map, visit):
    target_vars = ['LNS']
    lns_idxs = [feature_map[ft] for ft in ['total']]

    dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        lns_score = data_mtx[visit, lns_idxs].sum()
        dataframe.loc[idx, target_vars] = [lns_score]

    print("LNS domain finished!")
    return dataframe

def Semantic(dataframe, seq_data, feature_map, visit):
    target_vars = ['Semantic_Fluency']
    sf_idxs = [feature_map[ft] for ft in ['VLTANIM', 'VLTVEG', 'VLTFRUIT']]

    dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        sf_score = data_mtx[visit, sf_idxs].sum()
        dataframe.loc[idx, target_vars] = [sf_score]

    print("Semantic domain finished!")
    return dataframe

def SDM(dataframe, seq_data, feature_map, visit):
    target_vars = ['SDM']
    sdm_idxs = [feature_map[ft] for ft in ['DVT_SDM']]

    dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        sdm_score = data_mtx[visit, sdm_idxs].sum()
        dataframe.loc[idx, target_vars] = [sdm_score]

    print("SDM domain finished!")
    return dataframe

# motor
def UPDRS2(dataframe, seq_data, feature_map, visit):
    target_vars = ['updrs2']
    updrs2_idxs = [feature_map[ft] for ft in ['NP2SPCH', 'NP2SALV', 'NP2SWAL', 'NP2EAT',
                                              'NP2DRES', 'NP2HYGN', 'NP2HWRT', 'NP2HOBB',
                                              'NP2TURN', 'NP2TRMR', 'NP2RISE', 'NP2WALK', 'NP2FREZ']]

    dataframe = dataframe.reindex(columns=list(dataframe.columns)+target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        updrs2_score = data_mtx[visit, updrs2_idxs].sum()
        dataframe.loc[idx, target_vars] = [updrs2_score]

    print("UPDRS2 domain finished!")
    return dataframe

def UPDRS3(dataframe, seq_data, feature_map, visit):
    target_vars = ['updrs3', 'HY_stage']
    updrs3_idxs = [feature_map[ft] for ft in ["NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU", "NP3RIGLU",
                                              "PN3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR",
                                              "NP3HMOVL", "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL",
                                              "NP3LGAGR", "NP3LGAGL", "NP3RISNG", "NP3GAIT", "NP3FRZGT",
                                              "NP3PSTBL", "NP3POSTR", "NP3BRADY", "NP3PTRMR", "NP3PTRML",
                                              "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL",
                                              "NP3RTALL", "NP3RTALJ", "NP3RTCON"]]

    hy_idx = feature_map['NHY']

    dataframe = dataframe.reindex(columns=list(dataframe.columns)+target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        updrs3_score = data_mtx[visit, updrs3_idxs].sum()

        hy_stage = data_mtx[visit, hy_idx]

        dataframe.loc[idx, target_vars] = [updrs3_score, hy_stage]

    print("UPDRS3 domain finished!")
    return dataframe

def Schwab(dataframe, seq_data, feature_map, visit):
    target_vars = ['Schwab']
    schwab_idxs = [feature_map[ft] for ft in ['MSEADLG']]

    dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        schwab_score = data_mtx[visit, schwab_idxs].sum()
        dataframe.loc[idx, target_vars] = [schwab_score]

    print("Schwab domain finished!")
    return dataframe

def Tremor_PIGD(dataframe, seq_data, feature_map, visit):
    target_vars = ['Tremor_score', 'PIGD_score', 'is_TD', 'is_Intermediate', 'is_PIGD']
    tremor_idxs = [feature_map[ft] for ft in ['NP2TRMR', 'NP3PTRMR', 'NP3PTRML', 'NP3KTRMR',
                                              'NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL',
                                              'NP3RTALL', 'NP3RTALJ', 'NP3RTCON']]
    pigd_idxs = [feature_map[ft] for ft in ['NP2WALK', 'NP2FREZ', 'NP3GAIT', 'NP3FRZGT', 'NP3PSTBL']]

    dataframe = dataframe.reindex(columns=list(dataframe.columns)+target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        tremor_score = data_mtx[visit, tremor_idxs].mean()
        pigd_score = data_mtx[visit, pigd_idxs].mean()
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

        dataframe.loc[idx, target_vars] = [tremor_score, pigd_score, is_TD, is_Intermediate, is_PIGD]
    #     print(PATNO, tremor_score, pigd_score, ratio, is_TD, is_Intermediate, is_PIGD)
    # print(dataframe[target_vars])
    print("Tremor_PIGD domain finished!")
    return dataframe

# biospecimen
def Bio(dataframe, seq_data, feature_map, visit):
    target_vars = ['alpha_syn', 'abeta_42', 'p_tau181p', 'total_tau',
                   'abeta_42_total_tau_ratio', 'abeta_42_alpha_syn_ratio',
                   'p_tau181p_alpha_syn_ratio']

    alpha_syn_idxs = [feature_map[ft] for ft in ["alpha_syn"]]
    abeta_42_idxs = [feature_map[ft] for ft in ["abeta_42"]]
    p_tau181p_idxs = [feature_map[ft] for ft in ["p_tau181p"]]
    total_tau_idxs = [feature_map[ft] for ft in ["total_tau"]]

    dataframe = dataframe.reindex(columns=list(dataframe.columns) + target_vars)
    for idx, row in dataframe.iterrows():
        PATNO = row['PATNO']
        if PATNO not in seq_data:
            print("Error: PATNO %s not in sequence data!" % PATNO)
            return
        data_mtx = seq_data[PATNO]
        if visit >= len(data_mtx):
            continue

        alpha_syn = data_mtx[visit, alpha_syn_idxs].sum()
        abeta_42 = data_mtx[visit, abeta_42_idxs].sum()
        p_tau181p = data_mtx[visit, p_tau181p_idxs].sum()
        total_tau = data_mtx[visit, total_tau_idxs].sum()

        abeta_42_total_tau_ratio = abeta_42 / total_tau
        abeta_42_alpha_syn_ratio = abeta_42 / alpha_syn
        p_tau181p_alpha_syn_ratio = p_tau181p / alpha_syn

        dataframe.loc[idx, target_vars] = [alpha_syn, abeta_42, p_tau181p, total_tau,
                                           abeta_42_total_tau_ratio, abeta_42_alpha_syn_ratio, p_tau181p_alpha_syn_ratio]
    #     print(alpha_syn, abeta_42, p_tau181p, total_tau,
    #                                        abeta_42_total_tau_ratio, abeta_42_alpha_syn_ratio, p_tau181p_alpha_syn_ratio)
    # print(dataframe[target_vars])
    print("Biospecimen domain finished!")
    return dataframe

# unsolved: { 'upsit': ['UPSITBK1', 'UPSITBK2', 'UPSITBK3', 'UPSITBK4']}

