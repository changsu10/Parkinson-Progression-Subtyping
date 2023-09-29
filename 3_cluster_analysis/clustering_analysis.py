"""

Do clustering analysis

"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import copy
import pickle as pkl
import os
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import mixture
from sklearn import cluster
from sklearn.manifold import TSNE

import scipy.spatial as sp, scipy.cluster.hierarchy as hc

import var_updating

def clustering(data, method, clusterNum, random_state=0):
    """
    :param data: input data
    :param method: GMM, Agglomerative, KMeans
    :param clusterNum: number of cluster

    :return: list of labels
    """

    if method == 'GMM':
        model_gmm = mixture.GaussianMixture(n_components=clusterNum, covariance_type='tied', random_state=random_state)
        model_gmm.fit(data)
        labels = model_gmm.predict(data)
        return labels

    if method == 'Agglomerative':
        model_agg = cluster.AgglomerativeClustering(linkage='ward', n_clusters=clusterNum)
        """
        linkage: “ward”, “complete”, “average”, “single”
        """
        model_agg.fit(data)
        labels = model_agg.labels_
        return labels

    if method == 'KMeans':
        model_km = cluster.KMeans(n_clusters=clusterNum, random_state=random_state)
        model_km.fit(data)
        labels = model_km.labels_
        return labels



def cluster_visualization(X, y, save_path):
    C = len(set(y))
    for c in range(C):
        plt.scatter(X[y == c, 0], X[y == c, 1])
    plt.savefig(save_path)
    plt.close()





def load_visit_info(data_path):
    df = pd.read_csv(data_path + "/patient_visit_info.csv")
    return df.set_index('PATNO')

def load_static_feat(data_path):
    df = pd.read_csv(data_path + "/patient_info.csv")
    return df.set_index('PATNO')

def load_feat_map(data_path):
    idx = 0
    feat_map = {}
    idx_feat_map = {}
    with open(data_path + "/used_features.txt") as f:
        all_lines = f.readlines()
        for line in all_lines:
            feat = line.strip()
            feat_map[feat] = idx
            idx_feat_map[idx] = feat
            idx += 1
    return feat_map, idx_feat_map

def load_seq_data(data_path, imputation_method):
    if imputation_method == None:
        with open(data_path + "/sequence_data.pkl", "rb") as rf:
            patient_arrays = pkl.load(rf)
            return patient_arrays
    else:
        with open(data_path + "/sequence_data_%s_imputation.pkl" % imputation_method, "rb") as rf:
            patient_arrays = pkl.load(rf)
            return patient_arrays


def get_var_table(data_path, imputation_method=None):  # imputation_method: LOCF&FOCB or interpolate, or None
    if imputation_method == None:
        folder = "visit_data" + "/" + "non-imputation"
    else:
        folder = "visit_data" + "/" + imputation_method

    if os.path.exists(folder) == False:
        os.makedirs(folder)

    feat_map, _ = load_feat_map(data_path)

    static_df = load_static_feat(data_path)

    seq_data = load_seq_data(data_path, imputation_method=imputation_method)

    df = pd.DataFrame(data=list(seq_data.keys()), columns=['PATNO'])

    # initialize
    tables = {}
    for v in range(17):
        tables[v] = copy.deepcopy(df)

    # update
    for v in tables.keys():
        tables[v] = var_updating.UPDRS1(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.Epworth(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.GDS(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.STAI(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.SCOPA_Aut(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.MOCA(tables[v], seq_data, static_df, feat_map, v)  # adjusted
        tables[v] = var_updating.Benton(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.Hopkins(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.RBD(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.QUIP(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.LNS(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.Semantic(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.SDM(tables[v], seq_data, feat_map, v)

        tables[v] = var_updating.UPDRS2(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.UPDRS3(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.Schwab(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.Tremor_PIGD(tables[v], seq_data, feat_map, v)
        tables[v] = var_updating.Bio(tables[v], seq_data, feat_map, v)

        tables[v].to_csv(folder+'/'+'V%02d.csv'%v, index=False)
        print("------------ Visit %s finished! -------------" % v)
    print("Generate all tables successfully!")
    return tables

def load_var_table(imputation_method=None, target_visits=list(range(17))): # imputation_method: LOCF&FOCB or interpolate
    if imputation_method == None:
        folder = "visit_data" + "/" + "non-imputation"
    else:
        folder = "visit_data" + "/" + imputation_method

    if os.path.exists(folder) == False:
        print("Table not existing! Run get_var_table() first!")
    tables = {}
    for v in target_visits:
        df = pd.read_csv(folder+'/'+'V%02d.csv'%v)
        tables[v] = df
    return tables


def progression_plot(visit_data, label_df, dispalyed_variables, displayed_visits=[0, 2, 4, 6, 8, 10, 12], output_path=''):
    # print(label_df)
    # print(visit_data)
    label_types = label_df.columns.values.tolist()[1:]

    for l_type in label_types:
        tmp_label_df = label_df[['PATNO', l_type]]
        tables = {}
        for v in visit_data.keys():
            tables[v] = pd.merge(tmp_label_df, visit_data[v], on='PATNO')
            # print(l_type)
            # print(tables[v].columns.values.tolist())
            # print(tables[v])

        # make xticks
        xticks = list(range(len(displayed_visits)))
        xtick_labels = []
        for v in displayed_visits:
            if v == 0:
                xtick_labels.append('BL')
            else:
                xtick_labels.append('V%02d' % v)

        cols = 3
        rows = len(dispalyed_variables) // cols + 1
        figsize = (14, 20)

        fig, axes = plt.subplots(rows, cols, figsize=figsize
                                     #subplot_kw={'xticks': xticks, 'xticklabels':xtick_labels, 'yticks': []}
                                     )
        sns.set(font_scale=0.8)

        fig.subplots_adjust(hspace=0.3, wspace=0.25)

        x = list(range(len(displayed_visits)))

        sns_cols = ['PATNO', l_type] + dispalyed_variables + ['Visit']
        sns_df = pd.DataFrame(columns=sns_cols)
        for v in displayed_visits:
            if v == 0:
                visit = 'BL'
            else:
                visit = 'V%02d' % v
            tmp_df = tables[v][sns_cols[:-1]]
            tmp_df['Visit'] = visit
            sns_df = pd.concat([sns_df, tmp_df])

        clusters = set(label_df[l_type].values)
        for c in clusters:
            c_num = len(label_df[label_df[l_type] == c])
            # sns_df.replace(c, 'subtype_%s (%s)' % (c, c_num))
            sns_df = sns_df.replace({l_type: {c: 'subtype_%s (%s)' % (c, c_num)}})

        i = j = 0
        for var in dispalyed_variables:
            if j == cols:
                j = 0
                i += 1
            sns.factorplot(x="Visit", y=var, hue=l_type, data=sns_df, palette=None, legend=True, ax=axes[i, j])
            j += 1
        for i in range(len(dispalyed_variables)+1):  # close useless plots
            if i != 0:
                plt.close(i+1)

        save_path = output_path + '_' + l_type + '_.png'
        plt.savefig(save_path)
        plt.close()


def main():
    data_path = "[your directory]/processed_data"

    tables = get_var_table(data_path=data_path, imputation_method=None)
    static_df = load_static_feat(data_path).reset_index()
 
     ####----- Clustering analysis -----####
    PD_cohort = list(static_df[static_df['ENROLL_CAT'] == 'PD']['PATNO'].values)
        
    # specify DPPE output version
    p_emb_path = "[directory of DPPE output (patient embedding vectors) used for clustering analysis]"
    
    infile = open(p_emb_path, 'rb')
    new_dict = pkl.load(infile)

    bottleneck_data = []  # use the last layer of encoder
    cat_emb_data = []  # use the concatenation of all hidden states
    studied_PD_cohort = []  # studied PD patient

    for p in PD_cohort:
        if p in new_dict:
            bottleneck_data.append(new_dict[p][0, :])
            dims = new_dict[p].shape
            cat_emb_data.append(new_dict[p].reshape(dims[0]*dims[1]))
            studied_PD_cohort.append(p)

    bottleneck_data = np.asarray(bottleneck_data)
    cat_emb_data = np.asarray(cat_emb_data)

    ## T-sne
    tsne = TSNE(n_components=2)
    bottleneck_tsne = tsne.fit_transform(bottleneck_data)

    ## clustering
    label_df = pd.DataFrame({'PATNO': studied_PD_cohort})

    euclidean_mtx = sp.distance.squareform(sp.distance.pdist(bottleneck_tsne, metric='euclidean'))
    method = 'ward'
    linkage = hc.linkage(bottleneck_tsne, method=method, metric='euclidean')
    sns.clustermap(euclidean_mtx, row_linkage=linkage, col_linkage=linkage)
    plt.savefig(p_emb_path[:-4]+'_bottleneck_'+'_Agglomerative_clustermap.png', dpi=300)
    plt.close()

    labels = clustering(bottleneck_tsne, 'Agglomerative', 3)
    label_df['Agglomerative'] = labels
    cluster_visualization(bottleneck_tsne, labels, p_emb_path[:-4]+'_bottleneck_'+'_Agglomerative_3clus.pdf')

    label_df.to_csv(p_emb_path[:-4]+'_bottleneck_.csv', header=True, index=False, sep=',')

    DIS_VARS = ['updrs1', 'updrs2', 'updrs3', 'Schwab', 'moca', 'SDM', 'HVLT_total_recall',
                      'Semantic_Fluency', 'stai_state', 'stai_trait', 'epworth', 'GDS']
    progression_plot(tables, label_df, DIS_VARS, output_path=p_emb_path[:-4]+'_bottleneck_')


if __name__ == "__main__":
    main()