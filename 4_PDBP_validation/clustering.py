"""
_*_ coding: utf-8 _*_
"""

from sklearn import cluster
from sklearn.manifold import TSNE
import os
import numpy as np
import pandas as pd
import pickle as pkl


class Clustering:

    def __init__(self, input_data, static_data):
        PD_cohort = list(static_data[static_data['case_control_other_at_baseline'] == 'Case']['participant_id'].values)

        bottleneck_data = []  # use the last layer of encoder
        cat_emb_data = []  # use the concatenation of all hidden states
        studied_PD_cohort = []  # studied PD patient

        for p in PD_cohort:
            if p in input_data:
                bottleneck_data.append(input_data[p][0, :])
                dims = input_data[p].shape
                cat_emb_data.append(input_data[p].reshape(dims[0] * dims[1]))
                studied_PD_cohort.append(p)

        bottleneck_data = np.asarray(bottleneck_data)
        cat_emb_data = np.asarray(cat_emb_data)

        ## T-sne
        tsne = TSNE(n_components=2)
        self.bottleneck_tsne = tsne.fit_transform(bottleneck_data)
        self.studied_PD_cohort = studied_PD_cohort

    def tsne_data(self):
        return self.bottleneck_tsne

    def clustering(self, clusterNum, save_path=None):

        label_df = pd.DataFrame({'participant_id': self.studied_PD_cohort})

        model_agg = cluster.AgglomerativeClustering(linkage='ward', n_clusters=clusterNum)
        """
        linkage: “ward”, “complete”, “average”, “single”
        """
        model_agg.fit(self.bottleneck_tsne)
        labels = model_agg.labels_

        label_df['Agglomerative'] = labels

        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path + '/label.csv', 'w') as f:
                f.write(label_df.to_csv(index=False))

        return label_df


def main():
    f_path = '[your directory]/validation/'
    LSTM_output = '[your directory]/validation/LSTM_Output/sequence_data_Zscore_hs32_embNone_nly1_optAdam_lr0.001_1be0.9_2be0.999_eps1e-08_b1_seqL10/'
    with open(LSTM_output + "full_hidden_epoch_50.pkl", "rb") as f:
        cluster_data = pkl.load(f)

    static_data = pd.read_csv(f_path + "processed_data/static_info.csv")

    CLU = Clustering(cluster_data, static_data)
    label_df = CLU.clustering(3, save_path=f_path + "processed_data")
    tsne_data = CLU.tsne_data()
    with open(f_path + 'processed_data/tsne_data.obj', 'wb') as f:
        pkl.dump(tsne_data, f)
    f.close()


if __name__ == '__main__':
    main()
