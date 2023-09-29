"""
_*_ coding: utf-8 _*_
"""
import pandas as pd
import numpy as np
import scipy.stats as st


def hyper_geometric(distribution, sample_size, N_0, N_1, N_2, idx):
    x_0 = distribution.loc[idx, 'cluster_0']
    x_1 = distribution.loc[idx, 'cluster_1']
    x_2 = distribution.loc[idx, 'cluster_2']
    n = distribution.loc[idx, 'total']

    if (x_0 / N_0) >= (n / sample_size):
        pvalue_0 = st.hypergeom.sf(x_0 - 1, sample_size, n, N_0, loc=0)
    else:
        pvalue_0 = st.hypergeom.cdf(x_0, sample_size, n, N_0, loc=0)
    if (x_1 / N_1) >= (n / sample_size):
        pvalue_1 = st.hypergeom.sf(x_1 - 1, sample_size, n, N_1, loc=0)
    else:
        pvalue_1 = st.hypergeom.cdf(x_1, sample_size, n, N_1, loc=0)
    if (x_2 / N_2) >= (n / sample_size):
        pvalue_2 = st.hypergeom.sf(x_2 - 1, sample_size, n, N_2, loc=0)
    else:
        pvalue_2 = st.hypergeom.cdf(x_2, sample_size, n, N_2, loc=0)

    return [distribution.loc[idx, 'gene'], pvalue_0, pvalue_1, pvalue_2]


def construct_clusters_distribution():
    snps_data = pd.read_csv('[your directory]/snps_data_label.csv')
    snps_data = snps_data.dropna()

    snps_data_0 = snps_data[snps_data['label'] == '0']
    snps_data_1 = snps_data[snps_data['label'] == '1']
    snps_data_2 = snps_data[snps_data['label'] == '2']

    temp_0 = pd.DataFrame({'cluster_0': snps_data_0.iloc[:, 2:].sum()})
    temp_1 = pd.DataFrame({'cluster_1': snps_data_1.iloc[:, 2:].sum()})
    temp_2 = pd.DataFrame({'cluster_2': snps_data_2.iloc[:, 2:].sum()})
    res = pd.concat((temp_0, temp_1, temp_2), axis=1)
    res['total'] = res.sum(axis=1)
    res = res.reset_index()
    res = res.rename(columns={'index': 'snps'})
    print(res)
    res.to_csv('[your directory]/clusters_distribution.csv', index=False)


def enrichment_analysis():
    distribution = pd.read_csv("[your directory]/clusters_distribution.csv")
    snps_data = pd.read_csv('[your directory]/snps_data_label.csv')
    snps_data = snps_data.dropna()

    snps_data_0 = snps_data[snps_data['label'] == '0']
    snps_data_1 = snps_data[snps_data['label'] == '1']
    snps_data_2 = snps_data[snps_data['label'] == '2']
    N_0 = len(snps_data_0)
    N_1 = len(snps_data_1)
    N_2 = len(snps_data_2)

    sample_size = N_0 + N_1 + N_2
    print(sample_size)
    print(N_0, N_1, N_2)

    rows = []
    for idx in range(len(distribution)):
        row = hyper_geometric(distribution, sample_size, N_0, N_1, N_2, idx)
        rows.append(row)
    res = pd.DataFrame(rows, columns=distribution.columns[0:4])
    res.to_csv("[your directory]/enrichment_res_2.csv", index=False)
    print(res)

    return True


def PPMI_gene():
    gene_df = pd.read_csv('[your directory]/genetic_data.csv')
    cluster = pd.read_csv('[your directory]/cluster.csv') # cluster label of patients
    gene_df = pd.merge(cluster, gene_df, on='PATNO', how='left')
    gene_df = gene_df.drop_duplicates(subset='PATNO', keep='first')

    gene_df_0 = gene_df[gene_df['Agglomerative'] == 0]
    gene_df_1 = gene_df[gene_df['Agglomerative'] == 1]
    gene_df_2 = gene_df[gene_df['Agglomerative'] == 2]

    N_0 = len(gene_df_0)
    N_1 = len(gene_df_1)
    N_2 = len(gene_df_2)

    sample_size = N_0 + N_1 + N_2

    gene_list = ['APOE_e2', 'APOE_e4', 'GBA', 'LRRK2']
    distribution_df = pd.DataFrame(columns=['gene', 'cluster_0', 'cluster_1', 'cluster_2'])
    idx = 0
    for gene in gene_list:
        row = [gene, gene_df_0[gene].sum(), gene_df_1[gene].sum(), gene_df_2[gene].sum()]
        distribution_df.loc[idx] = row
        idx += 1
    distribution_df['total'] = distribution_df.iloc[:, 1:].sum(axis=1)
    print(distribution_df)

    rows = []
    for idx in range(len(distribution_df)):
        row = hyper_geometric(distribution_df, sample_size, N_0, N_1, N_2, idx)
        rows.append(row)
    res = pd.DataFrame(rows, columns=distribution_df.columns[0:4])
    print(res)


def main():
    # construct_clusters_distribution()
    # enrichment_analysis()
    PPMI_gene()


if __name__ == '__main__':
    main()
