import sys
sys.path.append("..")

# packages
import pandas as pd
# local
from consensus_kmeans import ConsensusKmeans
from pre_process import (
    read_csv_file,
    store_csv,
    standard_scaler,
    min_max_scaler,
    pca,
    svd,
    label_enconding
)


if __name__ == "__main__":

    # sys.setrecursionlimit(100000)

    df_path = '../datasets/missing_values/trips_mv'
    df = read_csv_file(df_path)

    df_label_encoded = label_enconding(df)

    # remove start, and end
    df_label_encoded = df_label_encoded[(df_label_encoded.columns.difference([
        'trip_start', 'trip_end', 'light_mode'
    ], sort=False))]

    print('Dataset shape:', df_label_encoded.shape)

    # normalize
    # scaled_df = standard_scaler(df_label_encoded)
    scaled_df = min_max_scaler(df_label_encoded)

    # dimensionality reduction
    X_train_pca, _ = pca(scaled_df, None, 0.90, False)

    # n_components = scaled_df.shape[1] - 1
    # _, _, best_n_comp = svd(scaled_df, None, n_components, 0.9, False)
    # X_train_svd, _, _ = svd(scaled_df, None, best_n_comp, 0.9, False)

    # consensus kmeans
    kmin = 10
    kmax = 30
    n_ensemble = 250
    linkages = ['average','complete', 'single', 'weighted']
    # complete only gives 1 cluster
    l = 'average'
    
    ck = ConsensusKmeans(kmin, kmax, n_ensemble)
    clusters = ck.ensemble(X_train_pca)
    path='./images/unsupervised/consensus_kmeans/no_light_mode/min_max_scaler_with_pca/coassoc_' + l
    coassoc = ck.coassoc_matrix(clusters, len(X_train_pca), path=None, show=False)
    k = 2
    clusters = ck.coassoc_partition(coassoc, k, l)
    path='./images/unsupervised/consensus_kmeans/no_light_mode/min_max_scaler_with_pca/'
    y_pred = ck.visualize_clusters(X_train_pca, clusters, path=None, show=True)
    ck.evaluate_clusters(X_train_pca, y_pred, path=None, show=True)

    # read df again and add new target column and save
    df = read_csv_file(df_path)
    df = df.assign(target=y_pred)
    store_csv('../datasets/supervised', 'trips_consensus_kmeans', df)

