import sys
sys.path.append("..")

# packages
import pandas as pd
# local
from consensus_kmeans import ConsensusKmeans
from pre_process import (
    read_csv_file,
    store_csv,
    normalize_by_duration,
    normalize_by_distance,
    pca,
    svd
)


if __name__ == "__main__":

    df = read_csv_file('../datasets/missing_values/trips_mv_all')

    # remove variables that dont relate to the objective of this thesis
    df = df[(df.columns.difference([
        'trip_start', 'trip_end', 'light_mode', 'zero_speed_time', 'n_zero_speed', 'n_ignition_on', 
        'n_ignition_off', 'n_high_beam', 'n_low_beam', 'n_wipers', 'n_signal_right', 'n_signal_left'
    ], sort=False))]

    print('Dataset shape:', df.shape)

    # normalize
    norm_distance = normalize_by_distance(df)
    # norm_duration = normalize_by_duration(df)
    
    # dimensionality reduction
    X_train_pca, _ = pca(norm_distance, None, 0.99, False)
    n_components = norm_distance.shape[1] - 1
    _, _, best_n_comp = svd(norm_distance, None, n_components, 0.99, debug=False)
    X_train_svd, _, _ = svd(norm_distance, None, best_n_comp, 0.99, debug=False)

    # consensus kmeans
    kmin = 10
    kmax = 30
    n_ensemble = 250
    linkages = ['average','complete', 'single', 'weighted']
    l = 'average'
    norm = 'distance_norm'
    
    ck = ConsensusKmeans(kmin, kmax, n_ensemble)
    clusters = ck.ensemble(norm_distance)
    
    path='./images/unsupervised/consensus_kmeans/{}/pca_coassoc_'.format(norm) + l
    coassoc = ck.coassoc_matrix(clusters, len(norm_distance), path=None, show=True)

    k = 2
    clusters = ck.coassoc_partition(coassoc, k, l)
    path='./images/unsupervised/consensus_kmeans/{}/no_red_'.format(norm)
    y_pred = ck.visualize_clusters(norm_distance, clusters, path=None, show=True)  # path+'clusters_'+l
    ck.evaluate_clusters(norm_distance, y_pred, path=path, show=True)
    

    # read df again and add new target column and save
    # df = read_csv_file(df_path)
    # df = df.assign(target=y_pred)
    # store_csv('../datasets/supervised', 'trips_consensus_kmeans', df)

