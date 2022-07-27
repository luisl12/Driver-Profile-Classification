"""
modeling.clustering
----------------------------

This module provides diferent aproaches to the clustering task:
    - Kmeans
"""

# append the path of the parent directory
import sys
from cv2 import dft
sys.path.append("..")

# packages
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# local
from kmeans import KmeansClustering
from dbscan import DBSCANClustering
from gaussian_mixture import GaussianMixtureClustering
from pre_process import (
    read_csv_file,
    store_csv,
    pca,
    tsne,
    svd,
    relevance_redundancy_filter,
    standard_scaler,
    min_max_scaler,
    robust_scaler,
    normalize_by_distance,
    normalize_by_duration,
    label_enconding
)


def elbow_method(data, max_clusters, path=None):
    """
    Calculate elbow score for different number of clusters

    Score based on the inertia - sum of the squared distances
    of samples to their closest cluster center

    Args:
        data (pandas.DataFrame): Dataset
        max_clusters (int): Number of max clusters
    """
    n_c = range(1, max_clusters+1)
    kmeans = [KMeans(n_clusters=i) for i in n_c]
    score = [kmeans[i].fit(data).inertia_ for i in range(len(kmeans))]
    plt.plot(n_c, score, '-o')
    plt.title('Elbow Score Plot')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Curve')
    plt.xticks(n_c)
    if path:
        plt.savefig(path + '.png')
    plt.show()

def silhouette_method(data, max_clusters, path=None):
    """
    Calculate silhouette score for different number of clusters
    The silhouette value measures how similar a point is to its
    own cluster (cohesion) compared to other clusters (separation).

    Score:
        * +1 - Clusters are clearly distinguished
        * 0  - Clusters are neutral in nature and can not be distinguished
        * -1 - Clusters are assigned in the wrong way

    Args:
        data (pandas.DataFrame): Dataset
        max_clusters (int): Number of max clusters
    """
    n_c = range(2, max_clusters+1)
    kmeans = [KMeans(n_clusters=i) for i in n_c]
    score = [
        silhouette_score(
            data, kmeans[i].fit_predict(data)
        ) for i in range(len(kmeans))
    ]
    plt.plot(n_c, score, '-o')
    plt.title('Silhouette Score Plot')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(n_c)
    if path:
        plt.savefig(path + '.png')
    plt.show()

def find_eps_value(data, min_pts):
        """
        Calculate best eps value for the dbscan clustering algorithm

        Algorithm:
            Calculate the average distance between each point and its k
            nearest neighbors, where k=the MinPts value you selected.
            The average k-distances are then plotted in ascending order.
            The optimal value for eps is at the point of maximum curvature

        Args:
            data (pandas.DataFrame): Dataset
            min_pts (int): Minimum number of data points to define a cluster.
        """

        # Calculate the average distance between each point in the
        # dataset and its 20 nearest neighbors (min_pts)
        neighbors = NearestNeighbors(n_neighbors=min_pts, metric='euclidean')
        neighbors_fit = neighbors.fit(data)
        distances, _ = neighbors_fit.kneighbors(data)

        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        plt.plot(distances)
        plt.title('Best eps Plot')
        plt.xlabel('Distance')
        plt.ylabel('Eps')
        plt.show()

def gaussian_best_comp(data, path=None):
    n_components = np.arange(2, 11)
    models = [GaussianMixture(n, n_init=10, random_state=0).fit(data) for n in n_components]
    plt.plot(n_components, [m.bic(data) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(data) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('Score')
    if path:
        plt.savefig(path + '.png')
    plt.show()

def analyse_clusters(data, y_pred):
    clusters = pd.Series(y_pred, name='clusters')
    df = pd.merge(data, clusters, left_index=True, right_index=True)
    # store_csv('./info', 'min', df.groupby(['clusters']).min().T)
    # store_csv('./info', 'max', df.groupby(['clusters']).max().T)
    # store_csv('./info', 'mean', df.groupby(['clusters']).mean().T)
    # store_csv('./info', 'std', df.groupby(['clusters']).std().T)
    # store_csv('./info', 'describe', df.describe())
    print(df.groupby(['clusters']).min().T)
    print(df.groupby(['clusters']).max().T)
    print(df.groupby(['clusters']).mean().T.to_string())
    print(df.groupby(['clusters']).std().T)


if __name__ == "__main__":

    df_path = '../datasets/missing_values/trips_mv_test'
    df = read_csv_file(df_path)

    df_label_encoded = label_enconding(df)

    # remove start, and end
    df_label_encoded = df_label_encoded[(df_label_encoded.columns.difference([
        'trip_start', 'trip_end', 'light_mode'
    ], sort=False))]

    print('Dataset shape:', df_label_encoded.shape)
    # print(df_label_encoded.describe(), "\n")

    print('\n ---------------------- Normalization ---------------------- \n')

    # scaled_df = robust_scaler(df)
    scaled_df = min_max_scaler(df_label_encoded)
    # scaled_df = standard_scaler(df_label_encoded)
    # scaled_df = normalize_by_distance(df)
    # scaled_df = normalize_by_duration(df)

    print('\n ----------------- Dimensionality Reduction ----------------- \n')

    # apply PCA to initial df
    X_train_pca, _ = pca(scaled_df, None, 0.90, False)

    # apply SVD to initial df
    # n_components = scaled_df.shape[1] - 1
    # _, _, best_n_comp = svd(scaled_df, None, n_components, 0.9, False)
    # X_train_svd, _, _ = svd(scaled_df, None, best_n_comp, 0.9, False)

    reductions = {
        'pca': X_train_pca, 
        # 'svd': X_train_svd
    }

    # apply TSNE to initial df -> only good for visualization
    # X_train_tsne = tsne(df, 3, False)

    print('\n ----------------- Best N clusters/components ----------------- \n')

    # path = 'images/unsupervised/best_n_clusters/no_light_mode/'
    # elbow_name = 'elbow_min_max_pca'
    # silhouette_name = 'silhouette_min_max_pca'
    # elbow_method(X_train_pca, 10, path=path+elbow_name)
    # silhouette_method(X_train_pca, 10, path=path+silhouette_name)
    
    # min_pts_pca = 2 * X_train_pca.shape[1]
    # min_pts_svd = 2 * X_train_svd.shape[1]
    # find_eps_value(X_train_pca, min_pts_pca)
    # find_eps_value(X_train_svd, min_pts_svd)

    # gaussian_best_comp(X_train_pca)
    # gaussian_best_comp(X_train_svd)

    """
    Results without scaler:
        PCA: From 63 dimensions to 2
        SVD: From 63 dimensions to 2

    Results with min max scaler:
        PCA: From 63 dimensions to 13
        SVD: From 63 dimensions to 14

    Results with standard scaler:
        PCA: From 63 dimensions to 29
        SVD: From 63 dimensions to 29

    - Sem normalização as dimensões foram reduzidas brutalmente.
    """

    print('\n ------------------ Clustering Approaches ------------------ \n')

    """
    Without scaler:
        PCA: 3 clusters best
        SDV: 3 clusters best

    For min max scaler:
        PCA: 3 clusters best
        SDV: 3 clusters best
        tSNE: 3 clusters best

    For standard scaler:
        PCA: 3 clusters best
        SDV: 3 clusters best
        tSNE: 4 clusters best
    """
    
    metrics = ['EUCLIDEAN', 'EUCLIDEAN_SQUARE', 'MANHATTAN', 
        'CHEBYSHEV', 'CANBERRA', 'CHI_SQUARE']
    metrics = ['EUCLIDEAN_SQUARE']
    k = 2
    for r in reductions:
        for m in metrics:
            print('------------ {} DISTANCE ------------'.format(m))
            k_means = KmeansClustering(n_clusters=k, init='kmeans++', metric=m, data=reductions[r])
            y_pred = k_means.fit_predict(reductions[r])
            path = './images/unsupervised/kmeans/k=2/no_light_mode/standard/{}/'.format(r)
            k_means.visualize_clusters(reductions[r], y_pred, path=None, show=False)
            k_means.evaluate_clusters(reductions[r], y_pred, path=None, show=False)
            analyse_clusters(data=df, y_pred=y_pred)
            # read df again and add new target column and save
            df = read_csv_file(df_path)
            df = df.assign(target=y_pred)
            store_csv('../datasets/supervised', 'trips_test', df)

    # metrics = ['manhattan', 'cosine', 'euclidean']
    # eps = 0.28
    # for r in reductions:
    #     min_pts = 2 * reductions[r].shape[1] 
    #     for i, m in enumerate(metrics):
    #         print('------------ {} DISTANCE ------------'.format(m))
    #         db = DBSCANClustering(eps=eps, min_pts=min_pts, metric=m)
    #         y_pred = db.fit_predict(reductions[r])
    #         path = './images/unsupervised/dbscan/no_light_mode/standard/{}/'.format(r)
    #         db.visualize_clusters(reductions[r], y_pred, path=path+m, show=False)
    #         db.evaluate_clusters(reductions[r], y_pred, path=path, show=False)

    # covariance_types = ['full', 'tied', 'diag', 'spherical']
    # k = 2
    # for r in reductions:
    #     for c in covariance_types:
    #         print('------------ Covariance {} ------------'.format(c))
    #         gm = GaussianMixtureClustering(n_clusters=k, covariance_type=c, init_params='random')
    #         y_pred = gm.fit_predict(reductions[r])
    #         path = './images/unsupervised/gaussian_mixture/no_light_mode/standard/{}/'.format(r)
    #         gm.visualize_clusters(reductions[r], y_pred, path=path+c, show=False)
    #         gm.evaluate_clusters(reductions[r], y_pred, path=path, show=False)
        

    """
    Without scaler:
        PCA: eps best -> 400
        SDV: eps best -> 350

    For min max scaler:
        PCA: eps 0.25 < 0.35 -> best 0.3
        SDV: eps 0.2 < 0.35 -> best 0.3
        tSNE: eps 1 < 1.4 -> best 1.2

    For standard scaler:
        PCA: eps between 7 and 11 -> best 8
        SDV: eps between 9 and 13 -> best 13
        tSNE: eps 1.3 < 1.6 -> Couldnt find best (5 used)
    """

    # methods = ['PCA', 'SVD']  # , 'tSNE']
    # preds = [y_pred_pca, y_pred_svd]  # , y_pred_tsne]
    # print('Prediction before dim reduction:', y_pred_kmeans)
    # for i, v in enumerate(methods):
    #     print('[{}] Prediction after dim reduction:'.format(v), preds[i])
    #     print('[{}] Number of diff predictions:'.format(v),
    #           len(y_pred_kmeans[y_pred_kmeans != preds[i]]))
    #     idx = np.where(y_pred_kmeans != preds[i])
    #     print('[{}] Cluster pred:'.format(v), y_pred_kmeans[idx])
    #     print('[{}] Reduction cluster pred:'.format(v), preds[i][idx])
    #     print(df.iloc[idx], '\n')

    """
    Results without scaler:
    Kmeans:
        - PCA:
            * Calinski score: 14958.936376116293
            * Davies-Bouldin score: 0.7418913771763039
            * Silhouette score: 0.6071458353962321
            * N instances belonging to cluster 0: 11287
            * N instances belonging to cluster 1: 381
            * N instances belonging to cluster 2: 2918
            * Number of diff predictions: 14198
        - SVD:
            * Calinski score: 14961.137281512805
            * Davies-Bouldin score: 0.7419939904282705
            * Silhouette score: 0.6068171776472491
            * N instances belonging to cluster 0: 11278
            * N instances belonging to cluster 1: 381
            * N instances belonging to cluster 2: 2927
            * Number of diff predictions: 14203
    """

    """
    Results with min max scaler:
    Kmeans:
        - PCA:
            * Calinski score: 12332.50977725619
            * Davies-Bouldin score: 1.0696614369458972
            * Silhouette score: 0.5343699219971181
            * N instances belonging to cluster 0: 10697
            * N instances belonging to cluster 1: 1832
            * N instances belonging to cluster 2: 2057
            * Number of diff predictions: 4
        - SVD:
            * Calinski score: 12304.222519067633
            * Davies-Bouldin score: 1.0713501918958421
            * Silhouette score: 0.538450286861021
            * N instances belonging to cluster 0: 10700
            * N instances belonging to cluster 1: 1832
            * N instances belonging to cluster 2: 2054
            * Number of diff predictions: 3
        - tSNE:
            * Calinski score: 7052.730283346609
            * Davies-Bouldin score: 1.1373977478334478
            * Silhouette score: 0.30471468
            * N instances belonging to cluster 0: 5004
            * N instances belonging to cluster 1: 5754
            * N instances belonging to cluster 2: 3828
            * Number of diff predictions: 11428
    DBSCAN:
        - PCA:
            * Calinski score: 6779.5693159035045
            * Davies-Bouldin score: 1.2498112214655002
            * Silhouette score: 0.49296192651305415
            * Number of outliers: 349
            * N instances belonging to cluster -1: 349
            * N instances belonging to cluster 0: 1560
            * N instances belonging to cluster 1: 12442
            * N instances belonging to cluster 2: 235
            * Number of diff predictions: 210
        - SVD:
            * Calinski score: 5041.50098477366
            * Davies-Bouldin score: 1.3220108029335722
            * Silhouette score: 0.4820388811166712
            * Number of outliers: 370
            * N instances belonging to cluster -1: 370
            * N instances belonging to cluster 0: 1548
            * N instances belonging to cluster 1: 12398
            * N instances belonging to cluster 2: 235
            * N instances belonging to cluster 3: 35
            * Number of diff predictions: 177
        - tSNE:
            * Calinski score: 56.42256721185377
            * Davies-Bouldin score: 0.7998977668660558
            * Silhouette score: -0.3090369
            * Number of outliers: 14406
            * N instances belonging to cluster -1: 14375
            * N instances belonging to cluster 0: 39
            * N instances belonging to cluster 1: 31
            * N instances belonging to cluster 2: 31
            * N instances belonging to cluster 3: 29
            * N instances belonging to cluster 4: 27
            * N instances belonging to cluster 5: 27
            * N instances belonging to cluster 6: 27
            * Number of diff predictions: 14294
    Conclusions:
        Dimensionality reduction definitely helps to differentiate clusters
        Kmeans:
            - PCA: Good.
            - SVD: Good.
            - tSNE: Good but a lot of different predictions.
        DBSCAN:
            - PCA: Maybe good.
            - SVD: 4 clusters and cluster 3 only with 35 instances. Bad.
            - tSNE: Way to high number of outliers. Bad.
    """

    """
    Results with standard scaler:
    Kmeans:
        - PCA:
            * Calinski score: 2628.216049653564
            * Davies-Bouldin score: 1.6662856650289821
            * Silhouette score: 0.4429278133888025
            * N instances belonging to cluster 0: 11456
            * N instances belonging to cluster 1: 2951
            * N instances belonging to cluster 2: 179
            * Number of diff predictions: 14298
        - SVD:
            * Calinski score: 2628.226428393188
            * Davies-Bouldin score: 1.6675848886095619
            * Silhouette score: 0.4441537822264973
            * N instances belonging to cluster 0: 11480
            * N instances belonging to cluster 1: 179
            * N instances belonging to cluster 2: 2927
            * Number of diff predictions: 14274
        - tSNE:
            * Calinski score: 6215.842999967155
            * Davies-Bouldin score: 1.1205967218443245
            * Silhouette score: 0.28565562
            * N instances belonging to cluster 0: 4184
            * N instances belonging to cluster 1: 3319
            * N instances belonging to cluster 2: 3605
            * N instances belonging to cluster 3: 3478
            * Number of diff predictions: 12535
    DBSCAN:
        - PCA:
            * Calinski score: 1832.8480295802337
            * Davies-Bouldin score: 1.9845337139173187
            * Silhouette score: 0.7461071366819063
            * N instances belonging to cluster -1: 466
            * N instances belonging to cluster 0: 14120
            * Number of diff predictions: 216
        - SVD:
            * Calinski score: 1613.2137118793662
            * Davies-Bouldin score: 1.8513469145431736
            * Silhouette score: 0.8167923641102609
            * N instances belonging to cluster -1: 197
            * N instances belonging to cluster 0: 14389
            * Number of diff predictions: 53
        - tSNE:
            * Calinski score: 395.7225168977439
            * Davies-Bouldin score: 1.549724071519156
            * Silhouette score: -0.0399373
            * N instances belonging to cluster -1: 16
            * N instances belonging to cluster 0: 13954
            * N instances belonging to cluster 1: 434
            * N instances belonging to cluster 2: 182
            * Number of diff predictions: 860
    Conclusions:
        Dimensionality reduction definitely helps to differentiate clusters.
        Kmeans:
            - PCA: Good.
            - SVD: Good.
            - tSNE: Surprinsingly has fewer diff predictions than PCA and SVD
                    (1 more cluster).
                    - It also has better scores. Best technique.
        DBSCAN:
            - PCA: Its bad. It only finds one cluster and outliers.
            - SVD: Its bad. It only finds one cluster and outliers.
            - tSNE: Its bad. Either has only one cluster, or a lot of clusters
                    and the scores are bad.
    """

    print('\n --------------------- Cluster Analysis --------------------- \n')

    """
    # features = relevance_redundancy_filter(df, 'MAD', 'CC', 5, 0.4)

    # add cluster predictions to the original dataset for better analysis
    # clusters = pd.Series(y_pred_pca, name='clusters')
    # cluster_pca_profile = pd.merge(
    #     df[features], clusters, left_index=True, right_index=True)

    # clusters = pd.Series(y_pred_svd, name='clusters')
    # cluster_svd_profile = pd.merge(
    #     df[features], clusters, left_index=True, right_index=True)

    # clusters = pd.Series(y_pred_tsne, name='clusters')
    # cluster_tsne_profile = pd.merge(
    #     df[features], clusters, left_index=True, right_index=True)

    # for c in cluster_pca_profile:
    #     grid = sns.FacetGrid(cluster_pca_profile, col='clusters')
    #     grid.map(plt.hist, c)
    # plt.show()

    # passed = []
    # for c1 in features:
    #     passed.append(c1)
    #     for c2 in features:
    #         if c1 == c2:
    #             continue
    #         if c2 in passed:
    #             continue
    #         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 15))
    #         sns.scatterplot(
    #             data=cluster_pca_profile,
    #             x=c1,
    #             y=c2,
    #             hue='clusters',
    #             s=85, alpha=0.4, palette='bright', ax=ax1
    #         ).set_title('(PCA) Clusters', fontsize=18)
    #         sns.scatterplot(
    #             data=cluster_svd_profile,
    #             x=c1,
    #             y=c2,
    #             hue='clusters',
    #             s=85, alpha=0.4, palette='bright', ax=ax2
    #         ).set_title('(SVD) Clusters', fontsize=18)
    #         sns.scatterplot(
    #             data=cluster_tsne_profile,
    #             x=c1,
    #             y=c2,
    #             hue='clusters',
    #             s=85, alpha=0.4, palette='bright', ax=ax3
    #         ).set_title('(tSNE) Clusters', fontsize=18)
    #         plt.show()
    """
    
    """
        De forma geral, o standard scaler aparenta ser o melhor quando comparado com o min max. 
        Sem escalamento, a redução dos dados pelo pca e svd é enorme (passam para 2 dim).

        - O kmeans com standard scaler e sem reduçao tem resultados piores do que com redução.
        - O dbscan com standard scaler e sem reduçao tem resultados piores do que com redução,
          mas apenas consegue encontrar 1 cluster sendo que o resto são outliers.
        - O gaussian mixture com scaler e sem reduçao tem resultados piores do que com redução,
          no entanto o min max scaler aparenta ter melhores resultados (em score) do que o standard scaler.

        - Em comparação dos algoritmos:
            - O dbscan é o pior.
            - O kmeans (standard scaler) em termos visuais parece conseguir dividir melhor os clusters 
              mas o gaussian mixture tem melhores scores (min max).
            - No entanto o kmeans acaba por ser a melhor opção.
    """