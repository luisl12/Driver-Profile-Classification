"""
modeling.clustering
----------------------------

This module provides diferent aproaches to the clustering task:
    - Kmeans
"""

# append the path of the parent directory
import sys
sys.path.append("..")

# packages
import time
from yellowbrick.cluster import silhouette_visualizer
from yellowbrick.cluster.elbow import kelbow_visualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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


def elbow_method(data, path=None, show=False):
    """
    Calculate elbow score for different number of clusters

    Score based on the inertia - sum of the squared distances
    of samples to their closest cluster center

    Args:
        data (pandas.DataFrame): Dataset
        path (str): Path to save
        show (bool): Show or not 
    """
    metrics = ['distortion', 'silhouette', 'calinski_harabasz']
    for m in metrics:
        plt.figure()
        kelbow_visualizer(KMeans(random_state=42), data, k=(2, 10), metric=m, show=show)
        if path and not show:
            plt.savefig(path + '_' + m + '.png')

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

def find_eps_value(data, min_pts, path=None):
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
        plt.savefig(path + '.png')

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


def clusters_info(data, y_pred, path=None, show=False):
    # clusters = pd.Series(y_pred, name='clusters')
    df = data.assign(target=y_pred)
    print(df)
    if path:
        store_csv(path, 'min', df.groupby(['target']).min().T)
        store_csv(path, 'max', df.groupby(['target']).max().T)
        store_csv(path, 'mean', df.groupby(['target']).mean().T)
        store_csv(path, 'std', df.groupby(['target']).std().T)
    if show: 
        print(df.groupby(['target']).min().T)
        print(df.groupby(['target']).max().T)
        print(df.groupby(['target']).mean().T)
        print(df.groupby(['target']).std().T)

def visualize_clusters_with_pca(data, target):
    model = PCA(n_components=2).fit(data)
    X_pc = model.transform(data)
    print(X_pc.shape)

    plt.figure(figsize=(16,7))
    sns.scatterplot(
        x=X_pc[:, 0],
        y=X_pc[:, 1],
        hue=target, 
        palette=sns.color_palette("hls", 2), 
        data=X_pc, 
        legend="full"
    )
    plt.show()

def analyse_via_pca_components(data, n_components=2):
    
    model = TruncatedSVD(n_components=n_components).fit(data)
    X_pc = model.transform(data)

    # number of components
    n_pcs = model.components_.shape[0]

    # get the index of the most important feature on EACH component
    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

    initial_feature_names = list(data.columns)

    # get the names
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

    # show most important features for fist component
    pc0 = model.components_[0]
    top_n_features = 5
    indexes = sorted(range(len(pc0)), key=lambda sub: pc0[sub])[:-(top_n_features+1):-1]
    most_f_names = [initial_feature_names[i] for i in indexes]
    print('Most important feature for component 0:', most_f_names)

    # build the dataframe
    df = pd.DataFrame(dic.items())
    print(df)

    return most_important_names, X_pc

def analyse_via_decision_tree(data, target, n_top_features=3):
    dec = DecisionTreeClassifier(max_depth=5)
    dec.fit(data, target)
    feature_importance = dec.feature_importances_
    print('Decision tree - feature importance:', feature_importance)
    print('Decision tree - most important feature:', data.columns[feature_importance.argmax()])
    indexes = sorted(range(len(feature_importance)), key=lambda sub: feature_importance[sub])[:-(n_top_features+1):-1]
    print('Decision tree - Top features:', data.columns[indexes])
    plot_tree(
        dec,
        feature_names=list(data.columns),
        class_names=['0', '1', '2'],
        filled=True,
        fontsize=10
    )
    plt.title("Decision tree")
    plt.show()

def analyse_via_random_forest(data, target, n_top_features=3):
    rfc = RandomForestClassifier(criterion='entropy', max_depth=10, n_estimators=50)
    rfc.fit(data, target)
    feature_importance = rfc.feature_importances_
    print('Random forest - feature importance:', feature_importance)
    print('Random forest - most important feature:', data.columns[feature_importance.argmax()])
    indexes = sorted(range(len(feature_importance)), key=lambda sub: feature_importance[sub])[:-(n_top_features+1):-1]
    print('Random forest - Top features:', data.columns[indexes])
    plot_tree(
        rfc.estimators_[0],
        feature_names=list(data.columns),
        class_names=['0', '1', '2'],
        filled=True,
        fontsize=8,
        max_depth=4
    )
    plt.title("Random Forest tree")
    plt.show()


if __name__ == "__main__":

    df = read_csv_file('../datasets/supervised/trips_kmeans')
    # df = read_csv_file('../datasets/missing_values/trips_mv_all')

    # remove variables that dont relate to the objective of this thesis
    df = df[(df.columns.difference([
        'trip_start', 'trip_end', 'light_mode', 'zero_speed_time', 'n_zero_speed', 'n_ignition_on',
        'n_ignition_off', 'n_high_beam', 'n_low_beam', 'n_wipers', 'n_signal_right', 'n_signal_left'
    ], sort=False))]
    print('Dataset shape:', df.shape)

    # get aggressive or non-aggressive trips
    df = df[df['target'] == 1]
    df = df[(df.columns.difference(['target'], sort=False))]
    print('Dataset shape:', df.shape)

    print('\n ---------------------- Normalization ---------------------- \n')

    norm_distance = normalize_by_distance(df)
    # norm_duration = normalize_by_duration(df)
    # no_norm = df

    # norms = {
    #     'no_norm': no_norm,
    #     'distance': norm_distance,
    #     'duration': norm_duration
    # }

    print('\n ----------------- Dimensionality Reduction ----------------- \n')

    # for n in norms:
    #     path = 'images/unsupervised/dim_reduction/'
    #     print('---------------------- Normalization:', n, '----------------------')

    # path = 'images/unsupervised/dim_reduction/{}_'.format(n)
    # apply PCA to initial df
    X_train_pca, _ = pca(norm_distance, None, 0.99, debug=True)
    
    # # apply SVD to initial df
    # n_components = norm_distance.shape[1] - 1
    # _, _, best_n_comp = svd(norm_distance, None, n_components, 0.99, debug=False)
    # X_train_svd, _, _ = svd(norm_distance, None, best_n_comp, 0.99, debug=False)

    reductions = {
        # 'no_red': norm_distance,
        'pca': X_train_pca,
        # 'svd': X_train_svd
    }

    print('\n ----------------- Best N clusters/components ----------------- \n')

    # for r in reductions:
    #     path = 'images/unsupervised/best_n_clusters/non_agressive/'
    #     norm = 'norm_distance'
    #     elbow_name = norm + 'elbow_{}'.format(r)
    #     elbow_method(reductions[r], path=path+elbow_name, show=False)
    
    # for r in reductions:
    #     min_pts = (2 * X_train_pca.shape[1]) - 1
    #     path = './images/unsupervised/best_eps_value/{}_'.format(r)
    #     norm = 'no_norm'
    #     find_eps_value(X_train_pca, min_pts, path=None)  # path+norm
        # distance norm: 7, 7, 7
        # duration norm: 0.065, 0.065, 0.065
        # no norm: 75, 75, 75

    # for r in reductions:
    #     path = 'images/unsupervised/best_n_components/{}_'.format(r)
    #     norm = 'no_norm'
    #     gaussian_best_comp(reductions[r], path=path+norm)

    print('\n ------------------ Clustering Approaches ------------------ \n')
    
    # metrics = ['EUCLIDEAN', 'EUCLIDEAN_SQUARE', 'MANHATTAN', 
    #     'CHEBYSHEV', 'CANBERRA', 'CHI_SQUARE']
    metrics = ['EUCLIDEAN_SQUARE']
    k = 2
    for r in reductions:
        for m in metrics:
            k_means = KmeansClustering(n_clusters=k, init='kmeans++', metric=m, data=reductions[r])
            y_pred = k_means.fit_predict(reductions[r])
            path = './images/unsupervised/kmeans/distance_norm/agressive_with_pca/'
            k_means.visualize_clusters(reductions[r], y_pred, path=None, show=True)
            k_means.evaluate_clusters(reductions[r], y_pred, path=None, show=True)

    # metrics = ['manhattan', 'cosine', 'euclidean']
    # metrics = ['euclidean']
    # eps_norms = [7, 0.065, 75]  # distance, duration, no norm
    # eps = eps_norms[0]
    # for r in reductions:
    #     min_pts = 2 * reductions[r].shape[1] 
    #     for i, m in enumerate(metrics):
    #         db = DBSCANClustering(eps=eps, min_pts=min_pts, metric=m)
    #         y_pred = db.fit_predict(reductions[r])
    #         path = './images/unsupervised/dbscan/duration_norm/{}/'.format(r)
    #         db.visualize_clusters(reductions[r], y_pred, path=None, show=True)
    #         db.evaluate_clusters(reductions[r], y_pred, path=None, show=True)


    # covariance_types = ['full', 'tied', 'diag', 'spherical']
    # covariance_types = ['spherical']
    # k = 2
    # for r in reductions:
    #     for c in covariance_types:
    #         gm = GaussianMixtureClustering(n_clusters=k, covariance_type=c, init_params='random')
    #         y_pred = gm.fit_predict(reductions[r])
    #         path = './images/unsupervised/gaussian_mixture/no_norm/{}/'.format(r)
    #         gm.visualize_clusters(reductions[r], y_pred, path=None, show=True)
    #         gm.evaluate_clusters(reductions[r], y_pred, path=None, show=True)


    print('\n --------------------- Cluster Analysis --------------------- \n')

    # best_features, X_pc = analyse_via_pca_components(norm_distance, n_components=2)
    # analyse_via_decision_tree(norm_distance, y_pred, n_top_features=3)
    # analyse_via_random_forest(norm_distance, y_pred, n_top_features=3)
    # path = './info/'
    # clusters_info(data=norm_distance, y_pred=y_pred, path=None, show=True)

    # plt.figure(figsize=(16,7))
    # sns.scatterplot(
    #     x='speed',
    #     y='n_tsr_level',
    #     hue=y_pred, 
    #     palette=sns.color_palette("hls", k), 
    #     data=norm_distance, 
    #     legend="full"
    # )
    # plt.show()

    # plt.figure(figsize=(16,7))
    # sns.scatterplot(
    #     x='n_tsr_level',
    #     y='n_tsr_level_2',
    #     hue=y_pred, 
    #     palette=sns.color_palette("hls", k), 
    #     data=norm_distance, 
    #     legend="full"
    # )
    # plt.show()

    # plt.figure(figsize=(16,7))
    # sns.scatterplot(
    #     x=X_train_pca[0],
    #     y=X_train_pca[1],
    #     hue=y_pred, 
    #     palette=sns.color_palette("hls", k), 
    #     data=X_train_pca, 
    #     legend="full"
    # )
    # plt.show()

    print('\n --------------------- Save Cluster Dataset and Model --------------------- \n')

    # read df again and add new target column and save
    # df = read_csv_file('../datasets/supervised/trips_kmeans')
    # print(df[df['target'] == 1])
    # print(df[df['target'] == 1].index)
    # print(np.array(y_pred) + 1, len(y_pred))
    # df.loc[df[df['target'] == 1].index, 'target'] = np.array(y_pred) + 1
    # df = norm_distance.assign(target=y_pred)
    # store_csv('../datasets/supervised', 'trips_kmeans_agressive', df)