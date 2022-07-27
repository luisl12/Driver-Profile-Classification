"""
preprocess.feature_reduction
----------------------------

This module provides diferent aproaches to the feature reduction
    (unsupervised) task:
    - Principal Component Analysis (PCA) - linear technique
    - Singular Value Decomposition (SVD)
    - t-Stochastic Neighbor Embedding (t-SNE) - non-linear technique
"""

# packages
# import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
# local
from .b_construct_dataset import read_csv_file


def pca(X_train, X_test, variance_th, debug=False):

    # model instance to preserve the minimum number of
    # principal components such that variance_th of the variance is retained
    pca = PCA(variance_th)

    # fit -> train model to find the principal components
    # transform -> apply data rotation and dimensionality reduction
    x_train = pca.fit_transform(X_train)

    x_test = None
    if X_test:
        x_test = pca.transform(X_test)

    n_comp = pca.n_components_
    eigen_val = pca.explained_variance_

    # debug
    if debug:
        # eigen_vect = pca.components_
        # print('Components:', eigen_vect)
        print('N components:', n_comp)
        print('Variance for each PC:', eigen_val)
        print('Variance ratio for each PC:', pca.explained_variance_ratio_)
        print('Original feature size:', X_train.shape[1])
        print('After PCA feature size:', x_train.shape[1])
        # n principal components
        pc = range(1, n_comp+1)
        plt.bar(pc, pca.explained_variance_ratio_)
        plt.xlabel('Principal Components')
        plt.ylabel('Variance %')
        plt.title('Best Principal Components')
        plt.xticks(pc)
        plt.show()
        if x_train.shape[1] > 2:
            # PC1 vs PC2 vs PC3
            fig = plt.figure(figsize=(10, 10))
            axis = fig.add_subplot(111, projection='3d')
            axis.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2])
            axis.set_title('PCA Data')
            axis.set_xlabel('PC1', fontsize=10)
            axis.set_ylabel('PC2', fontsize=10)
            axis.set_zlabel('PC3', fontsize=10)
        else:
            plt.scatter(x_train[:, 0], x_train[:, 1])
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('PCA Data')
        plt.show()

    return pd.DataFrame(x_train), pd.DataFrame(x_test)


def svd(X_train, X_test, n_components, variance_th, debug=False):

    # model instance to preserve the minimum number of
    # principal components such that variance_th of the variance is retained
    # Contrary to PCA, this estimator does not center the data before computing
    # the singular value decomposition. This means it can work with sparse
    # matrices efficiently.
    svd = TruncatedSVD(n_components)

    # fit -> train model to find the principal components
    # transform -> apply data rotation and dimensionality reduction
    x_train = svd.fit_transform(X_train)

    x_test = None
    if X_test:
        x_test = svd.transform(X_test)

    n_comp = svd.components_.shape[0]

    best_n_comp = select_n_components(
        svd.explained_variance_ratio_, variance_th
    )

    # debug
    if debug:
        print('N components:', n_comp)
        print('Best N components:', best_n_comp)
        print('Variance for each PC:', svd.explained_variance_)
        print('Variance ratio for each PC:', svd.explained_variance_ratio_)
        print('Total percentage of Variance:', sum(
            svd.explained_variance_ratio_
        ))
        print('Original feature size:', X_train.shape[1])
        print('After SVD feature size:', x_train.shape[1])
        # n principal components
        pc = range(1, n_comp+1)
        plt.bar(pc, svd.explained_variance_ratio_)
        plt.xlabel('Principal Components')
        plt.ylabel('Variance %')
        plt.title('Best Principal Components')
        plt.xticks(pc)
        plt.show()
        if x_train.shape[1] > 2:
            # PC1 vs PC2 vs PC3
            fig = plt.figure(figsize=(10, 10))
            axis = fig.add_subplot(111, projection='3d')
            axis.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2])
            axis.set_title('SVD Data')
            axis.set_xlabel("PC1", fontsize=10)
            axis.set_ylabel("PC2", fontsize=10)
            axis.set_zlabel("PC3", fontsize=10)
        else:
            plt.scatter(x_train[:, 0], x_train[:, 1])
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('SVD Data')
        plt.show()

    return pd.DataFrame(x_train), pd.DataFrame(x_test), best_n_comp


def tsne(X_train, n_components, debug=False):

    tsne = TSNE(n_components).fit_transform(X_train)

    # debug
    if debug:
        print('Original feature size:', X_train.shape[1])
        print('After TSNE feature size:', tsne.shape[1])
        if tsne.shape[1] > 2:
            # PC1 vs PC2 vs PC3
            fig = plt.figure(figsize=(10, 10))
            axis = fig.add_subplot(111, projection='3d')
            axis.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 2])
            axis.set_title('tSNE Data')
            axis.set_xlabel("Dimension 1", fontsize=10)
            axis.set_ylabel("Dimension 2", fontsize=10)
            axis.set_zlabel("Dimension 3", fontsize=10)
        else:
            plt.scatter(tsne[:, 0], tsne[:, 1])
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('tSNE Data')
        plt.show()
    return pd.DataFrame(tsne)


def select_n_components(var_ratio, goal_var):

    total_variance = 0.0
    n_components = 0

    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        # total variance reach goal
        if total_variance >= goal_var:
            break

    return n_components


if __name__ == "__main__":

    # read dataset
    trips = read_csv_file(
        '../datasets/normalization/trips_mm_s'
    )

    # ------------- PCA ------------- #
    X_train, X_test = pca(trips, None, 0.9, True)

    # ------------- SVD ------------- #
    # n_components = trips.shape[1] - 1
    # X_train, X_test, best_n_comp = svd(trips, None, n_components, 0.9, True)
    # X_train, X_test, _ = svd(trips, None, best_n_comp, 0.9, True)

    # ------------- TSNE ------------- #
    # X_train = tsne(trips, 2, True)

    # X_train = pd.DataFrame(X_train)
    # print(X_train, X_train.shape)

    # # ------------- TEST KMEANS ------------- #
    # from sklearn.cluster import KMeans
    # train_set = X_train[:7293]
    # test_set = X_train[7293:]
    # print('Train size:', train_set.shape)
    # print('Test size:', test_set.shape)

    # kmeans = KMeans(n_clusters=3, random_state=0).fit(train_set)
    # print('Labels:', kmeans.labels_, len(kmeans.labels_))
    # predicted = kmeans.predict(test_set)
    # print('Prediction:', predicted, len(predicted))
    # print('Clusters:', kmeans.cluster_centers_)

    # pd.options.display.float_format = '{:,.3f}'.format
    # plt.figure(figsize=(20, 20))
    # # annot_kws={'size': 5}
    # # annot=True
    # correlation = X_train.corr()
    # sns.heatmap(
    #     correlation, linewidths=.3, vmax=1, vmin=-1, center=0, cmap='vlag',
    #     annot=True
    # )
    # correlation = correlation.unstack()
    # correlation = correlation[abs(correlation) >= 0.7]
    # plt.show()
    # print(correlation.to_string())
