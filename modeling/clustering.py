"""
modeling.feature_selection
----------------------------

This module provides diferent aproaches to the clustering task:
    - Kmeans
"""

# append the path of the parent directory
import sys
sys.path.append("..")

# packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# local
from pre_process import read_csv_file
from pre_process import (
    relevance_filter, 
    relevance_redundancy_filter,
    pca
)


def kmeans_clustering(X_train, n_clusters, debug=False):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    predicted = kmeans.fit_predict(X_train)

    if debug:
        print('Labels:', kmeans.labels_, len(kmeans.labels_))
        print('Prediction:', predicted, len(predicted))
        print('Clusters centers:', kmeans.cluster_centers_)
        print('Number of iterations:', kmeans.n_iter_)
        
        # plt.scatter(X_train[:, 0], X_train[:, 1], c=predicted)
        

        fig = plt.figure(figsize=(10, 10))
        axis = fig.add_subplot(111, projection='3d')
        axis.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=predicted)
        axis.set_xlabel("PC1", fontsize=10)
        axis.set_ylabel("PC2", fontsize=10)
        axis.set_zlabel("PC3", fontsize=10)
        plt.show()
        # dataset = X_train.copy()
        # dataset['cluster'] = predicted
        # for c1 in dataset.columns[:-1]:
        #     for c2 in dataset.columns[:-1]:
        #         sns.scatterplot(
        #             data=dataset,
        #             x=c1,
        #             y=c2,
        #             hue="cluster",
        #             style="cluster"
        #         )
        #         plt.show()
    return predicted


def elbow_method(data):
    Nc = range(1, 20)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    score = [kmeans[i].fit(data).inertia_ for i in range(len(kmeans))]
    # inertia -> sum of the squared distances of samples to
    # their closest cluster center
    plt.plot(Nc, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Curve')
    plt.xticks(Nc)
    plt.show()


if __name__ == "__main__":

    df = read_csv_file('../datasets/normalization/trips_mm_s')

    print('Dataset shape:', df.shape)
    # train_set = df[:7293]
    # test_set = df[7293:]
    # print('Train size:', train_set.shape)
    # print('Test size:', test_set.shape)

    # silhoette... others...
    # elbow_method(df)

    # kmeans before PCA
    y_pred = kmeans_clustering(df, 3, False)

    # apply PCA
    X_train, X_test = pca(df, None, 0.9, True)
    print('Dataset shape after PCA:', X_train)

    # kmeans after PCA
    y_pred_pca = kmeans_clustering(X_train, 3, True)

    print('Prediction before PCA:', y_pred, len(y_pred))
    print('Prediction after PCA:', y_pred_pca, len(y_pred_pca))
    print('Number of different prediction:', len(y_pred[y_pred != y_pred_pca]))

    # get trips where the prediction was not the same
    idx = np.where(y_pred != y_pred_pca)
    print(df.iloc[idx])
