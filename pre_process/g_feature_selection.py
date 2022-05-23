"""
preprocess.feature_selection
----------------------------

This module provides diferent aproaches to the feature selection
    (unsupervised) task:

    - Algorithm 1: Relevance-only filter
    - Algorithm 2: Filter based on relevance and redundancy
        - To measure relevance use one of:
            * Mean Absolute Difference (MAD)
            * Arithmetic Mean (AM)
            * Geometric Mean (GM)
            * Arithmetic Mean Geometric Mean Quotient (AMGM)
            * Mean Median (MM)
            * Variance (VAR)
        - To measure redundancy use one of:
            * Absolute cosine (AC)
            * Correlation coefficient (CC)

Other filter based approaches:
    - Term-variance
    - Laplacian Score
    - Spectral methods
"""

# packages
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
# local
from .b_construct_dataset import read_csv_file


def relevance_filter(df, measure, m):
    """
    Select a subset of features based on the relevance of each feature.
    Sort the features by decreasing order and keep the top m.

    Args:
        df (pandas.DataFrame): Dataset
        measure (str): Relevance measure type, one of:
            - Mean Absolute Difference (MAD)
            - Arithmetic Mean (AM)
            - Geometric Mean (GM)
            - Arithmetic Mean Geometric Mean Quotient (AMGM)
            - Mean Median (MM)
            - Variance (VAR)
        m (int): Top m of features to keep

    Returns:
        pandas.Series: Top m most relevant features
    """

    try:
        assert m <= df.shape[1]
    except AssertionError:
        print("'m' must be <= to the current number of features")
        return None

    if measure == 'MAD':
        relevance = df.mad()
    elif measure == 'AM':
        relevance = df.mean()
    elif measure == 'GM':
        # returns 0 if there are any zeros in column
        # the normalization process can return a lot of zeros for the columns
        # Not a good measure !!!
        relevance = stats.gmean(df.iloc[:, 0])
    elif measure == 'AMGM':
        relevance = np.exp(df).mean() / stats.gmean(np.exp(df))
    elif measure == 'MM':
        relevance = (df.mean() - df.median()).abs()
    elif measure == 'VAR':
        relevance = df.var()

    # sort relevance by decreasing order
    if isinstance(relevance, pd.core.series.Series):
        relevance = relevance.sort_values(ascending=False).index
    else:
        # only for geometric mean
        relevance = np.sort(relevance)[::-1]

    # return top m features
    return relevance[:m]


def relevance_redundancy_filter(df, rev_measure, red_measure, m, ms):
    """
    Apply relevance filter and then check reduntant features.

    Args:
        df (pandas.DataFrame): Dataset
        rev_measure (str): Relevance measure type, one of:
            - Mean Absolute Difference (MAD)
            - Arithmetic Mean (AM)
            - Geometric Mean (GM)
            - Arithmetic Mean Geometric Mean Quotient (AMGM)
            - Mean Median (MM)
            - Variance (VAR)
        red_measure (str): Redundancy measure type, one of:
            - Correlation coefficient (CC)
            - Absolute cosine (AC)
        m (int): Top m of features to keep
        ms (int): Maximum allowed similarity between pairs of features

    Returns:
        pandas.Series: Top m most relevant features
    """

    try:
        assert m <= df.shape[1]
    except AssertionError:
        print("'m' must be <= to the current number of features")
        return None

    # calculate sorted relevance and keep all features
    relevance = relevance_filter(df, rev_measure, df.shape[1])
    print(relevance, len(relevance))

    # keep most relevant feature
    feature_keep = np.array([relevance[0]])

    prev = feature_keep[-1]

    # loop through relevances starting from second
    for i in relevance[1:]:
        # compute similarity (redundancy) of current feature and previous
        if red_measure == 'CC':
            # 0 < CC < 1
            # 0 - min similirity | 1 - max similarity
            s = df[i].corr(df[prev])
        elif red_measure == 'AC':
            # 0 < AC < 1
            # 0 - min similirity (ortogonal) | 1 - max similarity
            s = cosine_similarity(
                df[i].values.reshape(1, -1), df[prev].values.reshape(1, -1)
            )[0, 0]

        print('Similarity entre', i, 'e', prev, ':', s)

        # TODO: se o metodo de similaridade for o CC também funciona para valores negativos
        if s < ms:
            feature_keep = np.append(feature_keep, i)
            prev = i

        print(feature_keep, len(feature_keep))

        # we have enough features
        if len(feature_keep) == m:
            break

    return feature_keep


if __name__ == "__main__":

    # read dataset
    trips = read_csv_file(
        '../datasets/normalization/trips_mm_s'
    )

    print('Trips feature size before:', trips.shape[1])

    # ------------- Relevance Filters ------------- #
    # print('------ Mean Absolute Difference ------')
    # features = relevance_filter(trips, 'MAD', 2)
    # print(features, len(features), '\n')
    print('------ Arithmetic Mean ------')
    features = relevance_filter(trips, 'AM', 10)
    print(features, len(features), '\n')
    # print('------ Geometric Mean ------')
    # features = relevance_filter(trips, 'GM', 30)
    # print(features, len(features), '\n')
    # print('------ Arithmetic Mean Geometric Mean Quotient ------')
    # features = relevance_filter(trips, 'AMGM', 30)
    # print(features, len(features), '\n')
    # print('------ Mean Median ------')
    # features = relevance_filter(trips, 'MM', 30)
    # print(features, len(features), '\n')
    # print('------ Variance ------')
    # features = relevance_filter(trips, 'VAR', 2)
    # print(features, len(features), '\n')

    # ------------- Relevance Redundancy Filters ------------- #
    # features = relevance_redundancy_filter(trips, 'AMGM', 'AC', 30, 0.6)
    # print(features, len(features))

    X_train = trips[features]
    print(X_train, type(X_train))

    # ------------- TEST KMEANS ------------- #
    from sklearn.cluster import KMeans
    train_set = X_train[:7293]
    test_set = X_train[7293:]
    print('Train size:', train_set.shape)
    print('Test size:', test_set.shape)

    kmeans = KMeans(n_clusters=3, random_state=0).fit(train_set)
    print('Labels:', kmeans.labels_, len(kmeans.labels_))
    predicted = kmeans.predict(test_set)
    print('Prediction:', predicted, len(predicted))
    print('Clusters:', kmeans.cluster_centers_)

    X_train['cluster'] = np.concatenate((kmeans.labels_, predicted))
    for c1 in X_train.columns[:-1]:
        for c2 in X_train.columns[:-1]:
            sns.scatterplot(
                data=X_train, x=c1,
                y=c2,
                hue="cluster",
                style="cluster"
            )
            plt.show()

    # pd.options.display.float_format = '{:,.3f}'.format
    # plt.figure(figsize=(20, 20))
    # # annot_kws={'size': 5}
    # # annot=True
    # correlation = X_train.corr()
    # sns.heatmap(
    #     correlation, linewidths=.3, vmax=1, vmin=-1, center=0, cmap='vlag'
    # )
    # correlation = correlation.unstack()
    # correlation = correlation[abs(correlation) >= 0.7]
    # plt.show()
    # print(correlation.to_string())
