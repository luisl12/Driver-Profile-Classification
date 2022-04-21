"""
preprocess.normalization
-------

This module provides diferent aproaches to the normalization task:
    1 - StandardScaler
    2 - MinMaxScaler
    3 - MaxAbsScaler
    4 - RobustScaler
    5 - PowerTransformer
    6 - QuantileTransformer (Uniform output)
    7 - QuantileTransformer (Gaussian output)
    8 - Normalizer
    9 - Trip distance? Trip duration?

    TODO: Normalize all features?
"""

# packages
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    Normalizer
)
from sklearn.cluster import KMeans
import pandas as pd
# local
from construct_dataset import read_csv_file


def standard_scaler(df, with_mean=True, with_std=True):
    """
    Apply StandardScaler -> (x - mean(x)) / stdev(x)

    Args:
        df (pandas.DataFrame): Dataset
        with_mean (bool, optional): If True, center the data before scaling.
                                    Defaults to True.
        with_std (bool, optional): If True, scale the data to unit variance.
                                    Defaults to True.

    Returns:
        pandas.DataFrame: Dataset scaled
    """
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    # fit -> Compute the mean and std to be used for later scaling.
    # transform -> Perform standardization by centering and scaling.
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)


def min_max_scaler(df, feature_range=(0, 1)):
    """
    Apply MinMaxScaler -> (x - min(x)) / (max(x) - min(x))

    Args:
        df (pandas.DataFrame): Dataset
        feature_range (tuple, optional): Desired range of transformed data.
                                         Defaults to (0, 1).

    Returns:
        pandas.DataFrame: Dataset scaled
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    # fit -> Compute the minimum and maximum to be used for later scaling.
    # transform -> Scale features of X according to feature_range.
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)


def max_abs_scaler(df):
    """
    Apply MaxAbsScaler -> Equal to MinMaxScaler but only positive values [0, 1]

    Args:
        df (pandas.DataFrame): Dataset

    Returns:
        pandas.DataFrame: Dataset scaled
    """
    scaler = MaxAbsScaler()
    # fit -> Compute the maximum absolute value to be used for later scaling.
    # transform -> Scale the data.
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)


def robust_scaler(df, with_centering=True, with_scaling=True,
                  quantile_range=(25.0, 75.0), unit_variance=False):
    """
    Apply RobustScaler -> Removes the median and scales the data according to
                          the quantile range

    Args:
        df (pandas.DataFrame): Dataset
        with_centering (bool, optional): If True, center the data before
                                         scaling. Defaults to True.
        with_scaling (bool, optional): If True, scale the data to
                                       interquartile range. Defaults to True.
        quantile_range (tuple, optional): Quantile range used to calculate
                                          scale_. Defaults to (25.0, 75.0).
        unit_variance (bool, optional): If True, scale the data to
                                       interquartile range. Defaults to False.

    Returns:
        pandas.DataFrame: Dataset scaled
    """
    scaler = RobustScaler(
        with_centering=with_centering,
        with_scaling=with_scaling,
        quantile_range=quantile_range,
        unit_variance=unit_variance
    )
    # fit -> Compute the median and quantiles to be used for scaling.
    # transform -> Center and scale the data.
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)


def power_transformer_scaler(df, method='yeo-johnson', standardize=True):
    """
    Apply PowerTransformer -> Apply a power transform featurewise to make data
                              more Gaussian-like

    Args:
        df (pandas.DataFrame): Dataset
        method (str, optional): The power transform method.
                                Defaults to 'yeo-johnson'.
        standardize (bool, optional): Set to True to apply zero-mean,
                                      unit-variance normalization to
                                      the transformed output. Defaults to True.

    Returns:
        pandas.DataFrame: Dataset scaled
    """
    scaler = PowerTransformer(method=method, standardize=standardize)
    # fit -> Estimate the optimal parameter lambda for each feature.
    # transform -> Apply the power transform to each feature using the fitted
    # lambdas.
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)


def quantile_transformer_scaler(df, n_quantiles=1000,
                                output_distribution='uniform',
                                ignore_implicit_zeros=False, subsample=100000,
                                random_state=None):
    """
    Apply QuantileTransformer -> Transform features using quantiles
                                 information.

    Args:
        df (pandas.DataFrame): Dataset
        n_quantiles (int, optional): Number of quantiles to be computed.
                                     Defaults to 1000.
        output_distribution (str, optional): Marginal distribution for the
                                             transformed data.
                                             Defaults to 'uniform'.
        ignore_implicit_zeros (bool, optional): Only applies to sparse
                                                matrices.
                                                Defaults to False.
        subsample (int, optional): Maximum number of samples used to estimate
                                   the quantiles for computational efficiency.
                                   Defaults to 1e5.
        random_state (int, optional): Determines random number generation for
                                      subsampling and smoothing noise.
                                      Defaults to None.

    Returns:
        pandas.DataFrame: Dataset scaled
    """
    scaler = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        ignore_implicit_zeros=ignore_implicit_zeros,
        subsample=subsample,
        random_state=random_state
    )
    # fit -> Compute the quantiles used for transforming.
    # transform -> Feature-wise transformation of the data.
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)


def normalizer_scaler(df, norm='l2'):
    """
    Apply Normalizer -> Normalize samples individually to unit norm.

    Args:
        df (pandas.DataFrame): Dataset
        method (str, optional): The norm to use to normalize each non zero
                                sample. If norm='max' is used, values will be
                                rescaled by the maximum of the absolute values.
                                Defaults to 'l2'.

    Returns:
        pandas.DataFrame: Dataset scaled
    """
    scaler = Normalizer(norm=norm)
    # fit -> Do nothing and return the estimator unchanged.
    # transform -> Scale each non zero row of X to unit norm.
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)


if __name__ == "__main__":

    # read dataset
    trips = read_csv_file('../datasets/missing_values/trips_mv_v1')

    # remove start and end
    trips = trips.iloc[:, 2:]

    # standard scaler
    print('-------------- Standard Scaler --------------')
    st_c = standard_scaler(trips)
    print(st_c, '\n')

    # min max scaler
    print('-------------- Min Max Scaler --------------')
    mm_s = min_max_scaler(trips)
    print(mm_s, '\n')

    # max abs scaler
    print('-------------- Max Abs Scaler --------------')
    mabs_s = max_abs_scaler(trips)
    print(mabs_s, '\n')

    # robust scaler
    print('-------------- Robust Scaler --------------')
    r_s = robust_scaler(trips)
    print(r_s, '\n')

    # power transformer scaler
    print('-------------- Power Transformer Scaler --------------')
    pt_s = power_transformer_scaler(trips)
    print(pt_s, '\n')

    # quantile transformer scaler (uniform output)
    print('-------------- Quantile Transformer Scaler (Uniform output) \
           --------------')
    qt_s_u = quantile_transformer_scaler(trips)
    print(qt_s_u, '\n')

    # quantile transformer scaler (gaussian output)
    print('-------------- Quantile Transformer Scaler (Gaussian output) \
           --------------')
    qt_s_g = quantile_transformer_scaler(trips, output_distribution='normal')
    print(qt_s_g, '\n')

    # normalizer scaler
    print('-------------- Normalizer Scaler --------------')
    n_s = normalizer_scaler(trips)
    print(n_s, '\n')

    # test kmeans
    trips = mm_s
    train_set = trips[:60]  # 60 rows
    test_set = trips[60:]  # 15 rows

    kmeans = KMeans(n_clusters=3, random_state=0).fit(train_set)
    print(kmeans.labels_)
    predicted = kmeans.predict(test_set)
    print(predicted)
    print(kmeans.cluster_centers_)
