"""
preprocess.missing_values
-------

This module provides diferent aproaches to deal with missing values:
    1 - Remove rows and columns (only with all values missing)
    2 - Fill with a value
    3 - Fill with mean
    4 - Fill with median
    5 - Fill with mode
    6 - Fill with new category (for categorical)
    7 - Fill with Last observation carried forward (LOCF)
    8 - Predict the value with ml algorithms ?

Strategy taken (when it is not impossible to calculate a value):
    1 - Check the percentage of missing values in all columns
    2 - If is > 10%: Correlate with other columns
    3 - If is <= 10%: Fill with mean or median or mode
"""

# packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# local
from b_construct_dataset import read_csv_file
from d_trip_light import calculate_time_interval


def fill_hands_on_missing_values(df, column_names):
    if 'n_lod_0' in column_names:
        if not has_too_many_nulls(df, 'n_lod_0'):
            df['n_lod_0'] = df['n_lod_0'].fillna(df['n_lod_0'].median())
    if 'n_lod_1' in column_names:
        if not has_too_many_nulls(df, 'n_lod_1'):
            df['n_lod_1'] = df['n_lod_1'].fillna(df['n_lod_1'].median())
    if 'n_lod_2' in column_names:
        if not has_too_many_nulls(df, 'n_lod_2'):
            df['n_lod_2'] = df['n_lod_2'].fillna(df['n_lod_2'].median())
    if 'n_lod_3' in column_names:
        if not has_too_many_nulls(df, 'n_lod_3'):
            df['n_lod_3'] = df['n_lod_3'].fillna(df['n_lod_3'].median())
    return df


def fill_drowsiness_missing_values(df, column_names):
    if 'n_drowsiness_0' in column_names:
        if not has_too_many_nulls(df, 'n_drowsiness_0'):
            df['n_drowsiness_0'] = df['n_drowsiness_0'] \
                .fillna(df['n_drowsiness_0'].median())
    if 'n_drowsiness_1' in column_names:
        if not has_too_many_nulls(df, 'n_drowsiness_1'):
            df['n_drowsiness_1'] = df['n_drowsiness_1'] \
                .fillna(df['n_drowsiness_1'].median())
    if 'n_drowsiness_2' in column_names:
        if not has_too_many_nulls(df, 'n_drowsiness_2'):
            df['n_drowsiness_2'] = df['n_drowsiness_2'] \
                .fillna(df['n_drowsiness_2'].median())
    if 'n_drowsiness_3' in column_names:
        if not has_too_many_nulls(df, 'n_drowsiness_3'):
            df['n_drowsiness_3'] = df['n_drowsiness_3'] \
                .fillna(df['n_drowsiness_3'].median())
    return df


def fill_driving_events_missing_values(df, column_names):
    if 'n_ha' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_ha'):
            df['n_ha'] = df['n_ha'].fillna(df['n_ha'].median())
    if 'n_ha_l' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_ha_l'):
            df['n_ha_l'] = df['n_ha_l'].fillna(df['n_ha_l'].median())
    if 'n_ha_m' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_ha_m'):
            df['n_ha_m'] = df['n_ha_m'].fillna(df['n_ha_m'].median())
    if 'n_ha_h' in column_names:
        # median or mean (only 1 outlier)
        if not has_too_many_nulls(df, 'n_ha_h'):
            df['n_ha_h'] = df['n_ha_h'].fillna(df['n_ha_h'].median())
    if 'n_hb' in column_names:
        # median or mean (only 1 outlier)
        if not has_too_many_nulls(df, 'n_hb'):
            df['n_hb'] = df['n_hb'].fillna(df['n_hb'].median())
    if 'n_hb_l' in column_names:
        # median or mean (only 1 outlier)
        if not has_too_many_nulls(df, 'n_hb_l'):
            df['n_hb_l'] = df['n_hb_l'].fillna(df['n_hb_l'].median())
    if 'n_hb_m' in column_names:
        # median or mean (only 1 outlier)
        if not has_too_many_nulls(df, 'n_hb_m'):
            df['n_hb_m'] = df['n_hb_m'].fillna(df['n_hb_m'].median())
    if 'n_hb_h' in column_names:
        # median or mean (only 1 outlier)
        if not has_too_many_nulls(df, 'n_hb_h'):
            df['n_hb_h'] = df['n_hb_h'].fillna(df['n_hb_h'].median())
    if 'n_hc' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_hc'):
            df['n_hc'] = df['n_hc'].fillna(df['n_hc'].median())
    if 'n_hc_l' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_hc_l'):
            df['n_hc_l'] = df['n_hc_l'].fillna(df['n_hc_l'].median())
    if 'n_hc_m' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_hc_m'):
            df['n_hc_m'] = df['n_hc_m'].fillna(df['n_hc_m'].median())
    if 'n_hc_h' in column_names:
        # median or mean (only 1 outlier)
        if not has_too_many_nulls(df, 'n_hc_h'):
            df['n_hc_h'] = df['n_hc_h'].fillna(df['n_hc_h'].median())
    return df


def fill_distraction_missing_values(df, column_names):
    if 'distraction_time' in column_names:
        if not has_too_many_nulls(df, 'distraction_time'):
            df['distraction_time'] = df['distraction_time'] \
                .fillna(df['distraction_time'].median())
    if 'n_distractions' in column_names:
        if not has_too_many_nulls(df, 'n_distractions'):
            df['n_distractions'] = df['n_distractions'] \
                .fillna(df['n_distractions'].median())
    return df


def fill_ignition_missing_values(df, column_names):
    if 'n_ignition_on' in column_names:
        # median or mean
        if not has_too_many_nulls(df, 'n_ignition_on'):
            df['n_ignition_on'] = df['n_ignition_on'] \
                .fillna(df['n_ignition_on'].median())
    if 'n_ignition_off' in column_names:
        # median or mean
        if not has_too_many_nulls(df, 'n_ignition_off'):
            df['n_ignition_off'] = df['n_ignition_off'] \
                .fillna(df['n_ignition_off'].median())
    return df


def fill_me_aws_missing_values(df, column_names):
    if 'fcw_time' in column_names:
        # median
        if not has_too_many_nulls(df, 'fcw_time'):
            df['fcw_time'] = df['fcw_time'].fillna(df['fcw_time'].median())
    if 'hmw_time' in column_names:
        # median
        if not has_too_many_nulls(df, 'hmw_time'):
            df['hmw_time'] = df['hmw_time'].fillna(df['hmw_time'].median())
    if 'ldw_time' in column_names:
        # median
        if not has_too_many_nulls(df, 'ldw_time'):
            df['ldw_time'] = df['ldw_time'].fillna(df['ldw_time'].median())
    if 'pcw_time' in column_names:
        # median or mean
        if not has_too_many_nulls(df, 'pcw_time'):
            df['pcw_time'] = df['pcw_time'].fillna(df['pcw_time'].median())
    if 'n_pedestrian_dz' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_pedestrian_dz'):
            df['n_pedestrian_dz'] = df['n_pedestrian_dz'] \
                .fillna(df['n_pedestrian_dz'].median())
    if 'light_mode' in column_names:
        if not has_too_many_nulls(df, 'light_mode'):
            dataset = df[df['light_mode'].isnull()][[
                'trip_start', 'trip_end', 'light_mode'
            ]]
            dataset['trip_start'] = pd.to_datetime(dataset['trip_start'])
            dataset['trip_end'] = pd.to_datetime(dataset['trip_end'])
            for _, row in dataset.iterrows():
                row['light_mode'] = calculate_time_interval(row)
    if 'n_tsr_level' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_tsr_level'):
            df['n_tsr_level'] = df['n_tsr_level'] \
                .fillna(df['n_tsr_level'].median())
    if 'n_tsr_level_0' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_tsr_level_0'):
            df['n_tsr_level_0'] = df['n_tsr_level_0'] \
                .fillna(df['n_tsr_level_0'].median())
    if 'n_tsr_level_1' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_tsr_level_1'):
            df['n_tsr_level_1'] = df['n_tsr_level_1'] \
                .fillna(df['n_tsr_level_1'].median())
    if 'n_tsr_level_2' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_tsr_level_2'):
            df['n_tsr_level_2'] = df['n_tsr_level_2'] \
                .fillna(df['n_tsr_level_2'].median())
    if 'n_tsr_level_3' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_tsr_level_3'):
            df['n_tsr_level_3'] = df['n_tsr_level_3'] \
                .fillna(df['n_tsr_level_3'].median())
    if 'n_tsr_level_4' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_tsr_level_4'):
            df['n_tsr_level_4'] = df['n_tsr_level_4'] \
                .fillna(df['n_tsr_level_4'].median())
    if 'n_tsr_level_5' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_tsr_level_5'):
            df['n_tsr_level_5'] = df['n_tsr_level_5'] \
                .fillna(df['n_tsr_level_5'].median())
    if 'n_tsr_level_6' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_tsr_level_6'):
            df['n_tsr_level_6'] = df['n_tsr_level_6'] \
                .fillna(df['n_tsr_level_6'].median())
    if 'n_tsr_level_7' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_tsr_level_7'):
            df['n_tsr_level_7'] = df['n_tsr_level_7'] \
                .fillna(df['n_tsr_level_7'].median())
    if 'zero_speed_time' in column_names:
        # median
        if not has_too_many_nulls(df, 'zero_speed_time'):
            df['zero_speed_time'] = df['zero_speed_time'] \
                .fillna(df['zero_speed_time'].median())
    if 'n_zero_speed' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_zero_speed'):
            df['n_zero_speed'] = df['n_zero_speed'] \
                .fillna(df['n_zero_speed'].median())
    return df


def fill_me_car_missing_values(df, column_names):
    if 'n_high_beam' in column_names:
        # median or mean
        if not has_too_many_nulls(df, 'n_high_beam'):
            df['n_high_beam'] = df['n_high_beam'] \
                .fillna(df['n_high_beam'].median())
    if 'n_low_beam' in column_names:
        # median or mean
        if not has_too_many_nulls(df, 'n_low_beam'):
            df['n_low_beam'] = df['n_low_beam'] \
                .fillna(df['n_low_beam'].median())
    if 'n_wipers' in column_names:
        # median or mean
        if not has_too_many_nulls(df, 'n_wipers'):
            df['n_wipers'] = df['n_wipers'].fillna(df['n_wipers'].median())
    if 'n_signal_right' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_signal_right'):
            df['n_signal_right'] = df['n_signal_right'] \
                .fillna(df['n_signal_right'].median())
    if 'n_signal_left' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_signal_left'):
            df['n_signal_left'] = df['n_signal_left'] \
                .fillna(df['n_signal_left'].median())
    if 'n_brakes' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_brakes'):
            df['n_brakes'] = df['n_brakes'].fillna(df['n_brakes'].median())
    if 'speed' in column_names:
        # mean
        if not has_too_many_nulls(df, 'speed'):
            df['speed'] = df['speed'].fillna(df['speed'].mean())
    if 'over_speed_limit' in column_names:
        if not has_too_many_nulls(df, 'over_speed_limit'):
            df['over_speed_limit'] = df['over_speed_limit'] \
                .fillna(df['over_speed_limit'].median())
    return df


def fill_me_fcw_missing_values(df, column_names):
    if 'n_fcw' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_fcw'):
            df['n_fcw'] = df['n_fcw'].fillna(df['n_fcw'].median())
    return df


def fill_me_hmw_missing_values(df, column_names):
    if 'n_hmw' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_hmw'):
            df['n_hmw'] = df['n_hmw'].fillna(df['n_hmw'].median())
    return df


def fill_me_ldw_missing_values(df, column_names):
    if 'n_ldw' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_ldw'):
            df['n_ldw'] = df['n_ldw'].fillna(df['n_ldw'].median())
    if 'n_ldw_left' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_ldw_left'):
            df['n_ldw_left'] = df['n_ldw_left'] \
                .fillna(df['n_ldw_left'].median())
    if 'n_ldw_right' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_ldw_right'):
            df['n_ldw_right'] = df['n_ldw_right'] \
                .fillna(df['n_ldw_right'].median())
    return df


def fill_me_pcw_missing_values(df, column_names):
    if 'n_pcw' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_pcw'):
            df['n_pcw'] = df['n_pcw'].fillna(df['n_pcw'].median())
    return df


def fill_idreams_fatigue_missing_values(df, column_names):
    if 'n_fatigue_0' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_fatigue_0'):
            df['n_fatigue_0'] = df['n_fatigue_0'] \
                .fillna(df['n_fatigue_0'].median())
    if 'n_fatigue_1' in column_names:
        # median or mean (only 1 outlier)
        if not has_too_many_nulls(df, 'n_fatigue_1'):
            df['n_fatigue_1'] = df['n_fatigue_1'] \
                .fillna(df['n_fatigue_1'].median())
    if 'n_fatigue_2' in column_names:
        # median or mean (only 1 outlier)
        if not has_too_many_nulls(df, 'n_fatigue_2'):
            df['n_fatigue_2'] = df['n_fatigue_2'] \
                .fillna(df['n_fatigue_2'].median())
    if 'n_fatigue_3' in column_names:
        # median or mean (only 1 outlier)
        if not has_too_many_nulls(df, 'n_fatigue_3'):
            df['n_fatigue_3'] = df['n_fatigue_3'] \
                .fillna(df['n_fatigue_3'].median())
    return df


def fill_idreams_headway_missing_values(df, column_names):
    if 'n_headway__1' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_headway__1'):
            df['n_headway__1'] = df['n_headway__1'] \
                .fillna(df['n_headway__1'].median())
    if 'n_headway_0' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_headway_0'):
            df['n_headway_0'] = df['n_headway_0'] \
                .fillna(df['n_headway_0'].median())
    if 'n_headway_1' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_headway_1'):
            df['n_headway_1'] = df['n_headway_1'] \
                .fillna(df['n_headway_1'].median())
    if 'n_headway_2' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_headway_2'):
            df['n_headway_2'] = df['n_headway_2'] \
                .fillna(df['n_headway_2'].median())
    if 'n_headway_3' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_headway_3'):
            df['n_headway_3'] = df['n_headway_3'] \
                .fillna(df['n_headway_3'].median())
    return df


def fill_idreams_overtaking_missing_values(df, column_names):
    if 'n_overtaking_0' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_overtaking_0'):
            df['n_overtaking_0'] = df['n_overtaking_0'] \
                .fillna(df['n_overtaking_0'].median())
    if 'n_overtaking_1' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_overtaking_1'):
            df['n_overtaking_1'] = df['n_overtaking_1'] \
                .fillna(df['n_overtaking_1'].median())
    if 'n_overtaking_2' in column_names:
        # median or mean (only 1 outlier)
        if not has_too_many_nulls(df, 'n_overtaking_2'):
            df['n_overtaking_2'] = df['n_overtaking_2'] \
                .fillna(df['n_overtaking_2'].median())
    if 'n_overtaking_3' in column_names:
        # median or mean
        if not has_too_many_nulls(df, 'n_overtaking_3'):
            df['n_overtaking_3'] = df['n_overtaking_3'] \
                .fillna(df['n_overtaking_3'].median())
    return df


def fill_idreams_speeding_missing_values(df, column_names):
    if 'n_speeding_0' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_speeding_0'):
            df['n_speeding_0'] = df['n_speeding_0'] \
                .fillna(df['n_speeding_0'].median())
    if 'n_speeding_1' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_speeding_1'):
            df['n_speeding_1'] = df['n_speeding_1'] \
                .fillna(df['n_speeding_1'].median())
    if 'n_speeding_2' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_speeding_2'):
            df['n_speeding_2'] = df['n_speeding_2'] \
                .fillna(df['n_speeding_2'].median())
    if 'n_speeding_3' in column_names:
        # median
        if not has_too_many_nulls(df, 'n_speeding_3'):
            df['n_speeding_3'] = df['n_speeding_3'] \
                .fillna(df['n_speeding_3'].median())
    return df


def delete_missing_values(df):
    df = df.dropna(axis=1, how='all')  # columns with all NaN
    df = df.dropna(how='all')  # rows with all NaN
    # rows with all NaN (dont count start, end, distance and duration)
    df_mid = df.iloc[:, 4:].dropna(how='all')
    df = df.iloc[:, :4].join(df_mid)
    return df


def has_too_many_nulls(df, column_name, percent=0.1):
    n_values = df[column_name].isnull().sum()
    return (n_values / len(df[column_name])) > percent


if __name__ == "__main__":

    # read dataset
    trips = read_csv_file('../datasets/trips_v2.2')

    # remove columns/rows that have all values NaN
    trips = delete_missing_values(trips)

    # check columns with null values
    columns_nan = trips.columns[trips.isnull().any()].tolist()
    print("Columns with NaN values:", columns_nan, len(columns_nan))

    # fill missing values
    trips = fill_hands_on_missing_values(trips, columns_nan)
    trips = fill_drowsiness_missing_values(trips, columns_nan)
    trips = fill_driving_events_missing_values(trips, columns_nan)
    trips = fill_distraction_missing_values(trips, columns_nan)
    trips = fill_ignition_missing_values(trips, columns_nan)
    trips = fill_me_aws_missing_values(trips, columns_nan)
    trips = fill_me_car_missing_values(trips, columns_nan)
    trips = fill_me_fcw_missing_values(trips, columns_nan)
    trips = fill_me_hmw_missing_values(trips, columns_nan)
    trips = fill_me_ldw_missing_values(trips, columns_nan)
    trips = fill_me_pcw_missing_values(trips, columns_nan)
    trips = fill_idreams_fatigue_missing_values(trips, columns_nan)
    trips = fill_idreams_headway_missing_values(trips, columns_nan)
    trips = fill_idreams_overtaking_missing_values(trips, columns_nan)
    trips = fill_idreams_speeding_missing_values(trips, columns_nan)

    pd.options.display.float_format = '{:,.3f}'.format
    print(trips.corr())
    plt.figure(figsize=(20, 20))
    # annot_kws={'size': 5}
    # annot=True
    correlation = trips.corr()
    sns.heatmap(correlation, linewidths=.3, vmax=1, vmin=-1, center=0, cmap='vlag')
    correlation = correlation.unstack()
    correlation = correlation[abs(correlation) >= 0.7]
    plt.show()

    print(correlation.to_string())

    # store dataset
    # store_csv('../datasets/missing_values', 'trips_mv_v1', trips)

    # # test kmeans
    # trips = trips.iloc[:, 2:]  # remove start and end
    # train_set = trips[:60]  # 60 rows
    # test_set = trips[60:]  # 15 rows

    # kmeans = KMeans(n_clusters=3, random_state=0).fit(train_set)
    # print(kmeans.labels_)
    # predicted = kmeans.predict(test_set)
    # print(predicted)
    # print(kmeans.cluster_centers_)
