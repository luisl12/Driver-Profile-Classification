"""
preprocess.missing_values
-------

This module provides diferent aproaches to deal with missing values:
    1 - Remove rows and columns (only with all values missing)
    2 - Fill with a value
    3 - Fill with mean
    4 - Fill with median
    5 - Fill with mode (for categorical)
    6 - Fill with new category (for categorical)
    7 - Fill with Last observation carried forward (LOCF)
    8 - Predict the value with ml algorithms ?
"""

# packages
from sklearn.cluster import KMeans
# local
from construct_dataset import read_csv_file, store_csv


def fill_hands_on_missing_values(df, column_names):
    if 'n_lod_0' in column_names:
        df['n_lod_0'] = df['n_lod_0'].fillna(0)
    if 'n_lod_1' in column_names:
        df['n_lod_1'] = df['n_lod_1'].fillna(0)
    if 'n_lod_2' in column_names:
        df['n_lod_2'] = df['n_lod_2'].fillna(0)
    if 'n_lod_3' in column_names:
        df['n_lod_3'] = df['n_lod_3'].fillna(0)
    return df


def fill_drowsiness_missing_values(df, column_names):
    if 'n_drowsiness_0' in column_names:
        df['n_drowsiness_0'] = df['n_drowsiness_0'].fillna(0)
    if 'n_drowsiness_1' in column_names:
        df['n_drowsiness_1'] = df['n_drowsiness_1'].fillna(0)
    if 'n_drowsiness_2' in column_names:
        df['n_drowsiness_2'] = df['n_drowsiness_2'].fillna(0)
    if 'n_drowsiness_3' in column_names:
        df['n_drowsiness_3'] = df['n_drowsiness_3'].fillna(0)
    return df


def fill_driving_events_missing_values(df, column_names):
    if 'n_ha' in column_names:
        df['n_ha'] = df['n_ha'].fillna(0)
    if 'n_ha_l' in column_names:
        df['n_ha_l'] = df['n_ha_l'].fillna(0)
    if 'n_ha_m' in column_names:
        df['n_ha_m'] = df['n_ha_m'].fillna(0)
    if 'n_ha_h' in column_names:
        df['n_ha_h'] = df['n_ha_h'].fillna(0)
    if 'n_hb' in column_names:
        df['n_hb'] = df['n_hb'].fillna(0)
    if 'n_hb_l' in column_names:
        df['n_hb_l'] = df['n_hb_l'].fillna(0)
    if 'n_hb_m' in column_names:
        df['n_hb_m'] = df['n_hb_m'].fillna(0)
    if 'n_hb_h' in column_names:
        df['n_hb_h'] = df['n_hb_h'].fillna(0)
    if 'n_hc' in column_names:
        df['n_hc'] = df['n_hc'].fillna(0)
    if 'n_hc_l' in column_names:
        df['n_hc_l'] = df['n_hc_l'].fillna(0)
    if 'n_hc_m' in column_names:
        df['n_hc_m'] = df['n_hc_m'].fillna(0)
    if 'n_hc_h' in column_names:
        df['n_hc_h'] = df['n_hc_h'].fillna(0)
    return df


def fill_distraction_missing_values(df, column_names):
    if 'distraction_time' in column_names:
        df['distraction_time'] = df['distraction_time'].fillna(0)
    if 'n_distractions' in column_names:
        df['n_distractions'] = df['n_distractions'].fillna(0)
    return df


def fill_ignition_missing_values(df, column_names):
    if 'n_ignition_on' in column_names:
        df['n_ignition_on'] = df['n_ignition_on'].fillna(0)
    if 'n_ignition_off' in column_names:
        df['n_ignition_off'] = df['n_ignition_off'].fillna(0)
    return df


def fill_me_aws_missing_values(df, column_names):
    if 'fcw_time' in column_names:
        df['fcw_time'] = df['fcw_time'].fillna(0)
    if 'hmw_time' in column_names:
        df['hmw_time'] = df['hmw_time'].fillna(0)
    if 'ldw_time' in column_names:
        df['ldw_time'] = df['ldw_time'].fillna(0)
    if 'pcw_time' in column_names:
        df['pcw_time'] = df['pcw_time'].fillna(0)
    if 'n_pedestrian_dz' in column_names:
        df['n_pedestrian_dz'] = df['n_pedestrian_dz'].fillna(0)
    if 'light_mode' in column_names:
        # fill with mode (most common value)
        # obter pelas horas
        df['light_mode'] = df['light_mode'] \
              .fillna(df['light_mode'].mode()[0])
    if 'n_tsr_level' in column_names:
        df['n_tsr_level'] = df['n_tsr_level'].fillna(0)
    if 'n_tsr_level_0' in column_names:
        df['n_tsr_level_0'] = df['n_tsr_level_0'].fillna(0)
    if 'n_tsr_level_1' in column_names:
        df['n_tsr_level_1'] = df['n_tsr_level_1'].fillna(0)
    if 'n_tsr_level_2' in column_names:
        df['n_tsr_level_2'] = df['n_tsr_level_2'].fillna(0)
    if 'n_tsr_level_3' in column_names:
        df['n_tsr_level_3'] = df['n_tsr_level_3'].fillna(0)
    if 'n_tsr_level_4' in column_names:
        df['n_tsr_level_4'] = df['n_tsr_level_4'].fillna(0)
    if 'n_tsr_level_5' in column_names:
        df['n_tsr_level_5'] = df['n_tsr_level_5'].fillna(0)
    if 'n_tsr_level_6' in column_names:
        df['n_tsr_level_6'] = df['n_tsr_level_6'].fillna(0)
    if 'n_tsr_level_7' in column_names:
        df['n_tsr_level_7'] = df['n_tsr_level_7'].fillna(0)
    if 'zero_speed_time' in column_names:
        df['zero_speed_time'] = df['zero_speed_time'].fillna(0)
    if 'n_zero_speed' in column_names:
        df['n_zero_speed'] = df['n_zero_speed'].fillna(0)
    return df


def fill_me_car_missing_values(df, column_names):
    if 'n_high_beam' in column_names:
        df['n_high_beam'] = df['n_high_beam'].fillna(0)
    if 'n_low_beam' in column_names:
        df['n_low_beam'] = df['n_low_beam'].fillna(0)
    if 'n_wipers' in column_names:
        df['n_wipers'] = df['n_wipers'].fillna(0)
    if 'n_signal_right' in column_names:
        df['n_signal_right'] = df['n_signal_right'].fillna(0)
    if 'n_signal_left' in column_names:
        df['n_signal_left'] = df['n_signal_left'].fillna(0)
    if 'n_brakes' in column_names:
        df['n_brakes'] = df['n_brakes'].fillna(0)
    if 'speed' in column_names:
        # use the column mean
        df['speed'] = df['speed'].fillna(df['speed'].mean())
    if 'over_speed_limit' in column_names:
        df['over_speed_limit'] = df['over_speed_limit'].fillna(0)
    return df


def fill_me_fcw_missing_values(df, column_names):
    if 'n_fcw' in column_names:
        df['n_fcw'] = df['n_fcw'].fillna(0)
    return df


def fill_me_hmw_missing_values(df, column_names):
    if 'n_hmw' in column_names:
        df['n_hmw'] = df['n_hmw'].fillna(0)
    return df


def fill_me_ldw_missing_values(df, column_names):
    if 'n_ldw' in column_names:
        df['n_ldw'] = df['n_ldw'].fillna(0)
    if 'n_ldw_left' in column_names:
        df['n_ldw_left'] = df['n_ldw_left'].fillna(0)
    if 'n_ldw_right' in column_names:
        df['n_ldw_right'] = df['n_ldw_right'].fillna(0)
    return df


def fill_me_pcw_missing_values(df, column_names):
    if 'n_pcw' in column_names:
        df['n_pcw'] = df['n_pcw'].fillna(0)
    return df


def fill_idreams_fatigue_missing_values(df, column_names):
    if 'n_fatigue_0' in column_names:
        df['n_fatigue_0'] = df['n_fatigue_0'].fillna(0)
    if 'n_fatigue_1' in column_names:
        df['n_fatigue_1'] = df['n_fatigue_1'].fillna(0)
    if 'n_fatigue_2' in column_names:
        df['n_fatigue_2'] = df['n_fatigue_2'].fillna(0)
    if 'n_fatigue_3' in column_names:
        df['n_fatigue_3'] = df['n_fatigue_3'].fillna(0)
    return df


def fill_idreams_headway_missing_values(df, column_names):
    if 'n_headway__1' in column_names:
        df['n_headway__1'] = df['n_headway__1'].fillna(0)
    if 'n_headway_0' in column_names:
        df['n_headway_0'] = df['n_headway_0'].fillna(0)
    if 'n_headway_1' in column_names:
        df['n_headway_1'] = df['n_headway_1'].fillna(0)
    if 'n_headway_2' in column_names:
        df['n_headway_2'] = df['n_headway_2'].fillna(0)
    if 'n_headway_3' in column_names:
        df['n_headway_3'] = df['n_headway_3'].fillna(0)
    return df


def fill_idreams_overtaking_missing_values(df, column_names):
    if 'n_overtaking_0' in column_names:
        df['n_overtaking_0'] = df['n_overtaking_0'].fillna(0)
    if 'n_overtaking_1' in column_names:
        df['n_overtaking_1'] = df['n_overtaking_1'].fillna(0)
    if 'n_overtaking_2' in column_names:
        df['n_overtaking_2'] = df['n_overtaking_2'].fillna(0)
    if 'n_overtaking_3' in column_names:
        df['n_overtaking_3'] = df['n_overtaking_3'].fillna(0)
    return df


def fill_idreams_speeding_missing_values(df, column_names):
    if 'n_speeding_0' in column_names:
        df['n_speeding_0'] = df['n_speeding_0'].fillna(0)
    if 'n_speeding_1' in column_names:
        df['n_speeding_1'] = df['n_speeding_1'].fillna(0)
    if 'n_speeding_2' in column_names:
        df['n_speeding_2'] = df['n_speeding_2'].fillna(0)
    if 'n_speeding_3' in column_names:
        df['n_speeding_3'] = df['n_speeding_3'].fillna(0)
    return df


def delete_missing_values(df):
    df = df.dropna(axis=1, how='all')  # columns with all NaN
    df = df.dropna(how='all')  # rows with all NaN
    # rows with all NaN (dont count start, end, distance and duration)
    df_mid = df.iloc[:, 4:].dropna(how='all')
    df = df.iloc[:, :4].join(df_mid)
    return df


if __name__ == "__main__":

    # read dataset
    trips = read_csv_file('../datasets/trips_v2.2')

    # remove columns/rows that have all values NaN
    trips = delete_missing_values(trips)

    # check columns with null values
    columns_nan = trips.columns[trips.isnull().any()].tolist()
    print("Columns with NaN values:", columns_nan)

    # fill missing values
    trips = fill_hands_on_missing_values(trips, columns_nan)
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

    # store dataset
    store_csv('../datasets/missing_values', 'trips_mv_v1', trips)

    # test kmeans
    trips = trips.iloc[:, 2:]  # remove start and end
    train_set = trips[:60]  # 60 rows
    test_set = trips[60:]  # 15 rows

    kmeans = KMeans(n_clusters=3, random_state=0).fit(train_set)
    print(kmeans.labels_)
    predicted = kmeans.predict(test_set)
    print(predicted)
    print(kmeans.cluster_centers_)
