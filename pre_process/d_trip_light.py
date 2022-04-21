"""
preprocess.d_trip_light
-------

This module provides an approach to fill the light_mode feature missing values.
It calculates the time of day where the trip took place mostly (one of):
    - day -> 05h to 18:30h
    - dusk -> 18:31h to 19:10h
    - night -> 19:11h to 04:59h
"""

import pandas as pd


# define day, dusk and night thresholds
day_init = pd.Timedelta('05:00:00')
day_final = pd.Timedelta('18:30:00')
dusk_init = pd.Timedelta('18:31:00')
dusk_final = pd.Timedelta('19:10:00')
night_init = pd.Timedelta('19:11:00')
night_final = pd.Timedelta('04:59:00')
#  if trip took  more than one day
mid_night_of_next_day = pd.Timedelta('00:00:00')
mid_night = pd.Timedelta('23:59:59')

day_duration = (day_final - day_init).total_seconds()
dusk_duration = (dusk_final - dusk_init).total_seconds()
night_duration = (night_init - night_final).total_seconds()
day_duration = (day_final - day_init).total_seconds()
dusk_duration = (dusk_final - dusk_init).total_seconds()
night_duration = (night_init - night_final).total_seconds()


def calculate_time_interval(trip):
    """
    Calculates whether the trip took place mostly during the day, dusk or night

    Args:
        trip (pandas.Series): Trip instance

    Returns:
        str: Trip lighting condition
    """

    init_trip_date = pd.Timedelta(str(trip['trip_start'].time()))
    end_trip_date = pd.Timedelta(str(trip['trip_end'].time()))

    day_init_date = init_trip_date < day_final and \
        init_trip_date > day_init
    dusk_init_date = init_trip_date < dusk_final and \
        init_trip_date > dusk_init
    night_init_date = (
        init_trip_date > night_init and init_trip_date < mid_night
    ) or (
            init_trip_date < night_final and
            init_trip_date > mid_night_of_next_day
    )

    day_final_date = end_trip_date < day_final and \
        end_trip_date > day_init
    dusk_final_date = end_trip_date < dusk_final and \
        end_trip_date > dusk_init
    night_final_date = (
        end_trip_date > night_init and end_trip_date < mid_night
    ) or (
        end_trip_date < night_final and end_trip_date > mid_night_of_next_day
    )

    # day
    if day_init_date:
        if day_final_date:
            return "day"
        else:
            diff_trip_init_day_final = (day_final - init_trip_date) \
                .total_seconds()
            if dusk_final_date:
                diff_dusk_init_trip_final = (end_trip_date - dusk_init) \
                    .total_seconds()
                if diff_trip_init_day_final > diff_dusk_init_trip_final:
                    return "day"
                else:
                    return "dusk"
            else:
                diff_night_init_trip_final = (end_trip_date - night_init) \
                    .total_seconds()
                if diff_trip_init_day_final > dusk_duration and \
                   diff_trip_init_day_final > diff_night_init_trip_final:
                    return "day"
                elif dusk_duration > diff_night_init_trip_final and \
                        dusk_duration > diff_trip_init_day_final:
                    return "dusk"
                else:
                    return "night"

    # dusk
    elif dusk_init_date:
        if dusk_final_date:
            return "dusk"
        else:
            diff_trip_init_dusk_final = (dusk_final - init_trip_date) \
                .total_seconds()
            if night_final_date:
                diff_night_init_trip_final = (end_trip_date - night_init) \
                    .total_seconds()
                if diff_trip_init_dusk_final > diff_night_init_trip_final:
                    return "dusk"
                else:
                    return "night"
            else:
                diff_day_init_trip_final = (end_trip_date - day_init) \
                    .total_seconds()
                if diff_trip_init_dusk_final > night_duration and \
                        diff_trip_init_dusk_final > diff_day_init_trip_final:
                    return "dusk"
                elif night_duration > diff_trip_init_dusk_final and \
                        night_duration > diff_day_init_trip_final:
                    return "night"
                else:
                    return "day"

    # night
    elif night_init_date:
        if night_final_date:
            return "night"
        else:
            diff_trip_init_night_final = (night_final - init_trip_date) \
                .total_seconds()
            if day_final_date:
                diff_day_init_trip_final = (end_trip_date - day_init) \
                    .total_seconds()
                if diff_trip_init_night_final > diff_day_init_trip_final:
                    return "night"
                else:
                    return "day"
            else:
                diff_dusk_init_trip_final = (end_trip_date - dusk_init) \
                    .total_seconds()
                if diff_trip_init_night_final > day_duration and \
                        diff_trip_init_night_final > diff_dusk_init_trip_final:
                    return "night"
                elif day_duration > diff_trip_init_night_final and \
                        day_duration > diff_dusk_init_trip_final:
                    return "day"
                else:
                    return "dusk"


if __name__ == "__main__":

    d = {
        'trip_start': [
            pd.to_datetime('2021-04-21 05:42:57+00:00'),
            pd.to_datetime('2021-04-21 18:40:57+00:00'),
            pd.to_datetime('2021-04-21 23:42:57+00:00'),
            pd.to_datetime('2021-04-21 10:00:57+00:00')
        ],
        'trip_end': [
            pd.to_datetime('2021-04-21 06:42:57+00:00'),
            pd.to_datetime('2021-04-21 19:20:57+00:00'),
            pd.to_datetime('2021-04-21 01:42:57+00:00'),
            pd.to_datetime('2021-04-21 23:42:57+00:00')
        ],
        'light_mode': [
            None, 'light', None, None
        ]
    }

    df = pd.DataFrame(data=d)
    print('data_frame', df, "\n")

    dataset = df[~df['light_mode'].isnull()][[
        'trip_start', 'trip_end', 'light_mode'
    ]].to_string()

    for index, row in df.iterrows():
        print(row, type(row))
        print(calculate_time_interval(row))
