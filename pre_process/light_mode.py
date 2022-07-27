"""
preprocess.d_trip_light
-------

This module provides an approach to fill the light_mode feature missing values.
It calculates the time of day where the trip took place mostly (one of):
    - day
    - dusk 
    - night
"""

import pandas as pd
import datetime
from astral import LocationInfo
from astral.sun import sun


def calculate_time_interval(trip):
    """
    Calculates whether the trip took place mostly during the day, dusk or night

    Args:
        trip (pandas.Series): Trip instance

    Returns:
        str: Trip lighting condition
    """
    
    # latitude and longitude fetch
    # sun = Sun(trip['lat'], trip['lon'])
    
    # # date in your machine's local time zone
    # time_zone = pd.to_datetime(trip['trip_start'])
    # sun_rise = sun.get_sunrise_time(time_zone).strftime('%H:%M')
    # sun_dusk = sun.get_sunset_time(time_zone).strftime('%H:%M')

    loc = LocationInfo('', '', '', trip['lat'], trip['lon'])
    s = sun(loc.observer, date=pd.to_datetime(trip['trip_start']))
    sun_rise = s["sunrise"].strftime('%H:%M:%S')
    sun_dusk = s["dusk"].strftime('%H:%M:%S')
    print((
        f'Dawn:    {s["dawn"]}\n'
        f'Sunrise: {s["sunrise"]}\n'
        f'Noon:    {s["noon"]}\n'
        f'Sunset:  {s["sunset"]}\n'
        f'Dusk:    {s["dusk"]}\n'
    ))

    day_init = pd.Timedelta(sun_rise)
    day_final = pd.Timedelta(sun_dusk) - pd.Timedelta('00:00:01')
    dusk_init = pd.Timedelta(sun_dusk)
    dusk_final = pd.Timedelta(sun_dusk) + pd.Timedelta('02:00:00')
    night_init = dusk_final + pd.Timedelta('00:00:01')
    night_final = day_init - pd.Timedelta('00:00:01')

    day_duration = (day_final - day_init).total_seconds()
    dusk_duration = (dusk_final - dusk_init).total_seconds()
    night_duration = (night_init - night_final).total_seconds()
    day_duration = (day_final - day_init).total_seconds()
    dusk_duration = (dusk_final - dusk_init).total_seconds()
    night_duration = (night_init - night_final).total_seconds()

    init_trip_date = pd.Timedelta(str(trip['trip_start'].time()))
    end_trip_date = pd.Timedelta(str(trip['trip_end'].time()))

    


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
