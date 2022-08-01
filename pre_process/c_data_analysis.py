# packages
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# local
from b_construct_dataset import read_csv_file


df = read_csv_file('../datasets/missing_values/trips_mv_all')

# remove variables that dont relate to the objective of this thesis
#  'trip_start', 'trip_end', 
df = df[(df.columns.difference([
    'light_mode', 'zero_speed_time', 'n_zero_speed', 'n_ignition_on', 'n_ignition_off',
    'n_high_beam', 'n_low_beam', 'n_wipers', 'n_signal_right', 'n_signal_left'
], sort=False))]

print('Dataset shape:', df.shape)

print("------------------- Dataset ------------------- \n")
print(df, "\n")

print("------------------- Describe ------------------- \n")
print(df.describe(), "\n")

print(df[df['duration'] <= 0])
print(df[df['distance'] <= 0])

print("------------------- Info ------------------- \n")
# print(df.info(), "\n")

print("------------------- Missing Values ------------------- \n")
# print(df.isnull().sum().to_string(), "\n")

print("---------------- Column Names with Missing Values ---------------- \n")
# columns_nan = df.columns[df.isnull().any()].tolist()
# print(columns_nan, len(columns_nan), "\n")

# plot to see data distribution
"""
- Density plot gives us a more precise location. It also gives a continuous
  distribution view.
- An advantage of Density Plots over Histograms is that they're better at
  determining the distribution shape because they're not affected by the
  number of bins.
"""
# for c in columns_nan:
#     if c != 'light_mode':
#         sns.boxplot(trips[c])
#         plt.show()
#         sns.distplot(trips[c])
#         plt.show()
