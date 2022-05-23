# packages
import matplotlib.pyplot as plt
import seaborn as sns
# local
from b_construct_dataset import read_csv_file


dataset = '../datasets/constructed/trips_until_2022-05-13'
trips = read_csv_file(dataset)

print("------------------- Dataset ------------------- \n")
print(trips, "\n")

print("------------------- Describe ------------------- \n")
print(trips.describe(), "\n")

print("------------------- Info ------------------- \n")
print(trips.info(), "\n")

print("------------------- Missing Values ------------------- \n")
print(trips.isnull().sum().to_string(), "\n")

print("---------------- Column Names with Missing Values ---------------- \n")
columns_nan = trips.columns[trips.isnull().any()].tolist()
print(columns_nan, len(columns_nan), "\n")

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

# analyse light_mode feature (categorical)
sns.histplot(trips['light_mode'])
plt.show()
