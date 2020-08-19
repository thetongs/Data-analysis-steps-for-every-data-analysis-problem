## Data preprocessing
# Step 1 : Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2 : Load dataset
dataset = pd.read_csv('Data/data_pre1.csv')
dataset

# Step 3 : Missing data management
# Check for missing values
dataset.isna().sum()

# Check percentage
NAN = [(clm_name, dataset[clm_name].isna().mean() * 100) for clm_name in dataset]
NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])
NAN

# Check columns which are crossing threshold
NAN[NAN['percentage'] > 50]

# Using drop methods from original dataset
dataset.drop(columns = ['L1'], axis = 1, inplace = True)
dataset.head()

# For rest of missing values we can use below method
# Method 1 : replace() with mean, median, mode
dataset.Age = dataset.Age.replace(np.nan, dataset.Age.mean())
# Method 2 : fillna()
dataset.Salary = dataset.Salary.fillna(dataset.Salary.mean())
dataset.Purchased = dataset.Purchased.fillna(dataset.Purchased.mode()[0])
# Method 3 : SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
dataset.loc[:, ['Age', 'Salary']] = imputer.fit_transform(dataset.loc[:, ['Age', 'Salary']])
dataset

# Step 4 : Check data type
# Check data type each column
dataset.dtypes

# If needed change the data type
dataset.Age = dataset.Age.astype('int32')
dataset.dtypes

# Step 5 : Calculate measure of central dependency
# Mean
print("The average value of each columns are below.")
print("Mean\n{}\n".format(dataset.mean()))

# Median
print("The middle value of all the columns are below.")
print("Median\n{}\n".format(dataset.median()))

# Mode
print("The most common value of each column are below.")
print("Mode\n{}\n".format(dataset.mode().iloc[0]))

# Measure of dispersion
# Varience
print("Variance\n{}\n".format(dataset.var()))
print("""
The variance is small
- It means all column datapoints are tend to close together and close to mean.
If variance is big
- It means this column datapoints are spread-out with respect to each other and with respect to mean.
""")

# Standard deviation
print("Standard deviation\n{}\n".format(dataset.std()))
print("""
Standard deviation is small.
- It means data points are tightky clustered around mean.
Standard deviation is big.
- It means data points widely spread as compare to other columns.
""")

# Calculate moments
from scipy.stats import kurtosis
from scipy.stats import skew
# Skewness
print("Skewness\n{}\n".format(dataset.skew()))
skews = dataset.skew()
sk_list = list()

for i in skews:
    if(i == 0):
        sk_list.append("Normally distributed")
    elif(i < 0):
        sk_list.append("Negatively distributed")
    elif(i>0):
        sk_list.append("Positively distributed")
skewness_result = pd.Series(sk_list)
skewness_result.index = dataset.mean().index
print("The details informaton about skewness below.")
print(skewness_result)

# Kurtosis
print("Kurtosis\n{}\n".format(dataset.kurtosis()))
kur = dataset.kurtosis()
sk_list = list()
for i in kur:
    if(i == 0):
        sk_list.append("Mesokurtic")
    elif(i < 0):
        sk_list.append("Leptokurtic")
    elif(i>0):
        sk_list.append("Platykurtic")
kurtosis_result = pd.Series(sk_list)
kurtosis_result.index = dataset.mean().index
print("The details informaton about kurtosis below.")
print(kurtosis_result)

# Problem statement and solution

# Visualization
