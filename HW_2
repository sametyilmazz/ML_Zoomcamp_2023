import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)

df = pd.read_csv("master_housing_dataset.csv")
df.head()
df.shape
df.columns

df_one = df[df["ocean_proximity"].isin(["<1H OCEAN", "INLAND"])]
df_one.drop("ocean_proximity", axis=1, inplace=True)

# Q-A 1
for col in df_one.columns:
    if df_one[col].isnull().sum() > 0:
        print(f"ANSWER - 1 = {col} column, {df_one[col].isnull().sum()} missing value.")

# ANSWER - 1 = total_bedrooms column, 157 missing value..



# Q-A 2

population_median = df_one["population"].median()

print("ANSWER - 2 Median population:", population_median)
# ANSWER - 2 Median population: 1195.0



# Q-A 3


# Shuffle the dataset with a seed of 42
shuffled_df = df_one.sample(frac=1, random_state=42)

# Split the data into train/val/test sets with a 60%/20%/20% distribution
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

df_train, temp_data = train_test_split(shuffled_df, test_size=1 - train_ratio, random_state=42)
df_val, df_test = train_test_split(temp_data, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

# Apply log transformation to the 'median_house_value' variable
y_train = np.log1p(df_train["median_house_value"])
y_val = np.log1p(df_val["median_house_value"])
y_test = np.log1p(df_test["median_house_value"])

del df_train["median_house_value"]
del df_val["median_house_value"]
del df_test["median_house_value"]

#Linear Regression
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


#RMSE
def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)

# Computing the trainin mean
total_bedrooms_mean_value = df_train['total_bedrooms'].mean()
total_bedrooms_mean_value


def prepare_data(df, mean_replace_value=None):
    df_temp = df.copy()
    if mean_replace_value is None:
        X = df_temp.values

    else:
        df_temp['total_bedrooms'] = df_temp['total_bedrooms'].fillna(mean_replace_value)
        X = df_temp.values
        #print(replace_value)

    return X


X_train = prepare_data(df_train, 0)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_data(df_val, 0)
y_pred = w0 + X_val.dot(w)
rmse_zero_fill = round(rmse(y_val, y_pred), 2)
rmse_zero_fill

# 0.35

X_train = prepare_data(df_train, total_bedrooms_mean_value)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_data(df_val, total_bedrooms_mean_value)
y_pred = w0 + X_val.dot(w)
rmse_mean_fill = round(rmse(y_val, y_pred), 2)
rmse_mean_fill

# 0.35

# Both are equally good

# Q-A 4

#Regularization

r_list = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]

X_train = prepare_data(df_train,0)

for r in r_list:
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    print(r, round(w_0,2))


results = dict()
X_train = prepare_data(df_train, 0)
X_val = prepare_data(df_val, 0)
best_rmse = None
best_rmse_arg = None

for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    y_pred = w_0 + X_val.dot(w)
    print('%6s' %r, rmse(y_val, y_pred))
    raw_rmse = rmse(y_val, y_pred)
    if best_rmse is None:
        best_rmse = raw_rmse
        best_rmse_arg = r
    elif raw_rmse < best_rmse:
        best_rmse = raw_rmse
        best_rmse_arg = r
    results[r] = round(raw_rmse, 2)


print(f"[ANSWER-4] Best RMSE is {best_rmse} for r value: {best_rmse_arg}")

# [ANSWER-4] Best RMSE is 0.3464608098441441 for r value: 0


# Q-A 5

scores = list()
n = len(df_one)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    # split dataset into 3 parts
    df_train = df_one.iloc[idx[:n_train]]
    df_val = df_one.iloc[idx[n_train:n_train + n_val]]
    df_test = df_one.iloc[idx[n_train + n_val:]]



    # reset indexes
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Apply log transformation to the 'median_house_value' variable
    y_train= np.log1p(df_train['median_house_value'])
    y_val = np.log1p(df_val['median_house_value'])
    y_test = np.log1p(df_test['median_house_value'])


    # drop target variable from the dataset
    del df_train['median_house_value']
    del df_val['median_house_value']
    del df_test['median_house_value']


    X_train_seed = prepare_data(df_train, 0)

    w0, w = train_linear_regression(X_train_seed, y_train)

    X_val_seed = prepare_data(df_val, 0)
    y_pred = w0 + X_val_seed.dot(w)
    scores.append(rmse(y_val, y_pred))
    print('for seed =', seed, 'score =', scores[seed], '\n')


print(f"[ANSWER-5] The standard deviation of all the scores is: {round(np.std(scores), 3)}")

# Q-A 6

n = len(df_one)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

idx = np.arange(n)
np.random.seed(9)
np.random.shuffle(idx)

df_train = df_one.iloc[idx[:n_train+n_val]]
df_test = df_one.iloc[idx[n_train+n_val:]]

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.median_house_value.values)
y_test = np.log1p(df_test.median_house_value.values)

del df_train['median_house_value']
del df_test['median_house_value']

X_train = prepare_data(df_train, 0)
w0, w = train_linear_regression_reg(X_train, y_train, r=0.001)

X_test = prepare_data(df_test, 0)
y_pred = w0 + X_test.dot(w)


print(f"[ANSWER-6] The RMSE score in test dataset is: {round(rmse(y_test, y_pred), 2)}")
