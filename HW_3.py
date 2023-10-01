import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)

df = pd.read_csv("chapter-02-car-price.csv")
df.head()
df.shape
df.columns


# Columns to be used in the data set
df_v1 = df[["Make",
           "Model",
           "Year",
           "Engine HP",
           "Engine Cylinders",
           "Transmission Type",
           "Vehicle Style",
           "highway MPG",
           "city mpg",
           "MSRP"]]

# Convert column names
df_v1.columns = df_v1.columns.str.replace(" ", "_").str.lower()

# Missing value detection and filling with 0
for col in df_v1.columns:
    if df_v1[col].isnull().sum() > 0:
        print(f"{col} column, {df_v1[col].isnull().sum()} missing value.")

# engine_hp column, 69 missing value.
# engine_cylinders column, 30 missing value.

df_v1 = df_v1.fillna(0)

df_v1.rename(columns={"msrp": "price"}, inplace=True)

## Q1

df_v1["transmission_type"].mode()

# A1 = AUTOMATIC



## Q2

num_cols = df_v1.select_dtypes(include=[np.number])
corr_matrix = num_cols.corr()

max_correlation = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
top_two_corr = max_correlation.head(2)

print("Correlation Matrix:")
print(corr_matrix)

print("\nThe two features with the highest correlation:")
print(top_two_corr)

# A2 = highway_mpg and city_mpg



## Q3

round(df_v1["price"].mean(), 3)
# 40594.737

# Let's create a variable above_average which is 1 if the price is above its mean value and 0 otherwise.
df_v1['above_average'] = (df_v1["price"] > df_v1["price"].mean()).astype(int)

# Split the data into train/val/test sets with a 60%/20%/20% distribution
df_v1 = df_v1.drop(columns="price")  # Features
df_v1.head()

df_train_full, df_test = train_test_split(df_v1, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.above_average.values
y_val = df_val.above_average.values
y_test = df_test.above_average.values

# Categorical variables

cat_cols = ["make", "model", "transmission_type", "vehicle_style"]

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_train.above_average)

mi = round(df_train[cat_cols].apply(mutual_info_churn_score), 2)
mi.sort_values(ascending=False)

# A3 = transmission_type  0.02

## Q4

X_train = df_train.drop("above_average", axis=1)
X_val = df_val.drop("above_average", axis=1)
X_test = df_test.drop("above_average", axis=1)

num_cols = ['year', 'engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg']


print(f"num_cols : {num_cols}\ncat_cols : {cat_cols}")


train_dict = df_train[cat_cols + num_cols].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)
X_train

model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

val_dict = X_val[cat_cols + num_cols].to_dict(orient='records')
X_val = dv.transform(val_dict)
y_pred = model.predict(X_val)

accuracy = np.round(accuracy_score(y_val, y_pred), 2)
print(f"Q4 : {accuracy}")

## A4 = 0.95



## Q5

features = cat_cols + num_cols

orig_score = accuracy

for c in features:
    subset = features.copy()
    subset.remove(c)

    train_dict = df_train[subset].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    dv.fit(train_dict)

    X_train = dv.transform(train_dict)

    model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    val_dict = df_val[subset].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    y_pred = model.predict(X_val)

    score = accuracy_score(y_val, y_pred)
    print(c, orig_score - score, score)

# make 0.0024548887956357124 0.9475451112043642
# model 0.029731430969366257 0.9202685690306337
# transmission_type 0.004133445237096023 0.9458665547629039
# vehicle_style 0.017981535879143862 0.9320184641208561
# year 0.0016156105749055572 0.9483843894250944
# engine_hp 0.025115400755350348 0.9248845992446496
# engine_cylinders 0.0028745279060008455 0.9471254720939991
# highway_mpg 0.005812001678556444 0.9441879983214435
# city_mpg 0.01756189676877884 0.9324381032312211

## A5 = year



## Q6

df.rename(columns={"MSRP": "price"}, inplace=True)

df["price"] = np.log1p(df["price"])

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.price.values
y_val = df_val.price.values
y_test = df_test.price.values

del df_train["price"]
del df_val["price"]
del df_test["price"]

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)

X_val = dv.transform(val_dict)

for a in [0, 0.01, 0.1, 1, 10]:
    model = Ridge(alpha=a, solver="sag", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    score = np.sqrt(mean_squared_error(y_val, y_pred))

    print(a, round(score, 3))

# 0 0.49
# 0.01 0.49
# 0.1 0.49
# 1 0.49
# 10 0.49

## A6 = 0
