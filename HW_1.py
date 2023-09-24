import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("master_housing_dataset.csv")
df.head()

def soru_cevap(df):
    print(f"QA1 = Pandas {pd.__version__} version.")
    print(f"QA2 = {df.shape[1]} piece column.")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            print(f"QA3 = {col} column, {df[col].isnull().sum()} missing value.")
    print(f"QA4 = {df.ocean_proximity.nunique()} piece.")
    x = round(df[df.ocean_proximity == "NEAR BAY"]["median_house_value"].mean(), 0)
    print(f"QA5 = {x}")
    print(f"QA6 = {round(df.total_bedrooms.mean(), 4)} (Before filling in missing values.)")
    df["total_bedrooms"].fillna(df["total_bedrooms"].mean(), inplace=True)
    print(f"QA6 Sonuç = {round(df.total_bedrooms.mean(), 4)} (After filling in missing values) - No different")
    island_options = df[df["ocean_proximity"] == "ISLAND"]
    selected_columns = island_options[["housing_median_age", "total_rooms", "total_bedrooms"]]
    X = selected_columns.values
    XTX = X.T @ X  # Matris Çarpımı (X ve X Transpoze Matrisi)
    XTX_inverse = np.linalg.inv(XTX)  # Matrisin tersini hesapla
    y = np.array([950, 1300, 800, 1000, 1300])
    a = XTX_inverse @ X.T
    w = a @ y   # Burda hata alıyorum. Çünkü matrislerden y'nin boyutu 5 X'in boyutu 20640 bu sebeple matris uyumsuzlupu hatası veriyor.
    print(f"QA7 = {round(w[-1], 4)}")


soru_cevap(df)

# QA1 = Pandas 2.1.0 version.
# QA2 = 10 piece column.
# QA3 = total_bedrooms column, 207 missing value.
# QA4 = 5 piece.
# QA5 = 259212.0
# QA6 = 537.8706 (Before filling in missing values.)
# QA6 Sonuç = 537.8706 (After filling in missing values) - No different
# QA7 = 5.6992








