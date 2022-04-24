import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("~/project/ts_research_topics/data/retaildataset.csv")

df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df["day"] = df["Date"].dt.day
df["month"] = df["Date"].dt.month
df = df.set_index('Date')
feature_col = [
    "month",
    "day",
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
]
target_col = "Weekly_Sales"
df["IsHoliday"] = df["IsHoliday"].astype(int)
df[feature_col] = df[feature_col].fillna(0)

print(f"target col {target_col}")
for fc in feature_col:
    print(fc, df[fc].dtype)
stores = df["Store"].unique()
depts = df["Dept"].unique()

selected_s = [32, 45]
selected_d = [1, 2]
full_length = 0
total_series = 0

ts = None
exog = None
for s in stores:
    for d in depts:
        df_sd = df[(df["Store"] == s) & (df["Dept"] == d)]
        if df_sd.shape[0] > 20:
            total_series += 1
        # if df_sd.shape[0] > 0 and df_sd.shape[0] != 143:
        #    print(F"Not matching: {s}, {d}, {df_sd.shape}")
        if df_sd.shape[0] == 143:
            full_length += 1
            if ts is None:
                ts = df_sd[target_col].to_numpy()[np.newaxis, ...]
                exog = df_sd[feature_col].values[np.newaxis, ...]
            else:
                ts = np.concatenate(
                    (ts, df_sd[target_col].values[np.newaxis, ...]), axis=0
                )
                exog = np.concatenate(
                    (exog, df_sd[feature_col].values[np.newaxis, ...]), axis=0
                )
        if df_sd.shape[0] == 143 and s in selected_s and d in selected_d:
            print(
                df_sd[["Fuel_Price", "IsHoliday", "Temperature", "CPI", "Unemployment"]]
            )
            plt.plot(df_sd["Weekly_Sales"])

print(f"shapes: {ts.shape}, {exog.shape}")
print(f"Full series: {full_length}, Total: {total_series}")
plt.show()

# with open('retail_nparray_ts.npy', 'wb') as f:
#     np.save(f, ts)
# with open('retail_nparray_exog.npy', 'wb') as f:
#     np.save(f, exog)

# datas = df["Date"].unique()
# print(datas)

# for i in range(len(datas)):
#     df_d = df[df["Date"]==datas[i]]
#     print(df_d[["Fuel_Price", "IsHoliday", "Temperature", "CPI", "Unemployment", "Store"]])
