"""
Note that this code will not work with features
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from straikit.eval.metrics.forecasting_metrics import (
    symmetric_mean_absolute_percentage_error
)
from straikit.neural_networks.sequence.ts_conv1d import TSConv1DRegressor
from straikit.preprocessing.normalization.Standard_Scaler import StandardScaler
from straikit.trainer.trainer import Trainer
from basicscaler import average_scaler

from metrics import smape, wape, mape
from mv_xy import multivariate_xy

data_file = "traffic"
tau = None
n = None
test_start_id = None
WINDOW_SIZE = 64
NORMALIZE = True
NORM_MEHOD = "STD"
SHIFT_SIZE = WINDOW_SIZE
START_OVER = True

if data_file == "elec":
    data = pd.read_csv("~/project/ts_research_topics/data/electricity.csv")
    #data.drop(columns="Datetime", inplace=True)
    tau = 24
    n = 7
    test_start_id = data.shape[0] - tau * n

elif data_file == "traffic":
    data = pd.read_csv("~/project/ts_research_topics/data/traffic.csv")
    data.drop(columns='Unnamed: 0', inplace=True)
    tau = 24
    n = 7
    test_start_id = data.shape[0] - tau * n

elif data_file == "wiki":
    data = pd.read_csv("~/project/ts_research_topics/data/wiki.csv")
    data.drop(columns='Unnamed: 0', inplace=True)
    data = data.iloc[:, 30000:]
    tau = 14
    n = 4
    test_start_id = data.shape[0] - tau * n

else:
    raise ValueError("Data file not identified!")


conv_parmas = {
    "kernel_size": (32, 32),
    "hidden_layer_filters": (32, 32),
    "dense_hidden_layer_sizes": (12,),
    "learning_rate_init": 0.01,
    "optimizer_id": "adam",
    "activation": "relu",
    "prediction_step": tau,
}

trainer_param = {
    "batch_size": 64,
    "train_dir": f"./1dcnn_save_model/",
    "nb_epochs": 16,
    "early_stopping_patience": -1,
    "l2_regularizer": 0.0001,
}

cols = list(data)
if NORMALIZE:
    if NORM_MEHOD == "STD":
        ss = StandardScaler()
        data = ss.fit_transform(data)
    else:
        ss = average_scaler()
        data = ss.fit_transform(data)

# plt.plot(data["87"][5000:5500])
# plt.show()
print("Data loaded +++++++++++++++++++++++++")

dataloader = multivariate_xy(WINDOW_SIZE, SHIFT_SIZE, test_start_id, tau, n)


train_x, train_y = dataloader.train_xy(data)

test_x, test_y = dataloader.rolling_test_xy(data)

print("Train and test data prepared ++++++++++++++++++++++++++++++++++")
estimator = TSConv1DRegressor(**conv_parmas)

my_trainer = Trainer(estimator, **trainer_param)

if START_OVER:
    os.system("rm -r ./1dcnn_save_model/*")
my_trainer.fit({"x": train_x, "y": train_y}, {"x": train_x, "y": train_y})

print("++++++++++++++++++++++++++++++Training finished++++++++++++++++++++++++++++++++")

# model predict
total_y_pred = []
for tx, ty in zip(test_x, test_y):
    pred = my_trainer.predict({"x": tx}, yield_batch=False)
    total_y_pred.append(pred)



y_pred = np.hstack(tuple(total_y_pred))
y_true = np.hstack(tuple(test_y)).squeeze()

y_pred = pd.DataFrame(data=np.transpose(y_pred), columns=cols)
y_true = pd.DataFrame(data=np.transpose(y_true), columns=cols)

y_pred = ss.inverse_transform(y_pred)
y_true = ss.inverse_transform(y_true)

y_pred = np.transpose(y_pred.to_numpy())
y_true = np.transpose(y_true.to_numpy())

wape_score = wape(y_pred, y_true)
smape_score = smape(y_pred, y_true)
mape_score = mape(y_pred, y_true)

print(f"WAPE score: {wape_score}, MAPE Score: {mape_score}, SMAPE score: {smape_score}")

for i in range(7,20):
    plt.plot(y_pred[i,:], label="pred")
    plt.plot(y_true[i,:], label="true")
    plt.legend()
    plt.show()

# predict by simple shift
shift_y_pred = []
for tx in test_x:
    shift_y_pred.append(tx[:, -tau:, 0])

shift_y_pred = np.hstack(tuple(shift_y_pred)).squeeze()
shift_y_pred = pd.DataFrame(data=np.transpose(shift_y_pred), columns=cols)
shift_y_pred = ss.inverse_transform(shift_y_pred)
shift_y_pred = np.transpose(shift_y_pred.to_numpy())

s_wape_score = wape(shift_y_pred, y_true)
s_smape_score = smape(shift_y_pred, y_true)
s_mape_score = mape(shift_y_pred, y_true)
# for i in range(7,10):
#     plt.plot(y_true[i,:], label="pred")
#     plt.plot(shift_y_pred[i,:], label="true")
#     plt.legend()
#     plt.show()

print("Scores using shifted values: ")
print(f"WAPE score: {s_wape_score}, MAPE score: {s_mape_score}, SMAPE score: {s_smape_score}")

with open(data_file+'_result.dat','w') as outfile:
    outfile.write(f"WAPE score: {wape_score}, MAPE Score: {mape_score}, SMAPE score: {smape_score}\n")
    outfile.write("Scores using shifted values: \n")
    outfile.write(f"WAPE score: {s_wape_score}, MAPE score: {s_mape_score}, SMAPE score: {s_smape_score}")
