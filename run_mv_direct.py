import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from straikit.eval.metrics.forecasting_metrics import (
    symmetric_mean_absolute_percentage_error,
)
from straikit.neural_networks.sequence.ts_conv_direct import TSConv1DDirect
from straikit.preprocessing.normalization.Standard_Scaler import StandardScaler
from straikit.trainer.trainer import Trainer

from metrics import smape, wape
from mv_xy import multivariate_xy

data_file = "elec"
tau = None
n = None
test_start_id = None
WINDOW_SIZE = 64
NORMALIZE = True
START_OVER = True

if data_file == "elec":
    data = pd.read_csv("~/projects/deepglo/datasets/electricity.csv")
    data.drop(columns="Datetime", inplace=True)
    tau = 24
    n = 7
    test_start_id = data.shape[0] - tau * n

elif data_file == "traffic":
    data = pd.read_csv("~/projects/deepglo/datasets/traffic.csv")
    data.drop(columns="Datetime", inplace=True)
    tau = 24
    n = 7
    test_start_id = data.shape[0] - tau * n

else:
    raise ValueError("Data file not identified!")


conv_parmas = {
    "kernel_size": (20, 20, 20),
    "hidden_layer_filters": (16, 16, 16),
    "learning_rate_init": 0.01,
    "optimizer_id": "adam",
    "activation": "relu",
    "prediction_step": tau,
}

trainer_param = {
    "batch_size": 64,
    "train_dir": f"./1dcnn_save_model/",
    "nb_epochs": 32,
    "early_stopping_patience": -1,
    "l2_regularizer": 0.0001,
}

cols = list(data)
if NORMALIZE:
    ss = StandardScaler()
    data = ss.fit_transform(data)

# plt.plot(data["87"][5000:5500])
# plt.show()

dataloader = multivariate_xy(
    WINDOW_SIZE, WINDOW_SIZE, test_start_id, tau, n, direct=True
)

print("Data loaded +++++++++++++++++++++++++")
train_x, train_y = dataloader.train_xy(data)

test_x, test_y = dataloader.rolling_test_xy(data)


# shape manipulation to fit the format of direct model
train_y = train_y.squeeze()
test_y = [ty.squeeze() for ty in test_y]

print("Train and test data prepared ++++++++++++++++++++++++++++++++++")
estimator = TSConv1DDirect(**conv_parmas)

my_trainer = Trainer(estimator, **trainer_param)

if START_OVER:
    os.system("rm -r ./1dcnn_save_model/*")
my_trainer.fit({"x": train_x, "y": train_y}, {"x": train_x, "y": train_y})

print("++++++++++++++++++++++++++++++Training finished++++++++++++++++++++++++++++++++")

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

print(f"WAPE score: {wape_score}, SMAPE score: {smape_score}")

print(f"shapes: ({y_pred.shape}, {y_true.shape})")

for i in range(7, 20):
    plt.plot(y_pred[i, :], label="pred")
    plt.plot(y_true[i, :], label="true")
    plt.legend()
    plt.show()
