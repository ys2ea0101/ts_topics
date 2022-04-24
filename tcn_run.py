import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from straikit.eval.metrics.forecasting_metrics import (
    symmetric_mean_absolute_percentage_error,
)
from straikit.neural_networks.sequence.ts_conv1d import TSConv1DRegressor
from straikit.preprocessing.normalization.Standard_Scaler import StandardScaler
from straikit.trainer.trainer import Trainer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Permute, Reshape

from metrics import smape, wape
from mv_xy import multivariate_xy
from tcn import TCN, TCN_iter_pred

data_file = "elec"
tau = None
n = None
test_start_id = None
WINDOW_SIZE = 512
NORMALIZE = True

method = 'iter'

if data_file == "elec":
    data = pd.read_csv("~/projects/deepglo/datasets/electricity.csv")
    tau = 24
    n = 7
    test_start_id = data.shape[0] - tau * n

elif data_file == "traffic":
    data = pd.read_csv("~/projects/deepglo/datasets/traffic.csv")
    tau = 24
    n = 7
    test_start_id = data.shape[0] - tau * n

else:
    raise ValueError("Data file not identified!")


tcn_params = {
    #"input_shape": (WINDOW_SIZE, 1),
    "dilations": (1, 2, 4, 8, 16, 32, 64, 128),
    "nb_filters": 16,
    "return_sequences": True,
}

cols = list(data)
if NORMALIZE:
    ss = StandardScaler()
    data = ss.fit_transform(data)

# plt.plot(data["87"][5000:5500])
# plt.show()

dataloader = multivariate_xy(WINDOW_SIZE, test_start_id, tau, n)

print("Data loaded +++++++++++++++++++++++++")
train_x, train_y = dataloader.train_xy(data)

test_x, test_y = dataloader.rolling_test_xy(data)

print("Train and test data prepared ++++++++++++++++++++++++++++++++++")


tcn_layer = TCN(**tcn_params)
print("Receptive field size =", tcn_layer.receptive_field)

if method == "direct":
    m = Sequential(
        [
            tcn_layer,
            Permute((2, 1)),
            Dense(tau, activation="relu"),
            Permute((2, 1)),
            Dense(1, activation="linear"),
            Reshape((-1,)),
        ]
    )

else:
    m = Sequential(
        [
            tcn_layer,
            Dense(16, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )

m.compile(optimizer="adam", loss="mae")

m.fit(train_x, train_y, epochs=16)


print("++++++++++++++++++++++++++++++Training finished++++++++++++++++++++++++++++++++")

total_y_pred = []
for tx, ty in zip(test_x, test_y):
    pred = TCN_iter_pred(m, tx, tau)
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


# for i in range(4,10):
#     plt.plot(y_pred[i,:], label="pred")
#     plt.plot(y_true[i,:], label="true")
#     plt.legend()
#     plt.show()