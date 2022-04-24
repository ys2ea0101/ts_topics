"""
Note that this code will not work with features
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from straikit.eval.metrics.forecasting_metrics import (
    symmetric_mean_absolute_percentage_error
)
from straikit.neural_networks.sequence.ts_conv1d import TSConv1DRegressor
from straikit.preprocessing.normalization.Standard_Scaler import StandardScaler
from straikit.trainer.trainer import Trainer
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from basicscaler import average_scaler

from metrics import smape, wape, mape
from mv_xy import multivariate_xy

data_file = "elec"
tau = None
n = None
test_start_id = None
WINDOW_SIZE = 64
NORMALIZE = True
NORM_MEHOD = "STD"
SHIFT_SIZE = WINDOW_SIZE
START_OVER = True
batch_size = 32

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

def get_lstm_model(
    units=(64, 64, 64),
):
    model = Sequential()

    for i, unit in enumerate(units):
        if i == 0:
            model.add(
                LSTM(
                    unit,
                    stateful=False,
                    return_sequences=True,
                    batch_input_shape=(batch_size, WINDOW_SIZE, 1),
                )
            )
        else:
            model.add(LSTM(unit, stateful=False, return_sequences=True))

    model.add(Dense(1))

    return model


def iter_predict(model, prediction_step, inputs):
    def _predict_one_step(inputs):
        # Output of call has shape [N, window_size, 1]
        outputs = model(inputs, training=tf.constant(False))

        return outputs[:, -1, 0]  # Output the last step shape [N]

    if prediction_step == 1:
        step_out = _predict_one_step(inputs)  # Shape: [N]
        outputs = [step_out]
    else:
        outputs = []
        for step in range(prediction_step):
            step_out = _predict_one_step(inputs)
            outputs.append(step_out)
            if step < prediction_step - 1:
                inputs = tf.concat(
                    [
                        inputs[:, 1:, :],  # Shape: [N, n_steps - 1, 1]
                        tf.reshape(step_out, [-1, 1, 1]),  # Shape: [N, 1, 1]
                    ],
                    axis=1,
                )  # Shape: [N, n_steps, 1]
        # outputs is a list of len self.prediction_step of tensors of
        # shape [N]
    return tf.stack(outputs, axis=1)  # Shape [N, self.prediction_step]


lstm_parmas = {
    "units": (64, 64, 64),
}

trainer_param = {
    "batch_size": 64,
    "train_dir": f"./1dcnn_save_model/",
    "nb_epochs": 8,
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

# make sure all batches have the same size
mod1 = train_x.shape[0] % batch_size
if mod1 != 0:
    train_x = train_x[:-mod1, ...]
    train_y = train_y[:-mod1, ...]
print("Train and test data prepared ++++++++++++++++++++++++++++++++++")
estimator = get_lstm_model(**lstm_parmas)
estimator.compile(loss="mae", optimizer="adam")

estimator.summary()

estimator.fit(train_x, train_y, batch_size=batch_size, epochs=trainer_param['nb_epochs'])
print(f"------------Finished training--------------------")


# model predict
total_y_pred = []
for tx, ty in zip(test_x, test_y):
    pred = iter_predict(estimator, tau, tx)
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

with open(data_file+'_result_lstm.dat','w') as outfile:
    outfile.write(f"WAPE score: {wape_score}, MAPE Score: {mape_score}, SMAPE score: {smape_score}\n")
    outfile.write("Scores using shifted values: \n")
    outfile.write(f"WAPE score: {s_wape_score}, MAPE score: {s_mape_score}, SMAPE score: {s_smape_score}")
