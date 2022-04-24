import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from basicscaler import BasicScaler
from metrics import smape, wape
from mv_xy_edit import multivariate_xy
from seq2seq_model import get_seq_model, get_tcn_seq_model, seq2seq_pred
from tensorflow.keras import regularizers

# ------------------- param set up ----------------
WINDOW_SIZE = 60
tau = 4
n = 15
DO_SCALE = True
test_start = 143 - tau * n

conv_params = {
    "kernel_size": (18, 18, 18),
    "filter_size": (8, 8, 8),
    # "dropout": 0.1,
    # "l2": 0.1,
}

dense_param = {
    "hidden_size": (16, 1),
    "dropout": 0,
    "l2": 0.01,
}

tcn_params = {
    #"input_shape": (WINDOW_SIZE, 1),
    "dilations": (1, 2, 4, 8),
    "nb_filters": 16,
    "return_sequences": True,
}

# ------------   load data   -----------------
ts = np.load("retail_nparray_ts.npy")
exog = np.load("retail_nparray_exog.npy", allow_pickle=True)

# add original ts shifted, for debugging purpose
# shift = tau + 1
# trivial_exog = np.concatenate((ts[:, shift: , np.newaxis], np.zeros_like(ts[:, :shift, np.newaxis])), axis=1)
# exog = np.concatenate((exog, trivial_exog), axis=2)

print(f"shape of time sereis: {ts.shape}")
print(f"shape of exog data: {exog.shape}")

# not using datetime
# exog = exog[:,:,2:]

# ------------ tran and test split ---------------
# using 80% of time series as training, 20 % as test

ids = np.arange(len(ts))
np.random.shuffle(ids)

ltrain = int(len(ids) * 0.8)

train_ids = ids[:ltrain]
test_ids = ids[ltrain:]

train_data = ts[train_ids, :]
train_exog = exog[train_ids, :, :]

test_data = ts[test_ids, :]
test_exog = exog[test_ids, :, :]

# --------------data preparation -------------------

# need separate scalers for train and test,
# since we are using different series for train and test
tsscaler_train = BasicScaler()
exogscaler_train = BasicScaler()

tsscaler_test = BasicScaler()
exogscaler_test = BasicScaler()

if DO_SCALE:
    train_series = tsscaler_train.fit_transform(train_data)
    train_exog = exogscaler_train.fit_transform(train_exog)

    test_data = tsscaler_test.fit_transform(test_data)
    test_exog = exogscaler_test.fit_transform(test_exog)

mxy = multivariate_xy(
    window_size=WINDOW_SIZE,
    h_shift=2,
    test_start_id=test_start,
    test_tau=tau,
    test_n=n,
    direct=True,
)

# split exog data into encoder and decoder
# do it here since the data is already normalized
exog_e_id = [2,3,10,11]
exog_d_id = [5,6,7,8,9]
train_exog_e = train_exog[:,:,exog_e_id]
train_exog_d = train_exog[:,:,exog_d_id]
test_exog_e = test_exog[:,:,exog_e_id]
test_exog_d = test_exog[:,:,exog_d_id]

use_exog_e = (train_exog_e is not None)
use_exog_d = (train_exog_d is not None)
print(f"-------- {use_exog_d}")
train_x, train_y, train_exog_e_in, train_exog_d_in = mxy.train_xy(
    series=train_series, exog_e=train_exog_e, exog_d=train_exog_d, end_id=train_series.shape[1]
)
print(train_x.shape)
print(train_exog_e_in.shape)

if use_exog_e:
    train_x = np.concatenate((train_x, train_exog_e_in), axis=2)
test_x, test_x_true, test_exog_e_in, test_exog_d_in  = mxy.rolling_test_xy(
    series=test_data, exog_e=test_exog_e, exog_d=test_exog_d
)

print(f"Sizes: train_x: {train_x.shape}, train_y: {train_y.shape}, exog: {train_exog_d_in.shape}")

# --------------- model and fit -----------------------
s2s_model = get_seq_model(
    conv_params=conv_params,
    dense_params=dense_param,
    window_size=WINDOW_SIZE,
    #n_ts_feature=0,
    n_ts_feature=train_x.shape[-1]-1 if use_exog_e else 0,
    n_local_feature=train_exog_d.shape[-1] if train_exog_d is not None else 0,
    pred_step=tau,
    use_exog=use_exog_d,
)

# s2s_model = get_tcn_seq_model(
#     tcn_params=tcn_params,
#     dense_params=dense_param,
#     window_size=WINDOW_SIZE,
#     n_ts_feature=0,
#     n_local_feature=train_exog.shape[-1] if train_exog is not None else 0,
#     pred_step=tau,
# )

s2s_model.compile(loss="MAE", optimizer="adam")
s2s_model.summary()

s2s_model.fit([train_x, train_exog_d_in], train_y, batch_size=32, epochs=16)

# --------------- evaluation and plots ------------------------
total_y_pred = []
for tx, tee, ted in zip(test_x, test_exog_e_in, test_exog_d_in):
    if use_exog_e:
        pred = s2s_model([np.concatenate((tx[:,:,np.newaxis], tee), axis=2), ted], training=False)
    else:
        pred = s2s_model([tx[:,:,np.newaxis], ted], training=False)
    total_y_pred.append(pred)

y_pred = np.hstack(tuple(total_y_pred)).squeeze()
y_true = np.hstack(tuple(test_x_true)).squeeze()

y_pred = tsscaler_test.inverse_transform(y_pred)
y_true = tsscaler_test.inverse_transform(y_true)

wape_score = wape(y_pred, y_true)
smape_score = smape(y_pred, y_true)

print(f"WAPE score: {wape_score}, SMAPE score: {smape_score}")

# for i in range(7,20):
#     plt.plot(y_pred[i,:], label="pred")
#     plt.plot(y_true[i,:], label="true")
#     plt.legend()
#     plt.show()
