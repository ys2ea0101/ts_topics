import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from straikit.eval.metrics.generate_metrics import generate_time_series_metrics
from straikit.preprocessing.manipulation.ts_dataset import TSWindowedXY
from straikit.supervized.forecasting.prediction_loop import (
    multi_step_post_predict
)

from mv_xy import multivariate_xy
from seq2seq_model import get_seq_model, seq2seq_pred

window_size = 32
DO_SCALE = True
USE_EXOG = True

# toy example with random inputs and outputs
def random_dataset(shape1, shape2, shape3):
    ts = np.random.normal(size=shape1)
    exog = np.random.normal(size=shape2)
    y = np.random.normal(size=shape3)

    return ts, exog, y


conv_params = {
    "kernel_size": (16, 16),
    "filter_size": (8, 8),
}

dense_param = {
    "hidden_size": (8, 1),
    "dropout": 0,
}


def test_random():
    seq2seq = get_seq_model(conv_params, dense_param, 32, 2, 4, 3)
    seq2seq.compile(loss="MAE")
    seq2seq.summary()

    # random numbers to test if is running
    train_ts, train_exog, train_y = random_dataset(
        (1000, 32, 3), (1000, 3, 4), (1000, 3, 1)
    )
    seq2seq.fit([train_ts, train_exog], train_y, batch_size=32, epochs=10)


def test_sin():
    seq2seq = get_seq_model(conv_params, dense_param, window_size, 0, 4, 3)
    seq2seq.compile(loss="MAE", optimizer="adam")
    seq2seq.summary()

    # sine curve for basic debugging
    L = 3000
    series = np.array([np.sin(i * 2 * np.pi / 12) for i in range(L)])

    ltrain = int(len(series) * 0.8)
    train_data = series[:ltrain]
    test_data = series[ltrain:]
    train_feature = None
    test_feature = None

    xy = TSWindowedXY(window_size=32, use_feature=False, prediction_step=3, direct=True)
    train_data = xy.fit_transform(X=train_feature, y=train_data)

    train_x = np.array([x[0] for x in train_data["X"]])
    train_y = np.array([y[0] for y in train_data["y"]])

    print(train_x.shape, train_y.shape)

    exog = np.random.normal(size=(train_x.shape[0], 3, 4))

    seq2seq.fit([train_x, exog], train_y[:, :, np.newaxis], batch_size=32, epochs=32)

    test_in = xy.inf_transform(X=test_feature, y=test_data)
    test_x = np.array([x[0] for x in test_in["X"]])
    test_oxog = np.random.normal(size=(test_x.shape[0], 3, 4))
    ot = seq2seq_pred(seq2seq, test_x, test_oxog)

    fout = multi_step_post_predict(preds=ot, pred_step=3, original_data=test_data)
    plt.plot(fout, label="pred")
    plt.plot(test_data, label="true")
    plt.legend()
    plt.show()


def test_random_ts_with_almighty_feature():
    seq2seq = get_seq_model(conv_params, dense_param, window_size, 0, 1, 3)
    seq2seq.compile(loss="MAE", optimizer="adam")
    seq2seq.summary()

    # random ts
    L = 3000
    series = np.random.normal(size=(L,))

    ltrain = int(len(series) * 0.8)
    train_data = series[:ltrain]
    test_data = series[ltrain:]
    train_feature = None
    test_feature = None

    xy = TSWindowedXY(
        window_size=window_size, use_feature=False, prediction_step=3, direct=True
    )
    train_data = xy.fit_transform(X=train_feature, y=train_data)

    train_x = np.array([x[0] for x in train_data["X"]])
    train_y = np.array([y[0] for y in train_data["y"]])

    print(train_x.shape, train_y.shape)

    exog = train_y

    seq2seq.fit([train_x, exog], train_y[:, :, np.newaxis], batch_size=32, epochs=32)

    test_in = xy.transform(X=test_feature, y=test_data)
    test_x = np.array([x[0] for x in test_in["X"]])
    test_oxog = np.array([x[0] for x in test_in["y"]])
    ot = seq2seq_pred(seq2seq, test_x, test_oxog)

    fout = multi_step_post_predict(
        preds=ot, pred_step=3, original_data=test_data, cut_origin_step=2
    )
    plt.plot(fout[-100:], label="pred")
    plt.plot(test_data[-100:], label="true")
    plt.legend()
    plt.show()


def bench_mark_univariate(data_file):
    pred_step = 7
    if data_file == "bank":
        data = pd.read_csv(
            "~/projects/ts_dataset/client/Bank_Customer_Walkin_Forecasting_Mod.csv",
        )
        feature = data[["temperature", "weather", "workingday", "holiday", "season"]]
        series = data[["Bank Customer Walkin count"]]
        feature = feature.values
        series = series.to_numpy()
        #series = np.array([np.sin(i * 2 * np.pi / 24) for i in range(len(series))])

    elif data_file == "bike":
        data = pd.read_csv(
            "/home/yifeis/projects/ts_dataset/CB_MULTI_FORECASTING_K03_BikeSharingDemand.csv",
        )
        feature = data[["humidity", "windspeed", "weather", "temp", "holiday"]]
        series = data["count"]
        feature = feature.values
        series = series.to_numpy()

    elif data_file == "demand":
        data = pd.read_csv("/home/yifeis/projects/ts_dataset/client/20year_feature.csv")
        feature = data[["lwk", "christmas"]]
        series = data[["total_demand"]]
        feature = feature.values
        series = series.to_numpy()

    else:
        raise ValueError("Data file unknown!!")

    sscaler = StandardScaler()
    fscaler = StandardScaler()

    if DO_SCALE:
        series = sscaler.fit_transform(series)
        feature = fscaler.fit_transform(feature)

    feature = feature[np.newaxis, :, :]
    series = series[np.newaxis, :]

    ltest = int(series.shape[1] * 0.8)

    xy = multivariate_xy(
        window_size=window_size,
        h_shift=1,
        test_start_id=ltest,
        test_tau=pred_step,
        test_n=7,
        direct=True,
    )
    x, y, exog = xy.train_xy(series, feature)

    print(x.shape, y.shape, exog.shape)

    seq2seq = get_seq_model(conv_params, dense_param, window_size, 0, feature.shape[2], pred_step, use_exog=USE_EXOG)
    seq2seq.compile(loss="MAE", optimizer="adam")
    seq2seq.summary()

    seq2seq.fit([x, exog], y, batch_size=32, epochs=64)

    test_x, test_y, test_exog = xy.train_xy(series, feature, start_id=ltest, end_id=series.shape[1])
    pred = seq2seq_pred(seq2seq, test_x, test_exog)
    print(pred.shape)
    fout = multi_step_post_predict(
        preds=pred, pred_step=pred_step, original_data=series[0, ltest:], cut_origin_step=pred_step
    )

    plt.plot(fout[-100:], label="pred")
    plt.plot(series[0, -100:], label="true")
    plt.show()

    metrics = generate_time_series_metrics(
        y_pred=fout, y_true=series[0, ltest:], metrics={"mse", "smape", "msse", "mae"}
    )

    print(metrics.values[0,0])


# test_random_ts_with_almighty_feature()
bench_mark_univariate("demand")
