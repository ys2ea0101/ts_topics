import numpy as np
import pandas as pd


class multivariate_xy:
    def __init__(
        self, window_size, h_shift, test_start_id, test_tau, test_n, direct=False
    ):
        """
        this is data processing for multivariate time series.
        this class assumes data has shape: N * T, exog shape: N * T * nf
        if not direct, it will take i : i + window as training x, and i+1: i+window+1 as training y,
        at then increase i by h_shift

        if direct, it will take i : i + window as training x, and i + window: i+window+pred_step as training y,
        in this case pred_step should be just test_tau

        :param window_size:
        :param h_shift:
        :param test_start_id:
        :param test_tau:
        :param test_n:
        :param direct:
        """
        self.window_size = window_size
        self.h_shift = h_shift
        self.test_start_id = test_start_id
        self.test_tau = test_tau
        self.test_n = test_n
        self.direct = direct

    def train_xy(self, series, exog=None, start_id=0, end_id=None):
        """
        Do note that this operation is extremely inefficient when h_shift is small,
        can be improved using numpy operations

        :param series: time series of shape: N * T, N is number of series, T is length of each series
        :param exog: exog data, shape N * T * f
        :param start_id: start index for training
        :param end_id: end index for training
        :return:
        """
        if end_id is None:
            end_id = self.test_start_id

        if isinstance(series, pd.DataFrame):
            series = np.array(series)
            # when reading from csv, the shape is (T, N), transform to (N, t)
            # notice that this is DANGEROIUS
            series = np.transpose(series)

        # iterative method
        # right now exog data not used here
        if not self.direct:
            current_h = start_id
            x_out = series[:, : self.window_size]
            y_out = series[:, 1 : self.window_size + 1]
            while current_h + self.window_size + 1 <= end_id:
                x_out = np.concatenate(
                    (x_out, series[:, current_h : current_h + self.window_size]), axis=0
                )
                y_out = np.concatenate(
                    (
                        y_out,
                        series[:, current_h + 1 : current_h + self.window_size + 1],
                    ),
                    axis=0,
                )

                current_h += self.h_shift

            return x_out[:, :, np.newaxis], y_out[:, :, np.newaxis]

        # direct method
        # todo
        else:
            current_h = start_id
            x_out = series[:, : self.window_size]
            y_out = series[:, self.window_size : self.window_size + self.test_tau]
            if exog is not None:
                exog_out = exog[
                    :, self.window_size : self.window_size + self.test_tau, :
                ]
            while current_h + self.window_size + self.test_tau <= end_id:
                x_out = np.concatenate(
                    (x_out, series[:, current_h : current_h + self.window_size]), axis=0
                )
                y_out = np.concatenate(
                    (
                        y_out,
                        series[
                            :,
                            current_h
                            + self.window_size : current_h
                            + self.window_size
                            + self.test_tau,
                        ],
                    ),
                    axis=0,
                )
                if exog is not None:
                    exog_out = np.concatenate(
                        (
                            exog_out,
                            exog[
                            :,
                            current_h
                            + self.window_size: current_h
                                                + self.window_size
                                                + self.test_tau,
                            :,
                            ],
                        ),
                        axis=0,
                    )


                current_h += self.h_shift

            # batch_size * window_size * 1, batch_size * tau * 1, batch_size * tau * nf
            if exog is not None:
                return x_out[:, :, np.newaxis], y_out[:, :, np.newaxis], exog_out
            else:
                return x_out[:, :, np.newaxis], y_out[:, :, np.newaxis]

    def rolling_test_xy(self, series, exog=None):
        """
        test are done for test_n times each with test_tau steps,
        this is the same as done in deepglo
        :param series:
        :return:
        """
        if isinstance(series, pd.DataFrame):
            series = np.array(series)
            # when reading from csv, the shape is (T, N), transform to (N, t)
            series = np.transpose(series)

        start_id = self.test_start_id
        test_x_in = []
        test_x_true = []
        if exog is not None:
            test_exog_in = []
        if series.shape[1] < start_id + self.test_n * self.test_tau:
            raise ValueError("Test series not long enough to perform all the test!")
        for i in range(self.test_n):
            test_x_in.append(
                series[:, start_id - self.window_size : start_id, np.newaxis]
            )
            test_x_true.append(
                series[:, start_id : start_id + self.test_tau, np.newaxis]
            )
            if exog is not None:
                test_exog_in.append(exog[:, start_id : start_id + self.test_tau, :])
            start_id += self.test_tau

        if exog is None:
            return test_x_in, test_x_true
        else:
            return test_x_in, test_x_true, test_exog_in


if __name__ == "__main__":
    a = np.arange(0, 44, 1.1)

    Y = np.array([a + i for i in range(9)])

    print(Y.shape)

    dl = multivariate_xy(3,3, 10, 3, 5)

    x, y = dl.train_xy(Y)
    print(x[0, :, 0], y[0, :, 0])

    tx, ty = dl.rolling_test_xy(Y)

    print("# test: ", len(tx))
    print("test in 1: ", tx[0])
    print("test true 1: ", ty[0])
