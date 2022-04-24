import numpy as np
import pandas as pd


class multivariate_xy:
    def __init__(
        self, window_size, h_shift, test_start_id, test_tau, test_n, direct=False
    ):
        """
        this is data processing for multivariate time series.
        this class assumes data has shape: N * T,
        exog data shape: N * T * nf, exog data can have 2 parts, one to be added in the encoder,
        one to be added in the decoder,
        if not direct, it will take i : i + window as training x, and i+1: i+window+1 as training y,
        at then increase i by h_shift

        if direct, it will take i : i + window as training x, and i + window: i+window+pred_step as training y,
        in this case pred_step should be just test_tau

        :param window_size:
        :param h_shift:
        :param test_start_id:
        :param test_tau: in the case of direct method, it also acts as the prediction step
        :param test_n:
        :param direct:
        """
        self.window_size = window_size
        self.h_shift = h_shift
        self.test_start_id = test_start_id
        self.test_tau = test_tau
        self.test_n = test_n
        self.direct = direct
        if self.test_start_id is not None and self.test_start_id < self.window_size:
            raise ValueError("test starts too early, not enough space for one window")

    @staticmethod
    def _arrange_indices(h_shift, window_size, N, T, f=None):
        """
        create indices for the output, the output is presented as output[indices1, indices2, indices3(if f > 1)]
        takes out window_size samples, then move h_shift and do the same, util the end is reached
        Args:
            h_shift:
            window_size:
            N, T, f: input shape either N x T, or N x T x f

        Returns: indices1, indices2, (if f > 1) indices3

        """
        if window_size > T:
            raise ValueError("series too short for window size!!")

        # batches for 1 series
        n_batch = (T - window_size) // h_shift + 1
        # [0, 1, 2, .. w-1], size (1, window_size)
        indices1 = np.zeros(shape=(N * n_batch, window_size), dtype=int)

        tmp = np.arange(N).reshape(1, -1)
        tmp = np.tile(tmp, [n_batch, 1])
        tmp = tmp.reshape((-1, 1))

        indices1 = indices1 + tmp
        # print(indices1)

        # (1, window_size)
        indices2 = np.arange(window_size).reshape(1, -1)
        # (N, window_size)
        indices2 = np.tile(indices2, [N, 1])
        # (N * batch_size, window_size)
        indices2 = np.tile(indices2, [n_batch, 1])

        tmp = np.arange(n_batch).reshape(-1, 1)
        tmp = np.tile(tmp, [1, N])
        tmp = tmp.reshape(-1, 1)
        indices2 = indices2 + tmp * h_shift
        # print('3: ', indices2)

        if f is None:
            return indices1, indices2
        else:
            # in this case where input is three dimensional,
            # the third dimension is a trivial loop over
            indices3 = np.zeros_like(indices1, dtype=int)[:, :, np.newaxis]
            tmp = np.arange(f).reshape((1, 1, -1))
            indices3 = indices3 + tmp
            indices1 = np.tile(indices1[:, :, np.newaxis], [1, 1, f])
            indices2 = np.tile(indices2[:, :, np.newaxis], [1, 1, f])

            return indices1, indices2, indices3

    def train_xy(self, series, exog_e=None, exog_d=None, start_id=0, end_id=None):
        """
        :param series: time series of shape: N * T, N is number of series, T is length of each series
        :param exog_e: exog data for encoder, shape N * T * fe
        :param exog_d: exog data for decoder, shape N * T * fd
        :param start_id: start index for training, this is basically to overwrite self.test_start_id if needed
        :param end_id: end index for training
        :return:
            x_out, (N * n_batch, window size)
            y_out, iter: (N * n_batch, window size), direct: (N * n_batch, test_tau)
            exog_e_out, (N * nbatch, window size, f_e)
            exog_d_out, (N * nbatch, test_tau, f_d)
        """
        if end_id is None:
            end_id = self.test_start_id

        if isinstance(series, pd.DataFrame):
            series = np.array(series)
            # when reading from csv, the shape is (T, N), transform to (N, t)
            # notice that this is DANGEROIUS
            series = np.transpose(series)

        N = series.shape[0]
        T = series.shape[1]

        exog_e_out = None
        exog_d_out = None
        # iterative method
        if not self.direct:
            id1, id2 = self._arrange_indices(self.h_shift, self.window_size, N, T - 1)
            x_out = series[:, start_id : end_id - 1, ...][id1, id2]
            y_out = series[:, start_id + 1 : end_id, ...][id1, id2]

            if exog_e is not None:
                eid1, eid2, eid3 = self._arrange_indices(
                    self.h_shift, self.window_size, N, T - 1, f=exog_e.shape[-1]
                )
                exog_e_out = exog_e[eid1, eid2, eid3]

            if exog_d is not None:
                print("Warning: exogenous data can not be utilized in the decoder in iterative method")

        else:
            # direct case
            id1, id2 = self._arrange_indices(
                self.h_shift, self.window_size, N, T - self.test_tau
            )
            x_out = series[id1, id2]
            # y_out of direct case can be calculated using a trick,
            # by manipulating the indices and start positions,
            # start from window_size, then add window_size to all id2
            id1, id2 = self._arrange_indices(
                self.h_shift, self.test_tau, N, T - self.window_size
            )
            id2 = id2 + np.ones_like(id2, dtype=int) * self.window_size
            y_out = series[id1, id2]

            if exog_e is not None:
                eid1, eid2, eid3 = self._arrange_indices(
                    self.h_shift,
                    self.window_size,
                    N,
                    T - self.test_tau,
                    f=exog_e.shape[-1],
                )
                exog_e_out = exog_e[eid1, eid2, eid3]

            if exog_d is not None:
                # oxog_d is done with the same trick as y_out
                did1, did2, did3 = self._arrange_indices(
                    self.h_shift,
                    self.test_tau,
                    N,
                    T - self.window_size,
                    f=exog_d.shape[-1],
                )
                did2 = did2 + np.ones_like(did2, dtype=int) * self.window_size
                exog_d_out = exog_d[did1, did2, did3]

        return x_out[:, :, np.newaxis], y_out[:, :, np.newaxis], exog_e_out, exog_d_out

    def train_xy_old(self, series, exog_e=None, exog_d=None, start_id=0, end_id=None):
        """
        Do note that this operation is extremely inefficient when h_shift is small,
        can be improved using numpy operations

        :param series: time series of shape: N * T, N is number of series, T is length of each series
        :param exog_e: exog data for encoder, shape N * T * fe
        :param exog_d: exog data for decoder, shape N * T * fd
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
            current_h += self.h_shift
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
        else:
            current_h = start_id
            x_out = series[:, : self.window_size]
            y_out = series[:, self.window_size : self.window_size + self.test_tau]
            current_h += self.h_shift
            if exog_d is not None:
                exog_d_out = exog_d[
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
                if exog_d is not None:
                    exog_d_out = np.concatenate(
                        (
                            exog_d_out,
                            exog_d[
                                :,
                                current_h
                                + self.window_size : current_h
                                + self.window_size
                                + self.test_tau,
                                :,
                            ],
                        ),
                        axis=0,
                    )

                current_h += self.h_shift

            # batch_size * window_size * 1, batch_size * tau * 1, batch_size * tau * nf
            if exog_d is not None:
                return x_out[:, :, np.newaxis], y_out[:, :, np.newaxis], exog_d_out
            else:
                return x_out[:, :, np.newaxis], y_out[:, :, np.newaxis]

    def rolling_test_xy_old(self, series, exog_e=None, exog_d=None):
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
        if exog_d is not None:
            test_exog_d_in = []
        if series.shape[1] < start_id + self.test_n * self.test_tau:
            raise ValueError("Test series not long enough to perform all the test!")
        for i in range(self.test_n):
            test_x_in.append(
                series[:, start_id - self.window_size : start_id, np.newaxis]
            )
            test_x_true.append(
                series[:, start_id : start_id + self.test_tau, np.newaxis]
            )
            if exog_d is not None:
                test_exog_d_in.append(exog_d[:, start_id : start_id + self.test_tau, :])
            start_id += self.test_tau

        if exog_d is None:
            return test_x_in, test_x_true
        else:
            return test_x_in, test_x_true, test_exog_d_in

    def rolling_test_xy(self, series, exog_e=None, exog_d=None):
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

        start_id = (
            self.test_start_id if self.test_start_id is not None else self.window_size
        )
        test_x_in = []
        test_x_true = []
        test_exog_e_in = None if exog_e is None else []
        test_exog_d_in = None if exog_d is None else []

        if series.shape[1] < start_id + self.test_n * self.test_tau:
            raise ValueError("Test series not long enough to perform all the test!")

        # no difference in the case of direct or not
        N = series.shape[0]
        # this will only take out one batch
        xid1, xid2 = self._arrange_indices(
            self.window_size, self.window_size, N, self.window_size
        )
        yid1, yid2 = self._arrange_indices(
            self.window_size, self.test_tau, N, self.window_size
        )

        if exog_e is not None:
            eeid1, eeid2, eeid3 = self._arrange_indices(
                self.window_size,
                self.window_size,
                N,
                self.window_size,
                f=exog_e.shape[-1],
            )
        if exog_d is not None:
            edid1, edid2, edid3 = self._arrange_indices(
                self.window_size, self.test_tau, N, self.window_size, f=exog_d.shape[-1]
            )
        for i in range(self.test_n):
            test_x_in.append(
                series[
                    xid1,
                    xid2
                    + np.ones_like(xid2, dtype=int)
                    * (start_id - self.window_size + i * self.test_tau),
                ]
            )
            test_x_true.append(
                series[
                    yid1,
                    yid2
                    + np.ones_like(yid2, dtype=int) * (start_id + i * self.test_tau),
                ]
            )

            if exog_e is not None:
                test_exog_e_in.append(
                    exog_e[
                        eeid1,
                        eeid2
                        + np.ones_like(eeid2, dtype=int)
                        * (start_id - self.window_size + i * self.test_tau),
                        eeid3,
                    ]
                )

            if exog_d is not None:
                test_exog_d_in.append(
                    exog_d[
                        edid1,
                        edid2
                        + np.ones_like(edid2, dtype=int)
                        * (start_id + i * self.test_tau),
                        edid3,
                    ]
                )

        return test_x_in, test_x_true, test_exog_e_in, test_exog_d_in
