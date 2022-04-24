import numpy as np
import pandas as pd

class BasicScaler:
    def __init__(self):
        self.trained = False
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray, end_index: int = None):
        """
        X can be shape N * T or N * T * f
        :param X:
        :param end_index:
        :return:
        """

        if X.ndim == 2:
            if end_index:
                self.mean = np.mean(X[:, :end_index], axis=1, keepdims=True)
                self.std = np.std(X[:, :end_index], axis=1, keepdims=True)
            else:
                self.mean = np.mean(X, axis=1, keepdims=True)
                self.std = np.std(X, axis=1, keepdims=True)
        elif X.ndim == 3:
            if end_index:
                self.mean = np.mean(X[:, :end_index, :], axis=1, keepdims=True)
                self.std = np.std(X[:, :end_index, :], axis=1, keepdims=True)
            else:
                self.mean = np.mean(X, axis=1, keepdims=True)
                self.std = np.std(X, axis=1, keepdims=True)
        self.trained = True

    def transform(self, X):
        if not self.trained:
            raise ValueError("Not trained yet")

        return (X - self.mean) / (self.std+0.00001)

    def fit_transform(self, X, end_index: int = None):
        self.fit(X, end_index)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.std + self.mean


class average_scaler:
    """
    takes in pandas df
    """
    def __init__(self, shift=0. ):
        self.scale = {}
        self.shift = shift

    def fit(self, X: pd.DataFrame):
        for col in list(X):
            self.scale[col] = X[col].mean() + self.shift

    def transform(self, X: pd.DataFrame):
        if not self.scale:
            raise ValueError("Model not fitted when transforming!!  ")
        for col in list(X):
            X[col] = X[col] / self.scale[col]
        return X

    def inverse_transform(self, X: pd.DataFrame):
        if not self.scale:
            raise ValueError("Model not fitted when inversing!!  ")
        for col in X.columns:
            X[col] = X[col] * self.scale[col]
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    ts = np.arange(48).reshape((4, 12))
    exog = np.arange(144).reshape((4,12,3))

    ss = BasicScaler()
    es = BasicScaler()
    ts_scaled = ss.fit_transform(ts)
    exog_scaled = es.fit_transform(exog)

    ts_back = ss.inverse_transform(ts_scaled)
    exog_back = es.inverse_transform(exog_scaled)

    print(f"ts_scaled: {ts_scaled}")
    print(f"exog_scaled: {exog_scaled}")

    print(f"Diff ts: {ts_back - ts}")
    print(f"Diff exog: {exog_back - exog}")

    ts = pd.DataFrame(ts)
    print(f"Init df: {ts}")
    avs = average_scaler()
    ts_scaled = avs.fit_transform(ts)
    print(ts_scaled)
    ts_back = avs.inverse_transform(ts_scaled)

    print(f"ts_scaled: {ts_scaled}")
    print(f"Diff ts: {ts_back - ts}")

