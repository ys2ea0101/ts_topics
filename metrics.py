import numpy as np

# def wape(y_pred, y_true):
#     diff = np.abs(y_pred-y_true)
#     norm = np.abs(y_true)
#
#     return np.sum(diff) / np.sum(norm)
#
def smape(y_pred, y_true):
    nz = np.where(y_true > 0)
    y_pred = y_pred[nz]
    y_true = y_true[nz]
    norm = np.abs(y_pred) + np.abs(y_true)
    diff = 2 * np.abs(y_pred - y_true)
    return np.mean(diff / norm)

# These are directionly copied from DeepGLO
# def smape(P, A):
#     nz = np.where(A > 0)
#     Pz = P[nz]
#     Az = A[nz]
#
#     return np.mean(2 * np.abs(Az - Pz) / (np.abs(Az) + np.abs(Pz)))


def mape(P, A):
    nz = np.where(A > 0.01)
    Pz = P[nz]
    Az = A[nz]

    return np.mean(np.abs(Az - Pz) / np.abs(Az))


def wape(P, A):
    return np.mean(np.abs(A - P)) / np.mean(np.abs(A))
