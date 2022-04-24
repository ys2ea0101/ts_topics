from mv_xy_edit import multivariate_xy

import numpy as np


print('-----------test iter case--------------')
a = np.arange(0, 8.8, 1.1)

Y = np.array([a + i for i in range(5)])

print("Input time series: ", Y)

exog_e = np.arange(0.03, 4.03, 0.5).reshape((1, -1))
exog_e = np.tile(exog_e, [5, 1]) + np.arange(5).reshape((-1, 1))
exog_e = exog_e[:, :, np.newaxis]
print("Input exog for encoder: ", exog_e[:,:,0])

exog_d = exog_e - 0.5
print("Input exog for decoder: ", exog_d[:, :, 0])
# params: (window_size, h_shift, test_start_id, test_tau, test_n, direct=False)
dl = multivariate_xy(3, 2, 9, 3, 5)

x, y = dl.train_xy_old(Y)
x1, y1, eeo1, edo1 = dl.train_xy(Y, exog_e=exog_e, exog_d=exog_d)
print("x: ", x[:, :, 0])
print("x1: ", x1.squeeze())
print("y: ", y[:, :, 0])
print("y1: ", y1.squeeze())
print("exog e out 1: ", eeo1.squeeze())
# tx, ty = dl.rolling_test_xy(Y)
#
# print("# test: ", len(tx))
# print("test in 1: ", tx[0])
# print("test true 1: ", ty[0])

print('-----------test direct case--------------')

a = np.arange(0, 11, 1.1)

Y = np.array([a + i for i in range(5)])

print("Input series: ", Y)
# (5, 10, 7) exog_e, and (5, 10, 3) exog_d

exog_e = np.arange(0.01, 7.01, 0.02).reshape((5, 10, 7))
exog_d = np.arange(0.01, 3.01, 0.02).reshape((5, 10, 3))

pick_e_dim = 4
pick_d_dim = 2
print("Selected exog for encoder: ", exog_e[:,:,pick_e_dim])
print("Selected exog for decoder: ", exog_d[:,:,pick_d_dim])

# direct case
dl1 = multivariate_xy(3, 2, 10, 2, 1, direct=True)
x, y, ed1out = dl1.train_xy_old(Y, exog_d=exog_d)
x1, y1, exog_e_out, exog_d_out = dl1.train_xy(Y, exog_e=exog_e, exog_d=exog_d)

print("x: ", x.squeeze())
print("y: ", y.squeeze())
print("x1: ", x1.squeeze())
print("y1: ", y1.squeeze())
print("x, y shapes: ", x.shape, y.shape, x1.shape, y1.shape)
print("selected exog_e_out: ", exog_e_out[:, :, pick_e_dim])
print("selected exog_d_out: ", exog_d_out[:, :, pick_d_dim])
print("selected exog_d_out1: ", ed1out[:, :, pick_d_dim])

# --- take out the 3rd exog_e, and exog_d, check if correct
print("exog_d input: ", exog_d[2, :, :])
print("exog_out 1: ", exog_d_out[2, :, :])
print("exog_out 2: ", ed1out[2, :, :])


# -------- test rolling test --------
window_size = 60
h_shift = 4
test_tau = 4
test_n = 15
test_start_id = 143 - test_n * test_tau

Y = np.arange(143).reshape((1, -1))
Y = np.tile(Y, [7, 1])

#  (window_size, h_shift, test_start_id, test_tau, test_n, direct=False)
txy = multivariate_xy(window_size, h_shift, 143 - test_tau * test_n, test_tau, test_n)

txi, txt, teei, tedi = txy.rolling_test_xy(Y)
txi0, txt0 = txy.rolling_test_xy_old(Y)
print("----------test rolling test output-----------------")
print("rolling test x input: ", txi[-1])
print("rolling test x true: ", txt[-1])
print("rolling test 0 x input: ", txi0[-1].squeeze())
print("rolling test 0 x true: ", txt0[-1].squeeze())
